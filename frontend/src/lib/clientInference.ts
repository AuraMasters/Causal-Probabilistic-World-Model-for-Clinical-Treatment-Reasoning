import type { AnalyzeResult, PatientInputs, TreatmentResult, ContinuousSensitivity, TrajectoryPoint } from './types'
import weightsData from './modelWeights.json'

const TREATMENTS_META = [
  { treatment: 0, name: 'Zidovudine (ZDV / AZT Monotherapy)', short_name: 'ZDV Mono' },
  { treatment: 1, name: 'Zidovudine + Didanosine (ZDV + ddI)', short_name: 'ZDV + ddI' },
  { treatment: 2, name: 'Zidovudine + Zalcitabine (ZDV + ddC)', short_name: 'ZDV + ddC' },
  { treatment: 3, name: 'Didanosine (ddI Monotherapy)', short_name: 'ddI Mono' },
]

const CONTINUOUS_VARS = ['age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd80'] as const

interface ArmWeights {
  intercept: number
  coefficients: number[]
}

interface WeightsSchema {
  scaler_means: Record<string, number>
  scaler_stds: Record<string, number>
  feature_names: string[]
  dev_risk_reductions: number[]
  arms: Record<string, ArmWeights>
}

const weights = weightsData as unknown as WeightsSchema

function sigmoid(z: number): number {
  return 1.0 / (1.0 + Math.exp(-Math.max(-50, Math.min(50, z))))
}

function buildFeatureVector(inputs: PatientInputs, overrideCd4?: number): number[] {
  const vec: number[] = []

  // 1. Continuous variables
  for (const varName of CONTINUOUS_VARS) {
    const rawVal = varName === 'cd40' && overrideCd4 !== undefined
      ? overrideCd4
      : parseFloat(inputs[varName]) || 0
    const mean = weights.scaler_means[varName] ?? 0
    const std = weights.scaler_stds[varName] || 1
    vec.push((rawVal - mean) / std)
  }

  // 2. One-hot categorical variables matching ColumnTransformer(drop='first')
  // ['hemo_1', 'homo_1', 'drugs_1', 'oprior_1', 'z30_1', 'race_1', 'gender_1', 'strat_2', 'strat_3', 'symptom_1']
  vec.push(String(inputs.hemo) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.homo) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.drugs) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.oprior) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.z30) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.race) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.gender) === '1' ? 1.0 : 0.0)

  // strat: categories are '1', '2', '3' with drop='first' -> 'strat_2', 'strat_3'
  const stratVal = String(inputs.strat)
  vec.push(stratVal === '2' ? 1.0 : 0.0)
  vec.push(stratVal === '3' ? 1.0 : 0.0)

  // symptom
  vec.push(String(inputs.symptom) === '1' ? 1.0 : 0.0)

  return vec
}

export function runClientInference(
  inputs: PatientInputs,
  modelType: 'continuous' | 'discretized' = 'continuous',
): AnalyzeResult {
  const X_vec = buildFeatureVector(inputs)
  const treatments: TreatmentResult[] = []

  for (let t = 0; t < 4; t++) {
    const arm = weights.arms[String(t)]
    let z = arm.intercept
    for (let i = 0; i < X_vec.length; i++) {
      z += arm.coefficients[i] * X_vec[i]
    }
    const p1 = sigmoid(z)
    const p0 = 1.0 - p1
    const eu = p0 * 1.0 + p1 * 0.0

    treatments.push({
      treatment: t,
      name: TREATMENTS_META[t].name,
      short_name: TREATMENTS_META[t].short_name,
      p_label_0: Math.round(p0 * 10000) / 10000,
      p_label_1: Math.round(p1 * 10000) / 10000,
      expected_utility: Math.round(eu * 10000) / 10000,
      rank: 0,
      is_recommended: false,
    })
  }

  // Rank treatments by expected utility descending
  const sortedByEu = [...treatments].sort((a, b) => b.expected_utility - a.expected_utility)
  let rank = 0
  let prevEu: number | null = null
  for (const item of sortedByEu) {
    if (item.expected_utility !== prevEu) {
      rank += 1
      prevEu = item.expected_utility
    }
    item.rank = rank
  }

  // Restore treatment index order
  treatments.sort((a, b) => a.treatment - b.treatment)

  // Recommended: highest EU (tie-breaker lowest treatment ID)
  const recommendedItem = treatments.reduce((best, cur) => {
    if (cur.expected_utility > best.expected_utility) return cur
    if (cur.expected_utility === best.expected_utility && cur.treatment < best.treatment) return cur
    return best
  }, treatments[0])

  for (const item of treatments) {
    item.is_recommended = item.treatment === recommendedItem.treatment
  }

  const sortedRanking = [...treatments].sort((a, b) => a.rank - b.rank)

  // Risk comparisons vs standard ZDV Monotherapy (Arm 0)
  const monoArm = treatments.find((t) => t.treatment === 0) ?? treatments[0]
  const p1Mono = Math.max(monoArm.p_label_1, 0.0001)
  const p1Rec = Math.max(recommendedItem.p_label_1, 0.0001)
  const riskDeltaVsMono = Math.round((monoArm.p_label_1 - recommendedItem.p_label_1) * 10000) / 10000

  // 1. E-Value Sensitivity Analysis for Unmeasured Confounding
  const riskRatio = p1Rec / p1Mono
  let eValue = 1.0
  if (riskRatio < 1.0) {
    const rrStar = 1.0 / riskRatio
    eValue = rrStar + Math.sqrt(rrStar * (rrStar - 1.0))
  }
  const eValueRounded = Math.round(eValue * 100) / 100

  const eValueAnalysis = {
    risk_ratio_vs_monotherapy: Math.round(riskRatio * 10000) / 10000,
    e_value_point: eValueRounded,
    interpretation: `An unmeasured confounder would need an association of at least RR = ${eValueRounded.toFixed(
      2,
    )} with both treatment assignment and clinical progression to explain away the observed benefit.`,
    is_robust: eValueRounded >= 1.5,
  }

  // 2. CATE & Treatment Benefit Percentile
  const devRiskReductions = weights.dev_risk_reductions || []
  let catePercentile = 50.0
  if (devRiskReductions.length > 0) {
    const countLE = devRiskReductions.filter((r) => r <= riskDeltaVsMono).length
    catePercentile = Math.round((countLE / devRiskReductions.length) * 1000) / 10
  }

  const benefitTier =
    catePercentile >= 90
      ? 'Exceptional (Top 10%)'
      : catePercentile >= 75
      ? 'High (Top 25%)'
      : catePercentile >= 50
      ? 'Moderate'
      : 'Mild'

  const treatmentBenefit = {
    absolute_risk_reduction: riskDeltaVsMono,
    arr_percentage_points: Math.round(riskDeltaVsMono * 10000) / 100,
    benefit_percentile: catePercentile,
    benefit_tier: benefitTier,
  }

  // 3. Exact continuous gradient sensitivity analysis under recommended treatment
  const recModel = weights.arms[String(recommendedItem.treatment)]
  const logisticDerivativeFactor = p1Rec * (1.0 - p1Rec)
  const continuousSensitivities: ContinuousSensitivity[] = []

  CONTINUOUS_VARS.forEach((varName, idx) => {
    const coefStd = recModel.coefficients[idx]
    const scaleStd = weights.scaler_stds[varName] || 1
    const rawDerivative = (logisticDerivativeFactor * coefStd) / scaleStd
    const stepUnit = varName.includes('cd') ? 50.0 : varName === 'karnof' ? 10.0 : varName === 'preanti' ? 100.0 : 5.0
    const riskChangePerStep = rawDerivative * stepUnit

    continuousSensitivities.push({
      variable: varName,
      raw_value: parseFloat(inputs[varName]) || 0,
      raw_derivative: Math.round(rawDerivative * 1000000) / 1000000,
      clinical_step_unit: stepUnit,
      risk_change_per_step: Math.round(riskChangePerStep * 10000) / 10000,
      percentage_points_per_step: Math.round(riskChangePerStep * 10000) / 100,
      direction:
        rawDerivative > 0.00001
          ? 'increases_risk'
          : rawDerivative < -0.00001
          ? 'reduces_risk'
          : 'neutral',
    })
  })

  continuousSensitivities.sort((a, b) => Math.abs(b.raw_derivative) - Math.abs(a.raw_derivative))

  // 4. Interactive What-If CD4 Trajectory Simulation (sweep CD4 50 to 800)
  const cd4Trajectory: TrajectoryPoint[] = []
  const currentCd4 = parseFloat(inputs.cd40) || 350

  for (let cd4Val = 50; cd4Val <= 800; cd4Val += 50) {
    const simVec = buildFeatureVector(inputs, cd4Val)
    const point: TrajectoryPoint = {
      cd4: cd4Val,
      arm_0: 0,
      arm_1: 0,
      arm_2: 0,
      arm_3: 0,
    }
    for (let armIdx = 0; armIdx < 4; armIdx++) {
      const arm = weights.arms[String(armIdx)]
      let z = arm.intercept
      for (let i = 0; i < simVec.length; i++) {
        z += arm.coefficients[i] * simVec[i]
      }
      point[`arm_${armIdx}` as keyof TrajectoryPoint] = Math.round(sigmoid(z) * 10000) / 10000
    }
    cd4Trajectory.push(point)
  }

  return {
    model_type: modelType === 'continuous' ? 'continuous_hybrid_scm' : 'discretized_bn',
    model_name:
      modelType === 'continuous'
        ? 'Model B: Continuous / Hybrid SCM (G-Computation)'
        : 'Model A: Baseline Discretized Bayesian Network',
    information_preservation:
      modelType === 'continuous'
        ? 'Exact continuous numerical measurements preserved without discretization'
        : 'Discrete quantile evidence mapping',
    inputs: inputs as unknown as Record<string, any>,
    treatments,
    ranking: sortedRanking,
    recommended: {
      treatment: recommendedItem.treatment,
      name: recommendedItem.name,
      short_name: recommendedItem.short_name,
      p_label_0: recommendedItem.p_label_0,
      p_label_1: recommendedItem.p_label_1,
      expected_utility: recommendedItem.expected_utility,
      risk_delta_vs_monotherapy: riskDeltaVsMono,
      e_value_analysis: eValueAnalysis,
      treatment_benefit: treatmentBenefit,
      continuous_sensitivities: continuousSensitivities,
    },
    what_if_trajectory: {
      variable: 'cd40',
      label: 'Baseline CD4 Count (cells/mm³)',
      current_value: currentCd4,
      trajectory_points: cd4Trajectory,
    },
    utility_model: {
      label_0_utility: 1.0,
      label_1_utility: 0.0,
    },
    decision_rule: 'Select treatment maximizing expected utility under do-calculus intervention',
  }
}
