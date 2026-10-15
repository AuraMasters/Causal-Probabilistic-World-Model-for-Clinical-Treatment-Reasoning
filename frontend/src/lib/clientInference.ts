import type {
  AnalyzeResult,
  ContinuousSensitivity,
  DiscretizationMetadata,
  FeatureAttribution,
  NumericalFeedback,
  PatientInputs,
  TrajectoryPoint,
  TreatmentResult,
} from './types'
import weightsData from './modelWeights.json'
import discretizedData from './discretizedModelData.json'
import { discretizeValue } from './discretize'

const TREATMENTS_META = [
  { treatment: 0, name: 'Zidovudine (ZDV / AZT Monotherapy)', short_name: 'ZDV Mono' },
  { treatment: 1, name: 'Zidovudine + Didanosine (ZDV + ddI)', short_name: 'ZDV + ddI' },
  { treatment: 2, name: 'Zidovudine + Zalcitabine (ZDV + ddC)', short_name: 'ZDV + ddC' },
  { treatment: 3, name: 'Didanosine (ddI Monotherapy)', short_name: 'ddI Mono' },
]

const CONTINUOUS_VARS = ['age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd80'] as const
const CATEGORICAL_VARS = [
  'hemo',
  'homo',
  'drugs',
  'oprior',
  'z30',
  'race',
  'gender',
  'strat',
  'symptom',
] as const

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

interface DiscretizedSchema {
  model_name: string
  representation: string
  variables: Record<string, { method: string; bins?: number; positive_bins?: number; edges: (number | null)[] }>
  cpt_label_1: Record<string, number>
  ablation_no_cd40: Record<string, number>
  ablation_no_z30: Record<string, number>
}

const weights = weightsData as unknown as WeightsSchema
const discData = discretizedData as unknown as DiscretizedSchema

const discMetadata: DiscretizationMetadata = {
  representation: discData.representation || 'sparse',
  n_bins: 3,
  fit_dataset: 'actg175_development',
  variables: discData.variables,
}

function sigmoid(z: number): number {
  return 1.0 / (1.0 + Math.exp(-Math.max(-50, Math.min(50, z))))
}

function computeEValue(riskRatio: number) {
  let eValue = 1.0
  if (riskRatio < 1.0 && riskRatio > 0) {
    const rrStar = 1.0 / riskRatio
    eValue = rrStar + Math.sqrt(rrStar * (rrStar - 1.0))
  }
  const eValueRounded = Math.round(eValue * 100) / 100
  const isRobust = riskRatio < 1.0 && eValueRounded >= 1.5

  const interpretation =
    riskRatio < 1.0
      ? `An unmeasured confounder would need an association of at least RR = ${eValueRounded.toFixed(
          2,
        )} with both treatment assignment and clinical progression to explain away the observed benefit.`
      : `No progression risk reduction observed relative to monotherapy (RR = ${riskRatio.toFixed(
          2,
        )} >= 1.00); unmeasured confounding sensitivity analysis for risk reduction is not applicable.`

  return {
    risk_ratio_vs_monotherapy: Math.round(riskRatio * 10000) / 10000,
    e_value_point: eValueRounded,
    interpretation,
    is_robust: isRobust,
  }
}

function buildFeatureVector(inputs: PatientInputs, overrideCd4?: number): number[] {
  const vec: number[] = []

  // 1. Continuous variables
  for (const varName of CONTINUOUS_VARS) {
    const rawVal =
      varName === 'cd40' && overrideCd4 !== undefined
        ? overrideCd4
        : parseFloat(inputs[varName]) || 0
    const mean = weights.scaler_means[varName] ?? 0
    const std = weights.scaler_stds[varName] || 1
    vec.push((rawVal - mean) / std)
  }

  // 2. One-hot categorical variables matching ColumnTransformer(drop='first')
  vec.push(String(inputs.hemo) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.homo) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.drugs) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.oprior) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.z30) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.race) === '1' ? 1.0 : 0.0)
  vec.push(String(inputs.gender) === '1' ? 1.0 : 0.0)

  const stratVal = String(inputs.strat)
  vec.push(stratVal === '2' ? 1.0 : 0.0)
  vec.push(stratVal === '3' ? 1.0 : 0.0)

  vec.push(String(inputs.symptom) === '1' ? 1.0 : 0.0)

  return vec
}

// ============================================================
// MODEL A: TRUE DISCRETIZED BAYESIAN NETWORK CLIENT INFERENCE
// ============================================================
function runDiscretizedInference(inputs: PatientInputs): AnalyzeResult {
  const evidence: Record<string, string> = {}
  const numericalFeedback: NumericalFeedback[] = []

  // Discretize numerical inputs
  for (const varName of CONTINUOUS_VARS) {
    const rawVal = parseFloat(inputs[varName]) || 0
    const state = discretizeValue(varName, rawVal, discMetadata) || `${varName}_2`
    evidence[varName] = state

    const vMeta = discData.variables[varName]
    const ranges = []
    if (vMeta) {
      const [, e1, e2] = vMeta.edges
      if (varName === 'karnof') {
        ranges.push({ condition: `karnof <= ${e1}`, state: 'karnof_1' })
        ranges.push({ condition: `karnof > ${e1}`, state: 'karnof_2' })
      } else if (varName === 'preanti') {
        ranges.push({ condition: 'preanti = 0', state: 'zero' })
        ranges.push({ condition: `0 < preanti <= ${e1}`, state: 'positive_1' })
        ranges.push({ condition: `${e1} < preanti <= ${e2}`, state: 'positive_2' })
        ranges.push({ condition: `preanti > ${e2}`, state: 'positive_3' })
      } else {
        ranges.push({ condition: `${varName} <= ${e1}`, state: `${varName}_1` })
        ranges.push({ condition: `${e1} < ${varName} <= ${e2}`, state: `${varName}_2` })
        ranges.push({ condition: `${varName} > ${e2}`, state: `${varName}_3` })
      }
    }

    numericalFeedback.push({
      variable: varName,
      value: rawVal,
      state,
      ranges,
    })
  }

  // Categorical evidence
  for (const varName of CATEGORICAL_VARS) {
    evidence[varName] = String(inputs[varName] ?? '0').trim()
  }

  const cd40State = evidence.cd40 || 'cd40_2'
  const z30State = evidence.z30 === '1' ? '1' : '0'

  // Exact CPT query under mutilated graph: P(label=1 | cd40, trt=t, z30)
  const treatments: TreatmentResult[] = []

  for (let t = 0; t < 4; t++) {
    const key = `${cd40State}_${t}_${z30State}`
    const p1 = discData.cpt_label_1[key] ?? 0.2
    const p0 = 1.0 - p1
    const eu = p0 * 1.0

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

  treatments.sort((a, b) => a.treatment - b.treatment)

  const recommendedItem = treatments.reduce((best, cur) => {
    if (cur.expected_utility > best.expected_utility) return cur
    if (cur.expected_utility === best.expected_utility && cur.treatment < best.treatment) return cur
    return best
  }, treatments[0])

  for (const item of treatments) {
    item.is_recommended = item.treatment === recommendedItem.treatment
  }

  const sortedRanking = [...treatments].sort((a, b) => a.rank - b.rank)

  const monoArm = treatments.find((t) => t.treatment === 0) ?? treatments[0]
  const p1Mono = Math.max(monoArm.p_label_1, 0.0001)
  const p1Rec = Math.max(recommendedItem.p_label_1, 0.0001)
  const riskDeltaVsMono = Math.round((monoArm.p_label_1 - recommendedItem.p_label_1) * 10000) / 10000

  // E-Value Sensitivity Analysis
  const riskRatio = p1Rec / p1Mono
  const eValueAnalysis = computeEValue(riskRatio)

  // Treatment Benefit Percentile
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

  // Model A Evidence Ablation Attributions
  const recTrt = recommendedItem.treatment
  const featureAttributions: FeatureAttribution[] = []

  for (const varName of [...CONTINUOUS_VARS, ...CATEGORICAL_VARS]) {
    let deltaP = 0.0
    if (varName === 'cd40') {
      const ablatedP1 = discData.ablation_no_cd40[`${recTrt}_${z30State}`] ?? recommendedItem.p_label_1
      deltaP = recommendedItem.p_label_1 - ablatedP1
    } else if (varName === 'z30') {
      const ablatedP1 = discData.ablation_no_z30[`${cd40State}_${recTrt}`] ?? recommendedItem.p_label_1
      deltaP = recommendedItem.p_label_1 - ablatedP1
    } else {
      // In mutilated DAG, label is d-separated from other variables given {cd40, trt, z30}
      deltaP = 0.0
    }

    featureAttributions.push({
      feature: varName,
      observed_state: evidence[varName] || '0',
      risk_impact: Math.round(deltaP * 10000) / 10000,
      direction: deltaP > 0.0005 ? 'increases_risk' : deltaP < -0.0005 ? 'reduces_risk' : 'neutral',
      percentage_points: Math.round(deltaP * 10000) / 100,
    })
  }

  featureAttributions.sort((a, b) => Math.abs(b.risk_impact) - Math.abs(a.risk_impact))

  // Step-wise What-If CD4 Trajectory for Model A
  const cd4Trajectory: TrajectoryPoint[] = []
  const currentCd4 = parseFloat(inputs.cd40) || 350

  for (let cd4Val = 50; cd4Val <= 800; cd4Val += 50) {
    const stepCat = discretizeValue('cd40', cd4Val, discMetadata) || 'cd40_2'
    const point: TrajectoryPoint = {
      cd4: cd4Val,
      arm_0: Math.round((discData.cpt_label_1[`${stepCat}_0_${z30State}`] ?? 0.3) * 10000) / 10000,
      arm_1: Math.round((discData.cpt_label_1[`${stepCat}_1_${z30State}`] ?? 0.2) * 10000) / 10000,
      arm_2: Math.round((discData.cpt_label_1[`${stepCat}_2_${z30State}`] ?? 0.15) * 10000) / 10000,
      arm_3: Math.round((discData.cpt_label_1[`${stepCat}_3_${z30State}`] ?? 0.2) * 10000) / 10000,
    }
    cd4Trajectory.push(point)
  }

  return {
    model_type: 'discretized_bayesian_network',
    model_name: 'Model A: Baseline Discretized Bayesian Network',
    information_preservation: '3-bin quantile discretization (baseline)',
    evidence,
    numerical_feedback: numericalFeedback,
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
      feature_attributions: featureAttributions,
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

// ============================================================
// MODEL B: CONTINUOUS HYBRID SCM CLIENT INFERENCE
// ============================================================
function runContinuousInference(inputs: PatientInputs): AnalyzeResult {
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
    const eu = p0 * 1.0

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

  treatments.sort((a, b) => a.treatment - b.treatment)

  const recommendedItem = treatments.reduce((best, cur) => {
    if (cur.expected_utility > best.expected_utility) return cur
    if (cur.expected_utility === best.expected_utility && cur.treatment < best.treatment) return cur
    return best
  }, treatments[0])

  for (const item of treatments) {
    item.is_recommended = item.treatment === recommendedItem.treatment
  }

  const sortedRanking = [...treatments].sort((a, b) => a.rank - b.rank)

  const monoArm = treatments.find((t) => t.treatment === 0) ?? treatments[0]
  const p1Mono = Math.max(monoArm.p_label_1, 0.0001)
  const p1Rec = Math.max(recommendedItem.p_label_1, 0.0001)
  const riskDeltaVsMono = Math.round((monoArm.p_label_1 - recommendedItem.p_label_1) * 10000) / 10000

  // 1. E-Value Sensitivity Analysis for Unmeasured Confounding
  const riskRatio = p1Rec / p1Mono
  const eValueAnalysis = computeEValue(riskRatio)

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
    const stepUnit =
      varName.includes('cd') ? 50.0 : varName === 'karnof' ? 10.0 : varName === 'preanti' ? 100.0 : 5.0
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
    model_type: 'continuous_hybrid_scm',
    model_name: 'Model B: Continuous / Hybrid SCM (G-Computation)',
    information_preservation: 'Exact continuous numerical measurements preserved without discretization',
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

// ============================================================
// MAIN DISPATCHER
// ============================================================
export function runClientInference(
  inputs: PatientInputs,
  modelType: 'continuous' | 'discretized' = 'continuous',
): AnalyzeResult {
  if (modelType === 'discretized') {
    return runDiscretizedInference(inputs)
  }
  return runContinuousInference(inputs)
}

