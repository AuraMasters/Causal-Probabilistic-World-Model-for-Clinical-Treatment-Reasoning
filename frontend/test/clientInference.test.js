import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const libDir = path.join(__dirname, '..', 'src', 'lib')

test('1. Gender encoding is consistent and matches ACTG175 ground truth', () => {
  const fieldsSource = fs.readFileSync(path.join(libDir, 'fields.ts'), 'utf-8')
  
  // Verify Female is 0 and Male is 1
  assert.match(fieldsSource, /value:\s*'0',\s*label:\s*'Female'/, 'Gender 0 must be labeled Female')
  assert.match(fieldsSource, /value:\s*'1',\s*label:\s*'Male'/, 'Gender 1 must be labeled Male')
  
  // Verify 0 does not map to Male
  assert.doesNotMatch(fieldsSource, /value:\s*'0',\s*label:\s*'Male'/, 'Gender 0 must not be labeled Male')
})

test('2. Discretized Model A CPT artifact integrity', () => {
  const discData = JSON.parse(fs.readFileSync(path.join(libDir, 'discretizedModelData.json'), 'utf-8'))
  
  assert.equal(discData.model_name, 'Model A: Baseline Discretized Bayesian Network')
  assert.ok(discData.cpt_label_1, 'Must contain cpt_label_1')
  
  const keys = Object.keys(discData.cpt_label_1)
  assert.equal(keys.length, 24, 'Must have exactly 24 CPT keys (3 cd40 states * 4 trt arms * 2 z30 states)')
  
  // Validate every probability is in (0, 1)
  for (const [key, prob] of Object.entries(discData.cpt_label_1)) {
    assert.ok(prob >= 0.0 && prob <= 1.0, `Probability for ${key} must be valid: ${prob}`)
  }
  
  // Verify ablation tables exist
  assert.ok(discData.ablation_no_cd40, 'Must have ablation_no_cd40')
  assert.ok(discData.ablation_no_z30, 'Must have ablation_no_z30')
  assert.equal(Object.keys(discData.ablation_no_cd40).length, 8)
  assert.equal(Object.keys(discData.ablation_no_z30).length, 12)
})

test('3. Continuous Model B weights artifact integrity', () => {
  const weights = JSON.parse(fs.readFileSync(path.join(libDir, 'modelWeights.json'), 'utf-8'))
  
  assert.ok(weights.arms, 'Must contain 4 treatment arms')
  for (let arm = 0; arm < 4; arm++) {
    assert.ok(weights.arms[String(arm)], `Missing arm ${arm}`)
    assert.equal(typeof weights.arms[String(arm)].intercept, 'number')
    assert.equal(weights.arms[String(arm)].coefficients.length, 16, 'Expected 16 features (6 cont + 10 cat one-hot)')
  }
  
  assert.equal(weights.feature_names.length, 16)
  assert.ok(weights.dev_risk_reductions.length > 0, 'Must contain development risk reductions for CATE percentiles')
})

test('4. E-Value calculation bounds and robustness semantics', () => {
  function computeEValue(riskRatio) {
    let eValue = 1.0
    if (riskRatio < 1.0 && riskRatio > 0) {
      const rrStar = 1.0 / riskRatio
      eValue = rrStar + Math.sqrt(rrStar * (rrStar - 1.0))
    }
    const eValueRounded = Math.round(eValue * 100) / 100
    const isRobust = riskRatio < 1.0 && eValueRounded >= 1.5
    return { eValueRounded, isRobust }
  }

  // Significant risk reduction
  const res1 = computeEValue(0.4)
  assert.ok(res1.eValueRounded >= 2.0)
  assert.equal(res1.isRobust, true)

  // Modest risk reduction
  const res2 = computeEValue(0.9)
  assert.ok(res2.eValueRounded < 1.5)
  assert.equal(res2.isRobust, false)

  // Null or harmful (no risk reduction vs monotherapy)
  const res3 = computeEValue(1.1)
  assert.equal(res3.eValueRounded, 1.0)
  assert.equal(res3.isRobust, false)
})

test('5. Precomputed overview and DAG structure integrity', () => {
  const overview = JSON.parse(fs.readFileSync(path.join(libDir, 'precomputedOverview.json'), 'utf-8'))
  
  assert.equal(overview.dataset.patients, 2139)
  assert.equal(overview.dataset.development_rows, 1711)
  assert.equal(overview.dataset.test_rows, 428)
  assert.equal(overview.dataset.treatments, 4)
  assert.equal(overview.dag.length, 23, 'Must contain 23 DAG edges')
  
  const trtLabelEdge = overview.dag.find(e => e.source === 'trt' && e.target === 'label')
  assert.ok(trtLabelEdge, 'Must include trt -> label edge')
})

test('6. Survival analysis and right-censoring audit artifact integrity', () => {
  const overview = JSON.parse(fs.readFileSync(path.join(libDir, 'precomputedOverview.json'), 'utf-8'))
  
  assert.ok(overview.survival_audit, 'Overview must include survival_audit')
  assert.equal(overview.survival_audit.total_patients, 2139)
  assert.equal(overview.survival_audit.n_events, 521)
  assert.equal(overview.survival_audit.n_censored, 1618)
  
  // Follow-up & censoring metrics
  assert.ok(overview.survival_audit.pct_censored_ge_2yr >= 80.0, 'Over 80% censored followed >= 2 years')
  assert.ok(overview.survival_audit.median_followup_censored_days >= 1000)
  
  // Cox Proportional Hazards
  const cox = overview.survival_audit.cox_proportional_hazards
  assert.ok(cox.harrell_c_index > 0.65, 'Harrell C-index must exceed 0.65')
  assert.ok(cox.treatment_hazard_ratios.Arm_1_vs_0.hazard_ratio < 0.70)
  assert.ok(cox.treatment_hazard_ratios.Arm_2_vs_0.hazard_ratio < 0.70)
})

