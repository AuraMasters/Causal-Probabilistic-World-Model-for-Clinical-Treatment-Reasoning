export type EdgeValue = number | null

export interface DiscretizationMetadata {
  representation: string
  n_bins: number
  fit_dataset: string
  variables: Record<string, { method: string; bins?: number; positive_bins?: number; edges: EdgeValue[] }>
}

export interface TreatmentInfo {
  treatment: number
  name: string
  short_name: string
}

export interface CurvePoint {
  fpr?: number
  tpr?: number
  recall?: number
  precision?: number
}

export interface ConfusionMatrixData {
  true_negatives: number
  false_positives: number
  false_negatives: number
  true_positives: number
  total: number
}

export interface PredictiveMetrics {
  accuracy: number
  precision: number
  recall_sensitivity: number
  specificity: number
  f1_score: number
  roc_auc: number
  pr_auc: number
  log_loss: number
  brier_score: number
  confusion_matrix: ConfusionMatrixData
  roc_curve: CurvePoint[]
  pr_curve: CurvePoint[]
}

export interface MetricCI {
  point_estimate: number
  ci_lower: number
  ci_upper: number
  std_error: number
}

export interface CalibrationBin {
  bin: number
  lower: number
  upper: number
  count: number
  mean_predicted: number | null
  observed_rate: number | null
  absolute_gap: number | null
}

export interface CalibrationData {
  ece: number
  mce: number
  brier_score: number
  calibration_intercept?: number
  calibration_slope?: number
  calibration_interpretation?: string
  bins: CalibrationBin[]
  probability_distribution: {
    bin_labels: string[]
    label_0_counts: number[]
    label_1_counts: number[]
  }
}

export interface DcaPoint {
  threshold_probability: number
  net_benefit_model: number
  net_benefit_all: number
  net_benefit_none: number
  interventions_avoided_per_100: number
}

export interface DecisionCurveAnalysis {
  event_prevalence: number
  evaluation_cohort_size: number
  superior_threshold_range?: string
  interpretation: string
  dca_points: DcaPoint[]
}

export interface ThresholdSweepItem {
  threshold: number
  sensitivity_recall: number
  specificity: number
  precision: number
  f1_score: number
  accuracy: number
  balanced_accuracy: number
  youden_j: number
  true_positives: number
  false_positives: number
  true_negatives: number
  false_negatives: number
}

export interface ThresholdAnalysis {
  investigation_summary: {
    root_cause_of_low_sensitivity_at_0_5: string
    selection_criterion: string
    optimal_threshold_tau: number
    development_metrics_at_optimal: ThresholdSweepItem
    test_default_threshold_0_50: ThresholdSweepItem
    test_calibrated_threshold: ThresholdSweepItem
  }
  development_threshold_sweep: ThresholdSweepItem[]
  test_threshold_sweep: ThresholdSweepItem[]
}

export interface BaselineModelResult {
  model_name: string
  model_type: string
  roc_auc: number
  pr_auc: number
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  brier_score: number
  log_loss: number
  supports_do_calculus: boolean
  supports_counterfactuals: boolean
  interpretability: string
}

export interface CounterfactualPatientRecord {
  patient_idx: number
  observed_treatment: number
  observed_treatment_name: string
  observed_risk: number
  recommended_treatment: number
  recommended_treatment_name: string
  recommended_risk: number
  risk_reduction: number
  status: string
}

export interface CounterfactualEvaluation {
  test_patients: number
  better_count: number
  better_rate: number
  same_count: number
  same_rate: number
  worse_count: number
  worse_rate: number
  mean_risk_reduction: number
  median_risk_reduction: number
  max_risk_reduction: number
  recommended_distribution?: Record<string, number>
  observed_distribution?: Record<string, number>
  risk_reduction_histogram: {
    bins: string[]
    counts: number[]
  }
  sample_patient_decisions: CounterfactualPatientRecord[]
}

export interface SubgroupItem {
  subgroup_name: string
  sample_size: number
  positive_events: number
  event_rate: number
  roc_auc: number | null
  pr_auc?: number | null
  accuracy: number
  precision?: number
  sensitivity?: number
  specificity?: number
  f1_score: number
  brier_score: number
  is_reliable: boolean
  reliability_note: string
}

export interface MethodologyAndLimitations {
  data_partitioning: string
  causal_assumptions: string
  counterfactual_interpretation?: string
  continuous_modeling?: string
  clinical_disclaimer: string
}

export interface ComprehensiveValidation {
  dataset: string
  development_rows: number
  test_rows: number
  dag_edges?: number
  parameter_prior?: string
  inference_engine?: string
  model_architecture?: string
  information_preservation?: string
  random_seed: number
  predictive_metrics: PredictiveMetrics
  confidence_intervals_95: Record<string, MetricCI>
  calibration: CalibrationData
  decision_curve_analysis?: DecisionCurveAnalysis
  threshold_analysis?: ThresholdAnalysis
  baseline_comparison?: BaselineModelResult[]
  counterfactual_treatment_evaluation: CounterfactualEvaluation
  subgroup_analysis: SubgroupItem[]
  methodology_and_limitations: MethodologyAndLimitations
}

export interface ModelComparisonItem {
  dimension: string
  model_a_discretized: string
  model_b_continuous: string
  advantage: string
}

export interface ModelComparisonData {
  title: string
  cohort: string
  summary: string
  comparison_table: ModelComparisonItem[]
  model_a_summary: {
    name: string
    roc_auc: number
    pr_auc: number
    brier_score: number
    ece: number
    optimal_tau: number
    calibrated_f1: number
  }
  model_b_summary: {
    name: string
    roc_auc: number
    pr_auc: number
    brier_score: number
    ece: number
    optimal_tau: number
    calibrated_f1: number
  }
}

export interface DagEdgeInfo {
  source: string
  target: string
  bootstrap_stability?: number
  support_category?: string
  reverse_stability?: number
  is_intervention_edge?: boolean
}

export interface KeyFindings {
  strengths: string[]
  limitations: string[]
}

export interface Overview {
  dataset: {
    name: string
    patients: number
    development_rows: number
    test_rows: number
    treatments: number
  }
  model: {
    dag_edges: number
    nodes: number
    parameter_learning: string
    active_model?: string
  }
  dag: DagEdgeInfo[]
  variables: {
    numerical: string[]
    categorical: string[]
  }
  states: Record<string, string[]>
  discretization: DiscretizationMetadata
  treatments: TreatmentInfo[]
  validation: {
    log_loss: number
    brier_score: number
    roc_auc: number
    accuracy: number
    ece: number
    test_patients: number
  }
  treatment_decision_validation: {
    better_count: number
    total: number
    rate: number
  }
  comprehensive_validation?: ComprehensiveValidation | null
  continuous_validation?: ComprehensiveValidation | null
  model_comparison?: ModelComparisonData | null
  key_findings?: KeyFindings
  utility_model: {
    label_0_utility: number
    label_1_utility: number
  }
}

export interface DiscretizationRange {
  condition: string
  state: string
}

export interface NumericalFeedback {
  variable: string
  value: number
  state: string
  ranges: DiscretizationRange[]
}

export interface TreatmentResult {
  treatment: number
  name: string
  short_name: string
  p_label_0: number
  p_label_1: number
  expected_utility: number
  rank: number
  is_recommended: boolean
}

export interface FeatureAttribution {
  feature: string
  observed_state: string
  risk_impact: number
  direction: 'increases_risk' | 'reduces_risk' | 'neutral'
  percentage_points: number
}

export interface ContinuousSensitivity {
  variable: string
  raw_value: number
  raw_derivative: number
  clinical_step_unit: number
  risk_change_per_step: number
  percentage_points_per_step: number
  direction: 'increases_risk' | 'reduces_risk' | 'neutral'
}

export interface EValueAnalysis {
  risk_ratio_vs_monotherapy: number
  e_value_point: number
  interpretation: string
  is_robust: boolean
}

export interface TreatmentBenefit {
  absolute_risk_reduction: number
  arr_percentage_points: number
  benefit_percentile: number
  benefit_tier: string
}

export interface TrajectoryPoint {
  cd4: number
  arm_0: number
  arm_1: number
  arm_2: number
  arm_3: number
}

export interface WhatIfTrajectory {
  variable: string
  label: string
  current_value: number
  trajectory_points: TrajectoryPoint[]
}

export interface RecommendedTreatment {
  treatment: number
  name: string
  short_name: string
  p_label_0: number
  p_label_1: number
  expected_utility: number
  risk_delta_vs_monotherapy?: number
  feature_attributions?: FeatureAttribution[]
  continuous_sensitivities?: ContinuousSensitivity[]
  e_value_analysis?: EValueAnalysis
  treatment_benefit?: TreatmentBenefit
}

export interface AnalyzeResult {
  model_type?: string
  model_name?: string
  information_preservation?: string
  evidence?: Record<string, string>
  inputs?: Record<string, any>
  numerical_feedback?: NumericalFeedback[]
  treatments: TreatmentResult[]
  ranking: TreatmentResult[]
  recommended: RecommendedTreatment
  what_if_trajectory?: WhatIfTrajectory
  utility_model: {
    label_0_utility: number
    label_1_utility: number
  }
  decision_rule?: string
}

export interface PatientInputs {
  age: string
  wtkg: string
  karnof: string
  preanti: string
  cd40: string
  cd80: string
  hemo: string
  homo: string
  drugs: string
  oprior: string
  z30: string
  race: string
  gender: string
  strat: string
  symptom: string
}
