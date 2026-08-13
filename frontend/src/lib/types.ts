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
  }
  dag: { source: string; target: string }[]
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

export interface AnalyzeResult {
  evidence: Record<string, string>
  numerical_feedback: NumericalFeedback[]
  treatments: TreatmentResult[]
  ranking: TreatmentResult[]
  recommended: {
    treatment: number
    name: string
    short_name: string
    p_label_0: number
    p_label_1: number
    expected_utility: number
  }
  utility_model: {
    label_0_utility: number
    label_1_utility: number
  }
  decision_rule: string
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
