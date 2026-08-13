export interface NumericalFieldDef {
  key: 'age' | 'wtkg' | 'karnof' | 'preanti' | 'cd40' | 'cd80'
  label: string
  unit: string
  placeholder: string
  hint: string
}

export interface CategoricalFieldDef {
  key: 'hemo' | 'homo' | 'drugs' | 'oprior' | 'z30' | 'race' | 'gender' | 'strat' | 'symptom'
  label: string
  description: string
  options: { value: string; label: string }[]
}

export const NUMERICAL_FIELDS: NumericalFieldDef[] = [
  { key: 'age', label: 'Age', unit: 'years', placeholder: 'e.g. 44', hint: 'ACTG175 range 12–70' },
  { key: 'wtkg', label: 'Weight', unit: 'kg', placeholder: 'e.g. 70', hint: 'Body weight in kilograms' },
  { key: 'karnof', label: 'Karnofsky score', unit: '0–100', placeholder: 'e.g. 90', hint: 'Performance status' },
  { key: 'preanti', label: 'Pre-ART exposure', unit: 'days', placeholder: 'e.g. 366', hint: 'Days of prior antiretroviral therapy' },
  { key: 'cd40', label: 'CD4 count', unit: 'cells/mm³', placeholder: 'e.g. 445', hint: 'Baseline CD4' },
  { key: 'cd80', label: 'CD8 count', unit: 'cells/mm³', placeholder: 'e.g. 900', hint: 'Baseline CD8' },
]

export const CATEGORICAL_FIELDS: CategoricalFieldDef[] = [
  {
    key: 'hemo',
    label: 'Hemophilia',
    description: 'Hemophilia',
    options: [
      { value: '0', label: 'No' },
      { value: '1', label: 'Yes' },
    ],
  },
  {
    key: 'homo',
    label: 'Homosexuality',
    description: 'Male homosexuality',
    options: [
      { value: '0', label: 'No' },
      { value: '1', label: 'Yes' },
    ],
  },
  {
    key: 'drugs',
    label: 'IV drug use',
    description: 'History of IV drug use',
    options: [
      { value: '0', label: 'No' },
      { value: '1', label: 'Yes' },
    ],
  },
  {
    key: 'oprior',
    label: 'Prior opportunistic infection',
    description: 'Prior opportunistic infection',
    options: [
      { value: '0', label: 'No' },
      { value: '1', label: 'Yes' },
    ],
  },
  {
    key: 'z30',
    label: 'Prior ZDV use',
    description: 'Zidovudine use within 30 days before entry',
    options: [
      { value: '0', label: 'No' },
      { value: '1', label: 'Yes' },
    ],
  },
  {
    key: 'race',
    label: 'Race',
    description: 'Race',
    options: [
      { value: '0', label: 'White' },
      { value: '1', label: 'Non-white' },
    ],
  },
  {
    key: 'gender',
    label: 'Gender',
    description: 'Gender',
    options: [
      { value: '0', label: 'Male' },
      { value: '1', label: 'Female' },
    ],
  },
  {
    key: 'strat',
    label: 'Stratification',
    description: 'CD4 stratum',
    options: [
      { value: '1', label: '1' },
      { value: '2', label: '2' },
      { value: '3', label: '3' },
    ],
  },
  {
    key: 'symptom',
    label: 'Symptoms',
    description: 'Symptomatic disease',
    options: [
      { value: '0', label: 'Asymptomatic' },
      { value: '1', label: 'Symptomatic' },
    ],
  },
]

export const NUMERICAL_KEYS = NUMERICAL_FIELDS.map((field) => field.key)
export const CATEGORICAL_KEYS = CATEGORICAL_FIELDS.map((field) => field.key)

export const TREATMENT_LABEL: Record<string, string> = {
  '0': 'Zidovudine (ZDV/AZT)',
  '1': 'Zidovudine + Didanosine',
  '2': 'Zidovudine + Zalcitabine',
  '3': 'Didanosine (ddI)',
}

export const TREATMENT_SHORT: Record<string, string> = {
  '0': 'ZDV',
  '1': 'ZDV + ddI',
  '2': 'ZDV + ddC',
  '3': 'ddI',
}
