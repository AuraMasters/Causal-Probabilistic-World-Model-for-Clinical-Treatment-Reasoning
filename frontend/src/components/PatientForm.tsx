import { motion } from 'framer-motion'
import {
  Eraser,
  FlaskConical,
  HeartPulse,
  Info,
  Play,
  Stethoscope,
  User,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { discretizeValue } from '../lib/discretize'
import { CATEGORICAL_FIELDS, NUMERICAL_FIELDS, type CategoricalFieldDef, type NumericalFieldDef } from '../lib/fields'
import type { DiscretizationMetadata, PatientInputs } from '../lib/types'

interface PatientFormProps {
  metadata: DiscretizationMetadata
  analyzing: boolean
  onAnalyze: (inputs: PatientInputs) => void
}

const PRESETS: { name: string; tag: string; desc: string; inputs: PatientInputs }[] = [
  {
    name: 'Sample 1 · ART-Naive',
    tag: 'Baseline Good',
    desc: 'CD4 480 · 0 prior ART days · Asymptomatic',
    inputs: {
      age: '35',
      wtkg: '72',
      karnof: '100',
      preanti: '0',
      cd40: '480',
      cd80: '820',
      hemo: '0',
      homo: '1',
      drugs: '0',
      oprior: '0',
      z30: '0',
      race: '0',
      gender: '1',
      strat: '1',
      symptom: '0',
    },
  },
  {
    name: 'Sample 2 · Moderate (Female)',
    tag: 'Prior ZDV',
    desc: 'Female · CD4 350 · 366 prior ART days · ZDV Experienced',
    inputs: {
      age: '44',
      wtkg: '64',
      karnof: '90',
      preanti: '366',
      cd40: '350',
      cd80: '900',
      hemo: '0',
      homo: '0',
      drugs: '0',
      oprior: '0',
      z30: '1',
      race: '0',
      gender: '0',
      strat: '2',
      symptom: '0',
    },
  },
  {
    name: 'Sample 3 · Advanced',
    tag: 'High Risk',
    desc: 'CD4 180 · 650 prior ART days · Symptomatic',
    inputs: {
      age: '52',
      wtkg: '62',
      karnof: '80',
      preanti: '650',
      cd40: '180',
      cd80: '680',
      hemo: '1',
      homo: '0',
      drugs: '1',
      oprior: '1',
      z30: '1',
      race: '1',
      gender: '1',
      strat: '3',
      symptom: '1',
    },
  },
]

const EMPTY_INPUTS: PatientInputs = {
  age: '',
  wtkg: '',
  karnof: '',
  preanti: '',
  cd40: '',
  cd80: '',
  hemo: '',
  homo: '',
  drugs: '',
  oprior: '',
  z30: '',
  race: '',
  gender: '',
  strat: '',
  symptom: '',
}

const FIELD_TOOLTIPS: Record<string, string> = {
  age: 'Patient chronological age in years at baseline trial screening.',
  wtkg: 'Baseline body weight in kilograms (kg).',
  gender: 'Biological sex: 0 = Female, 1 = Male.',
  race: 'Demographic race classification: 0 = White, 1 = Non-White.',
  cd40: 'Baseline CD4+ T-helper lymphocyte count (cells/mm³). Key immunologic surrogate marker for HIV progression.',
  cd80: 'Baseline CD8+ cytotoxic T-lymphocyte count (cells/mm³). Evaluates immunologic response and immune activation.',
  karnof: 'Karnofsky Performance Scale (70–100%). Measures general functional capacity and activities of daily living.',
  preanti: 'Cumulative duration of prior antiretroviral exposure (days) prior to study entry.',
  symptom: 'Clinical disease presentation at baseline: 0 = Asymptomatic, 1 = Symptomatic HIV infection.',
  z30: 'History of Zidovudine (ZDV/AZT) administration within 30 days prior to baseline randomization.',
  oprior: 'History of prior non-ZDV antiretroviral therapy (e.g. ddI, ddC, or other nucleoside analogs).',
  strat: 'Trial stratification group: 1 = ART-Naive, 2 = 1 to 52 weeks prior ART, 3 = >52 weeks prior ART.',
  hemo: 'Documented medical history of hemophilia (0 = No, 1 = Yes).',
  homo: 'Transmission risk factor classification for male homosexual contact (0 = No, 1 = Yes).',
  drugs: 'Documented history of intravenous (IV) drug use (0 = No, 1 = Yes).',
}

export function PatientForm({ metadata, analyzing, onAnalyze }: PatientFormProps) {
  const [inputs, setInputs] = useState<PatientInputs>(PRESETS[0].inputs)
  const [activeTab, setActiveTab] = useState<'basic' | 'labs' | 'history'>('labs')

  const discretizedValues = useMemo(() => {
    const result: Record<string, { state: string; label: string } | null> = {}
    for (const field of NUMERICAL_FIELDS) {
      const rawVal = inputs[field.key as keyof PatientInputs]
      const state = discretizeValue(field.key, rawVal, metadata)
      if (state) {
        result[field.key] = {
          state,
          label: `Bin: ${state}`,
        }
      } else {
        result[field.key] = null
      }
    }
    return result
  }, [inputs, metadata])

  const filledCount = Object.values(inputs).filter((v) => v !== '').length
  const totalCount = Object.keys(inputs).length
  const isComplete = filledCount === totalCount

  const handleFieldChange = (key: keyof PatientInputs, value: string) => {
    setInputs((prev) => ({ ...prev, [key]: value }))
  }

  const handlePreset = (preset: typeof PRESETS[0]) => {
    setInputs({ ...preset.inputs })
  }

  const handleClear = () => {
    setInputs(EMPTY_INPUTS)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!isComplete || analyzing) return
    onAnalyze(inputs)
  }

  // Define field keys in each tab
  const basicFields = ['age', 'gender', 'wtkg', 'race']
  const labFields = ['cd40', 'cd80', 'karnof', 'preanti']
  const historyFields = ['symptom', 'z30', 'oprior', 'strat', 'hemo', 'homo', 'drugs']

  return (
    <form onSubmit={handleSubmit} className="card-surface rounded-xl p-6 shadow-xl border border-zinc-800 flex flex-col justify-between">
      <div className="space-y-6">
        {/* Header with Sample Presets */}
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-zinc-800 pb-4">
          <div>
            <div className="flex items-center gap-2">
              <Stethoscope className="h-5 w-5 text-zinc-300" />
              <h3 className="font-display text-base font-bold text-white tracking-tight">Patient Clinical Profile</h3>
            </div>
            <p className="text-xs text-zinc-400 mt-0.5">
              Specify demographic biomarkers, baseline labs, and medical history to simulate causal treatment interventions.
            </p>
          </div>

          {/* Quick Presets */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-mono text-[11px] font-bold text-zinc-400 uppercase mr-1">Presets:</span>
            {PRESETS.map((preset) => {
              const isSelected = JSON.stringify(inputs) === JSON.stringify(preset.inputs)
              return (
                <button
                  key={preset.name}
                  type="button"
                  onClick={() => handlePreset(preset)}
                  className={`rounded-md px-2.5 py-1 text-xs font-mono font-medium transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-white text-black font-bold shadow-sm'
                      : 'border border-zinc-800 bg-zinc-900 text-zinc-400 hover:border-zinc-700 hover:text-white'
                  }`}
                  title={preset.desc}
                >
                  {preset.tag}
                </button>
              )
            })}
            <button
              type="button"
              onClick={handleClear}
              className="rounded-md border border-zinc-800 p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-white transition-colors cursor-pointer"
              title="Clear all inputs"
            >
              <Eraser className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* 3 Logical Clinical Tabs */}
        <div className="flex border-b border-zinc-800 gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('basic')}
            className={`inline-flex items-center gap-2 pb-2.5 px-3 font-mono text-xs font-bold border-b-2 transition-colors cursor-pointer ${
              activeTab === 'basic'
                ? 'border-white text-white'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <User className="h-3.5 w-3.5" />
            <span>1. Demographics &amp; Vitals</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('labs')}
            className={`inline-flex items-center gap-2 pb-2.5 px-3 font-mono text-xs font-bold border-b-2 transition-colors cursor-pointer ${
              activeTab === 'labs'
                ? 'border-white text-white'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <FlaskConical className="h-3.5 w-3.5" />
            <span>2. Lab Biomarkers &amp; CD4</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('history')}
            className={`inline-flex items-center gap-2 pb-2.5 px-3 font-mono text-xs font-bold border-b-2 transition-colors cursor-pointer ${
              activeTab === 'history'
                ? 'border-white text-white'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            <HeartPulse className="h-3.5 w-3.5" />
            <span>3. Clinical History &amp; Risks</span>
          </button>
        </div>

        {/* Form Fields Rendered by Tab */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {/* TAB 1: BASIC DEMOGRAPHIC PROFILE */}
          {activeTab === 'basic' && (
            <>
              {NUMERICAL_FIELDS.filter((f) => basicFields.includes(f.key)).map((field) => (
                <NumericalInputCard
                  key={field.key}
                  field={field}
                  value={inputs[field.key as keyof PatientInputs]}
                  discretized={discretizedValues[field.key]}
                  tooltip={FIELD_TOOLTIPS[field.key]}
                  onChange={(val) => handleFieldChange(field.key as keyof PatientInputs, val)}
                />
              ))}
              {CATEGORICAL_FIELDS.filter((f) => basicFields.includes(f.key)).map((field) => (
                <CategoricalInputCard
                  key={field.key}
                  field={field}
                  value={inputs[field.key as keyof PatientInputs]}
                  tooltip={FIELD_TOOLTIPS[field.key]}
                  onChange={(val) => handleFieldChange(field.key as keyof PatientInputs, val)}
                />
              ))}
            </>
          )}

          {/* TAB 2: LAB BIOMARKERS & CD4 COUNTS */}
          {activeTab === 'labs' && (
            <>
              {NUMERICAL_FIELDS.filter((f) => labFields.includes(f.key)).map((field) => (
                <NumericalInputCard
                  key={field.key}
                  field={field}
                  value={inputs[field.key as keyof PatientInputs]}
                  discretized={discretizedValues[field.key]}
                  tooltip={FIELD_TOOLTIPS[field.key]}
                  onChange={(val) => handleFieldChange(field.key as keyof PatientInputs, val)}
                />
              ))}
            </>
          )}

          {/* TAB 3: CLINICAL HISTORY & RISK FACTORS */}
          {activeTab === 'history' && (
            <>
              {CATEGORICAL_FIELDS.filter((f) => historyFields.includes(f.key)).map((field) => (
                <CategoricalInputCard
                  key={field.key}
                  field={field}
                  value={inputs[field.key as keyof PatientInputs]}
                  tooltip={FIELD_TOOLTIPS[field.key]}
                  onChange={(val) => handleFieldChange(field.key as keyof PatientInputs, val)}
                />
              ))}
            </>
          )}
        </div>
      </div>

      {/* Action Footer Bar */}
      <div className="mt-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-t border-zinc-800 pt-4">
        <div className="flex items-center gap-3">
          <div className="flex h-2 w-28 overflow-hidden rounded-full bg-zinc-800">
            <div
              className="bg-white transition-all duration-300"
              style={{ width: `${(filledCount / totalCount) * 100}%` }}
            />
          </div>
          <span className="font-mono text-xs text-zinc-400">
            {filledCount} of {totalCount} variables specified
          </span>
        </div>

        <button
          type="submit"
          disabled={!isComplete || analyzing}
          className={`inline-flex items-center justify-center gap-2.5 rounded-lg px-6 py-2.5 font-display text-sm font-bold tracking-wide transition-all shadow-md cursor-pointer ${
            isComplete && !analyzing
              ? 'bg-white text-black hover:bg-zinc-200 shadow-white/10'
              : 'bg-zinc-900 text-zinc-600 border border-zinc-800 cursor-not-allowed'
          }`}
        >
          {analyzing ? (
            <>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                className="h-4 w-4 rounded-full border-2 border-black border-t-transparent"
              />
              <span>Evaluating Causal Inferences...</span>
            </>
          ) : (
            <>
              <Play className="h-4 w-4 fill-current" />
              <span>Simulate Interventions do(trt = k)</span>
            </>
          )}
        </button>
      </div>
    </form>
  )
}

function NumericalInputCard({
  field,
  value,
  discretized,
  tooltip,
  onChange,
}: {
  field: NumericalFieldDef
  value: string
  discretized: { state: string; label: string } | null
  tooltip?: string
  onChange: (val: string) => void
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-4 shadow-sm hover:border-zinc-700 transition-colors">
      <div className="flex items-center justify-between">
        <label htmlFor={`input-${field.key}`} className="font-mono text-xs font-bold text-zinc-300">
          {field.label} {field.unit && <span className="text-[10px] text-zinc-500">({field.unit})</span>}
        </label>
        {tooltip && (
          <span className="text-zinc-500 hover:text-zinc-300 cursor-help" title={tooltip}>
            <Info className="h-3.5 w-3.5" />
          </span>
        )}
      </div>

      <div className="mt-2 relative">
        <input
          id={`input-${field.key}`}
          name={field.key}
          type="number"
          step="any"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder || `e.g. ${field.hint}`}
          className="w-full rounded-md border border-zinc-800 bg-black px-3 py-2 text-sm font-mono text-white placeholder-zinc-600 focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-all"
        />
      </div>

      <div className="mt-2 flex items-center justify-between text-[11px] font-mono">
        <span className="text-zinc-500">{field.hint}</span>
        {discretized && (
          <span className="rounded bg-zinc-800 border border-zinc-700 px-1.5 py-0.5 font-bold text-zinc-200">
            {discretized.label}
          </span>
        )}
      </div>
    </div>
  )
}

function CategoricalInputCard({
  field,
  value,
  tooltip,
  onChange,
}: {
  field: CategoricalFieldDef
  value: string
  tooltip?: string
  onChange: (val: string) => void
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-4 shadow-sm hover:border-zinc-700 transition-colors">
      <div className="flex items-center justify-between">
        <label htmlFor={`select-${field.key}`} className="font-mono text-xs font-bold text-zinc-300">{field.label}</label>
        {tooltip && (
          <span className="text-zinc-500 hover:text-zinc-300 cursor-help" title={tooltip}>
            <Info className="h-3.5 w-3.5" />
          </span>
        )}
      </div>

      <div className="mt-2">
        <select
          id={`select-${field.key}`}
          name={field.key}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-md border border-zinc-800 bg-black px-3 py-2 text-sm font-mono text-white focus:border-white focus:outline-none focus:ring-1 focus:ring-white transition-all cursor-pointer"
        >
          <option value="" disabled>Select option...</option>
          {field.options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      <div className="mt-2 text-[11px] font-mono text-zinc-500 truncate">
        {field.options.find((o) => o.value === value)?.label || field.description || 'No selection'}
      </div>
    </div>
  )
}
