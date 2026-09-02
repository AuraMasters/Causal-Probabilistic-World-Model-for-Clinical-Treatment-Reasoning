import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  Clock,
  Eraser,
  FlaskConical,
  Layers,
  Play,
  Sparkles,
  Stethoscope,
  UserCheck,
} from 'lucide-react'
import { useMemo, useState } from 'react'
import { describeRanges, discretizeValue } from '../lib/discretize'
import { CATEGORICAL_FIELDS, NUMERICAL_FIELDS } from '../lib/fields'
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
    desc: 'CD4 480 · 0 prior ART days · No symptoms',
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
    name: 'Sample 2 · Moderate',
    tag: 'Prior ZDV',
    desc: 'CD4 350 · 366 prior ART days · Prior ZDV',
    inputs: {
      age: '44',
      wtkg: '70',
      karnof: '90',
      preanti: '366',
      cd40: '350',
      cd80: '900',
      hemo: '0',
      homo: '1',
      drugs: '0',
      oprior: '0',
      z30: '1',
      race: '0',
      gender: '1',
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

export function PatientForm({ metadata, analyzing, onAnalyze }: PatientFormProps) {
  const [inputs, setInputs] = useState<PatientInputs>(PRESETS[0].inputs)
  const [activeTab, setActiveTab] = useState<'biomarkers' | 'history' | 'all'>('biomarkers')
  const [error, setError] = useState<string | null>(null)

  const setValue = (key: keyof PatientInputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: value }))
    setError(null)
  }

  const applyPreset = (presetInputs: PatientInputs) => {
    setInputs(presetInputs)
    setError(null)
  }

  const clearAll = () => {
    setInputs(EMPTY_INPUTS)
    setError(null)
  }

  const isComplete = useMemo(() => {
    return NUMERICAL_FIELDS.every((field) => Number.isFinite(Number(inputs[field.key])))
  }, [inputs])

  const handleAnalyze = () => {
    if (!isComplete) {
      setError('Please provide valid numbers for all continuous biomarker fields.')
      return
    }
    const categoricalMissing = CATEGORICAL_FIELDS.find((field) => inputs[field.key] === '')
    if (categoricalMissing) {
      setError(`Please select a value for: ${categoricalMissing.label}.`)
      return
    }
    setError(null)
    onAnalyze(inputs)
  }

  return (
    <div className="card-surface rounded-2xl p-5 sm:p-8 shadow-xl">
      {/* Top Header: Presets & Controls */}
      <div className="mb-6 flex flex-col gap-4 border-b border-cyan-400/15 pb-6 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <UserCheck className="h-5 w-5 text-cyan-300" />
            <h3 className="font-display text-lg font-bold text-white">
              Patient Baseline Biomarkers & History
            </h3>
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Select a calibrated trial sample profile or specify custom clinical measurements.
          </p>
        </div>

        {/* Preset Profiles */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-xs font-semibold tracking-wider text-slate-400 uppercase mr-1">
            Presets:
          </span>
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              onClick={() => applyPreset(preset.inputs)}
              className="inline-flex items-center gap-1.5 rounded-xl border border-cyan-400/35 bg-ink-850 px-3.5 py-1.5 text-xs font-semibold text-cyan-200 shadow-sm transition-all hover:border-cyan-300 hover:bg-ink-800 hover:text-white cursor-pointer active:scale-98"
            >
              <Sparkles className="h-3 w-3 text-cyan-300" />
              <span>{preset.name}</span>
            </button>
          ))}
          <button
            type="button"
            onClick={clearAll}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700 bg-ink-900 px-3 py-1.5 text-xs font-medium text-slate-400 transition-colors hover:border-slate-500 hover:text-slate-200 cursor-pointer"
          >
            <Eraser className="h-3 w-3" />
            Reset
          </button>
        </div>
      </div>

      {/* Logical Section Tabs */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2 rounded-xl bg-ink-950/80 p-1 border border-slate-800">
          <button
            type="button"
            onClick={() => setActiveTab('biomarkers')}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'biomarkers'
                ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/35 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Stethoscope className="h-3.5 w-3.5" />
            <span>Primary Biomarkers (6)</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('history')}
            className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'history'
                ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/35 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Clock className="h-3.5 w-3.5" />
            <span>Clinical History & Demographics (9)</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('all')}
            className={`hidden sm:inline-flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-bold transition-all cursor-pointer ${
              activeTab === 'all'
                ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/35 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            <span>View All</span>
          </button>
        </div>

        <span className="hidden font-mono text-xs text-slate-400 sm:inline-block">
          Exact BDeu Evidence States Active
        </span>
      </div>

      {/* Numerical Fields: Primary Biomarkers */}
      {(activeTab === 'biomarkers' || activeTab === 'all') && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="font-mono text-xs font-bold tracking-wider text-cyan-300 uppercase">
              Continuous Biomarkers & Physical Measurements
            </p>
            <span className="font-mono text-[11px] text-slate-400">
              Quantile discretization fitted on development set
            </span>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {NUMERICAL_FIELDS.map((field, index) => (
              <NumericField
                key={field.key}
                index={index}
                label={field.label}
                unit={field.unit}
                placeholder={field.placeholder}
                hint={field.hint}
                value={inputs[field.key]}
                metadata={metadata}
                onChange={(value) => setValue(field.key, value)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Categorical Fields: Clinical History & Demographics */}
      {(activeTab === 'history' || activeTab === 'all') && (
        <div className={`space-y-4 ${activeTab === 'all' ? 'mt-8 border-t border-slate-800 pt-6' : ''}`}>
          <div className="flex items-center justify-between">
            <p className="font-mono text-xs font-bold tracking-wider text-cyan-300 uppercase">
              Clinical History & Demographic Attributes
            </p>
            <span className="font-mono text-[11px] text-slate-400">
              9 discrete clinical indicators
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
            {CATEGORICAL_FIELDS.map((field) => (
              <CategoricalField
                key={field.key}
                label={field.label}
                description={field.description}
                options={field.options}
                value={inputs[field.key]}
                onChange={(value) => setValue(field.key, value)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Error Banner */}
      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="mt-5 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200 font-semibold"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      {/* Form Bottom Action Bar */}
      <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-cyan-400/15 pt-6">
        <p className="flex items-center gap-2 text-xs text-slate-400">
          <FlaskConical className="h-4 w-4 text-cyan-300" />
          <span>Executes exact Bayesian Variable Elimination for interventional do(trt = k).</span>
        </p>

        <motion.button
          whileHover={{ scale: analyzing ? 1 : 1.01 }}
          whileTap={{ scale: analyzing ? 1 : 0.98 }}
          onClick={handleAnalyze}
          disabled={analyzing}
          className={`inline-flex items-center gap-2.5 rounded-xl px-8 py-3.5 text-sm font-extrabold tracking-wide transition-all shadow-lg cursor-pointer ${
            analyzing
              ? 'cursor-not-allowed bg-slate-700 text-slate-300'
              : 'bg-gradient-to-r from-cyan-400 via-mint-300 to-cyan-300 text-ink-950 shadow-cyan-500/25 hover:shadow-cyan-500/40'
          }`}
        >
          {analyzing ? (
            <>
              <Activity className="h-4 w-4 animate-spin text-ink-950" />
              <span>Computing Causal Posterior...</span>
            </>
          ) : (
            <>
              <Play className="h-4 w-4 fill-ink-950" />
              <span>Run Interventional Inference</span>
            </>
          )}
        </motion.button>
      </div>
    </div>
  )
}

function NumericField({
  index,
  label,
  unit,
  placeholder,
  hint,
  value,
  metadata,
  onChange,
}: {
  index: number
  label: string
  unit: string
  placeholder: string
  hint: string
  value: string
  metadata: DiscretizationMetadata
  onChange: (value: string) => void
}) {
  const [focused, setFocused] = useState(false)
  const numValue = Number(value)
  const hasValue = value.trim() !== '' && Number.isFinite(numValue)
  const fieldKey = NUMERICAL_FIELDS[index].key
  const assignedState = hasValue ? discretizeValue(fieldKey, numValue, metadata) : null
  const ranges = describeRanges(fieldKey, metadata)

  return (
    <div
      className={`relative rounded-xl border p-4 transition-all ${
        focused
          ? 'border-cyan-400/70 bg-ink-850 shadow-md shadow-cyan-500/10'
          : 'border-slate-700/60 bg-ink-900/80 hover:border-slate-600'
      }`}
    >
      <div className="flex items-center justify-between">
        <label className="text-sm font-bold text-slate-100">
          {label}
          {unit && <span className="ml-1 text-xs font-normal text-slate-400">({unit})</span>}
        </label>
        <span className="font-mono text-[11px] text-slate-400">{hint}</span>
      </div>

      <div className="mt-2.5 flex items-center gap-2">
        <input
          type="number"
          step="any"
          placeholder={placeholder}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          className="ring-focus w-full rounded-lg border border-slate-700 bg-ink-950 px-3.5 py-2 font-mono text-sm font-semibold text-white placeholder-slate-600"
        />
        {assignedState !== null && (
          <span className="shrink-0 rounded-lg border border-cyan-400/35 bg-cyan-400/10 px-2.5 py-1.5 font-mono text-xs font-bold text-mint-200">
            {assignedState}
          </span>
        )}
      </div>

      <AnimatePresence>
        {focused && ranges.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="mt-2.5 rounded-lg border border-slate-800 bg-ink-950 p-2.5 text-xs text-slate-400">
              <p className="font-mono font-bold text-cyan-200 text-[11px]">Discretized Bins:</p>
              <div className="mt-1 space-y-0.5 font-mono text-[11px]">
                {ranges.map((range, rangeIndex) => {
                  const isCurrent = assignedState ? range.includes(assignedState) : false
                  return (
                    <div
                      key={rangeIndex}
                      className={`flex justify-between ${
                        isCurrent ? 'font-bold text-mint-200' : 'text-slate-400'
                      }`}
                    >
                      <span>Bin {rangeIndex + 1}:</span>
                      <span>{range}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function CategoricalField({
  label,
  description,
  options,
  value,
  onChange,
}: {
  label: string
  description: string
  options: { value: string; label: string }[]
  value: string
  onChange: (value: string) => void
}) {
  return (
    <div className="rounded-xl border border-slate-700/60 bg-ink-900/80 p-3.5 hover:border-slate-600 transition-colors">
      <div className="mb-2 flex items-center justify-between">
        <label className="text-xs font-bold text-slate-100">
          {label}
        </label>
        <span className="font-mono text-[10px] text-slate-400 uppercase">{description}</span>
      </div>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="ring-focus w-full cursor-pointer rounded-lg border border-slate-700 bg-ink-950 px-3 py-2 text-xs font-semibold text-slate-100"
      >
        <option value="" disabled>
          Select state...
        </option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  )
}
