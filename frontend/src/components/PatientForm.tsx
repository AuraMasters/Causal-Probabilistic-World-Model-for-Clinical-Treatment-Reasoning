import { AnimatePresence, motion } from 'framer-motion'
import { Activity, Eraser, FlaskConical, Play, Sparkles, UserCheck } from 'lucide-react'
import { useMemo, useState } from 'react'
import { describeRanges, discretizeValue } from '../lib/discretize'
import { CATEGORICAL_FIELDS, NUMERICAL_FIELDS } from '../lib/fields'
import type { DiscretizationMetadata, PatientInputs } from '../lib/types'

interface PatientFormProps {
  metadata: DiscretizationMetadata
  analyzing: boolean
  onAnalyze: (inputs: PatientInputs) => void
}

const PRESETS: { name: string; tag: string; inputs: PatientInputs }[] = [
  {
    name: 'Sample 1 · ART-Naive',
    tag: 'Good Baseline',
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
    tag: 'Low CD4 Risk',
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
      setError('All numerical inputs must be valid numbers.')
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
    <div className="card-surface rounded-2xl p-6 sm:p-8">
      {/* Form header & sample profiles */}
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <UserCheck className="h-5 w-5 text-mint-300" />
            <h3 className="font-display text-lg font-bold text-slate-100">Patient Clinical Profile</h3>
          </div>
          <p className="mt-1 text-sm text-slate-400">
            Discretized boundaries automatically convert continuous biomarkers into evidence factors.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <span className="font-mono text-xs font-semibold tracking-wider text-slate-400 uppercase mr-0.5">
            Presets:
          </span>
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              onClick={() => applyPreset(preset.inputs)}
              className="inline-flex items-center gap-2 rounded-xl border border-cyan-400/40 bg-gradient-to-r from-cyan-950/80 to-ink-800 px-3.5 py-2 text-xs font-bold text-cyan-200 shadow-sm shadow-cyan-950/40 transition-all hover:border-cyan-300 hover:text-white hover:shadow-md hover:shadow-cyan-500/20 active:scale-98 cursor-pointer"
            >
              <Sparkles className="h-3.5 w-3.5 text-cyan-300 shrink-0" />
              <span>{preset.name}</span>
            </button>
          ))}
          <button
            type="button"
            onClick={clearAll}
            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-600/60 bg-ink-800/90 px-3.5 py-2 text-xs font-semibold text-slate-300 shadow-sm transition-all hover:border-slate-400 hover:bg-ink-700 hover:text-white active:scale-98 cursor-pointer"
          >
            <Eraser className="h-3.5 w-3.5 text-slate-400" />
            Clear
          </button>
        </div>
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

      <div className="mt-8">
        <p className="mb-3 font-mono text-[11px] font-semibold tracking-[0.2em] text-slate-400 uppercase">
          Categorical &amp; Baseline Indicators
        </p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
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

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="mt-5 rounded-lg border border-rose-400/30 bg-rose-400/10 px-4 py-2.5 text-sm text-rose-200 font-medium"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-slate-700/40 pt-6">
        <p className="flex items-center gap-2 text-xs text-slate-400">
          <FlaskConical className="h-4 w-4 text-mint-400" />
          Executes exact Pearlian Variable Elimination do(trt = k) across all 4 drug arms.
        </p>
        <motion.button
          whileHover={{ scale: analyzing ? 1 : 1.02 }}
          whileTap={{ scale: analyzing ? 1 : 0.98 }}
          onClick={handleAnalyze}
          disabled={analyzing}
          className={`inline-flex items-center gap-2.5 rounded-xl px-8 py-3.5 text-sm font-bold tracking-wide transition-all ${
            analyzing
              ? 'cursor-not-allowed bg-slate-700 text-slate-300'
              : 'bg-gradient-to-r from-mint-400 via-mint-500 to-cyan-400 text-ink-950 shadow-lg shadow-mint-500/25 hover:shadow-xl hover:shadow-mint-500/40 cursor-pointer'
          }`}
        >
          {analyzing ? (
            <>
              <Activity className="h-4 w-4 animate-spin text-slate-900" />
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
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.03, duration: 0.3 }}
      className={`relative rounded-xl border p-3.5 transition-all ${
        focused
          ? 'border-cyan-400/60 bg-ink-850 shadow-md shadow-cyan-400/5'
          : 'border-slate-600/30 bg-ink-900/60 hover:border-slate-500/40'
      }`}
    >
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold text-slate-200">
          {label}
          {unit && <span className="ml-1 text-xs font-normal text-slate-400">({unit})</span>}
        </label>
        <span className="font-mono text-[10px] text-slate-500">{hint}</span>
      </div>

      <div className="mt-2 flex items-center gap-2">
        <input
          type="number"
          step="any"
          placeholder={placeholder}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          className="ring-focus w-full rounded-lg border border-slate-600/40 bg-ink-950/90 px-3 py-2 font-mono text-sm text-slate-100 placeholder-slate-600"
        />
        {assignedState !== null && (
          <span className="shrink-0 rounded-md border border-mint-400/30 bg-mint-400/10 px-2 py-1 font-mono text-xs font-semibold text-mint-300">
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
            <div className="mt-2.5 rounded-lg border border-slate-700/50 bg-ink-950/80 p-2 text-[11px] text-slate-400">
              <p className="font-mono font-semibold text-slate-300">Discretized bins:</p>
              <div className="mt-1 space-y-0.5 font-mono text-[10px]">
                {ranges.map((range, rangeIndex) => {
                  const isCurrent = assignedState ? range.includes(assignedState) : false
                  return (
                    <div
                      key={rangeIndex}
                      className={`flex justify-between ${
                        isCurrent ? 'font-bold text-mint-300' : 'text-slate-500'
                      }`}
                    >
                      <span>Bin {rangeIndex}:</span>
                      <span>{range}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
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
    <div className="rounded-xl border border-slate-600/30 bg-ink-900/60 p-3.5 hover:border-slate-500/40 transition-colors">
      <label className="mb-2 flex items-center justify-between text-sm font-semibold text-slate-200">
        {label}
        <span className="font-mono text-[10px] font-medium tracking-wider text-slate-500 uppercase">{description}</span>
      </label>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="ring-focus w-full cursor-pointer rounded-lg border border-slate-600/40 bg-ink-950/90 px-3 py-2 text-sm text-slate-100 font-medium"
      >
        <option value="" disabled>
          Select...
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
