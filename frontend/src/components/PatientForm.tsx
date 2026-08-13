import { AnimatePresence, motion } from 'framer-motion'
import { Activity, Eraser, FlaskConical, Play, Sparkles } from 'lucide-react'
import { useMemo, useState } from 'react'
import { discretizeValue, describeRanges } from '../lib/discretize'
import { CATEGORICAL_FIELDS, NUMERICAL_FIELDS } from '../lib/fields'
import type { DiscretizationMetadata, PatientInputs } from '../lib/types'

interface PatientFormProps {
  metadata: DiscretizationMetadata
  analyzing: boolean
  onAnalyze: (inputs: PatientInputs) => void
}

const EXAMPLE_INPUTS: PatientInputs = {
  age: '44',
  wtkg: '70',
  karnof: '90',
  preanti: '366',
  cd40: '445',
  cd80: '900',
  hemo: '0',
  homo: '1',
  drugs: '0',
  oprior: '0',
  z30: '0',
  race: '0',
  gender: '1',
  strat: '3',
  symptom: '0',
}

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
  const [inputs, setInputs] = useState<PatientInputs>(EMPTY_INPUTS)
  const [error, setError] = useState<string | null>(null)

  const setValue = (key: keyof PatientInputs, value: string) => {
    setInputs((current) => ({ ...current, [key]: value }))
    setError(null)
  }

  const fillExample = () => {
    setInputs(EXAMPLE_INPUTS)
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
      <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="font-display text-lg font-semibold text-slate-100">Patient profile</h3>
          <p className="mt-1 text-sm text-slate-400">
            Numerical values are converted to the model's discretized states using the development-fitted boundaries.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={fillExample}
            className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-400/25 bg-cyan-400/10 px-3 py-1.5 text-xs font-medium text-cyan-200 transition-colors hover:bg-cyan-400/20"
          >
            <Sparkles className="h-3.5 w-3.5" />
            Example patient
          </button>
          <button
            type="button"
            onClick={clearAll}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-500/25 bg-slate-500/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:bg-slate-500/20"
          >
            <Eraser className="h-3.5 w-3.5" />
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
        <p className="mb-3 font-mono text-[11px] font-medium tracking-[0.2em] text-slate-500 uppercase">
          Categorical characteristics
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
            className="mt-5 rounded-lg border border-rose-400/25 bg-rose-400/10 px-4 py-2.5 text-sm text-rose-200"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      <div className="mt-7 flex flex-wrap items-center justify-between gap-4">
        <p className="flex items-center gap-2 text-xs text-slate-500">
          <FlaskConical className="h-4 w-4 text-mint-400" />
          All four treatments are evaluated automatically by the model.
        </p>
        <motion.button
          whileHover={{ scale: analyzing ? 1 : 1.02 }}
          whileTap={{ scale: analyzing ? 1 : 0.98 }}
          onClick={handleAnalyze}
          disabled={analyzing}
          className={`inline-flex items-center gap-2.5 rounded-xl px-7 py-3.5 text-sm font-semibold transition-all ${
            analyzing
              ? 'cursor-not-allowed bg-slate-700 text-slate-300'
              : 'bg-gradient-to-r from-mint-500 to-cyan-glow text-ink-950 shadow-lg shadow-mint-500/20 hover:shadow-xl hover:shadow-mint-500/30'
          }`}
        >
          {analyzing ? (
            <>
              <motion.span
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
              >
                <Activity className="h-4 w-4" />
              </motion.span>
              Running inference…
            </>
          ) : (
            <>
              <Play className="h-4 w-4" strokeWidth={2.4} />
              Analyze patient
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
  const variable = NUMERICAL_FIELDS[index].key
  const numericValue = Number(value)
  const hasValidNumber = value.trim() !== '' && Number.isFinite(numericValue)
  const state = hasValidNumber ? discretizeValue(variable, value, metadata) : null
  const ranges = describeRanges(variable, metadata)

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.05, duration: 0.4 }}
      className="rounded-xl border border-slate-500/20 bg-ink-900/60 p-4"
    >
      <label className="mb-2 flex items-center justify-between text-sm font-medium text-slate-200">
        {label}
        <span className="font-mono text-[11px] text-slate-500">{unit}</span>
      </label>
      <input
        type="number"
        inputMode="decimal"
        step="any"
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        className="ring-focus w-full rounded-lg border border-slate-600/40 bg-ink-950/80 px-3 py-2.5 font-mono text-sm text-slate-100 placeholder:text-slate-600"
      />
      <p className="mt-1.5 text-[11px] text-slate-500">{hint}</p>

      <AnimatePresence>
        {state && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="mt-3 rounded-lg border border-mint-400/20 bg-mint-400/5 px-3 py-2.5">
              <p className="font-mono text-xs text-slate-300">
                <span className="text-slate-500">{label} = </span>
                {value}
                <span className="mx-2 text-slate-600">→</span>
                <span className="font-semibold text-mint-300">model state · {state}</span>
              </p>
              <div className="mt-2 space-y-0.5">
                {ranges.map((range) => (
                  <p
                    key={range}
                    className={`font-mono text-[10.5px] ${
                      range.includes(state) ? 'text-mint-300/90' : 'text-slate-600'
                    }`}
                  >
                    {range}
                  </p>
                ))}
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
    <div className="rounded-xl border border-slate-500/20 bg-ink-900/60 p-3.5">
      <label className="mb-2 flex items-center justify-between text-sm font-medium text-slate-200">
        {label}
        <span className="font-mono text-[10px] font-medium tracking-wider text-slate-500 uppercase">{description}</span>
      </label>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="ring-focus w-full cursor-pointer rounded-lg border border-slate-600/40 bg-ink-950/80 px-3 py-2.5 text-sm text-slate-100"
      >
        <option value="" disabled>
          Select…
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
