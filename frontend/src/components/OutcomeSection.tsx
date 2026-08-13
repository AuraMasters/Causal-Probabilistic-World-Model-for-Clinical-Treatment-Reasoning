import { motion } from 'framer-motion'
import { CheckCircle2, XCircle } from 'lucide-react'
import type { AnalyzeResult } from '../lib/types'
import { formatPercent, formatProbability } from '../lib/format'

export function OutcomeSection({ result }: { result: AnalyzeResult }) {
  const recommended = result.recommended

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      <OutcomeCard
        label="Label 0"
        meaning="Desirable study outcome"
        probability={recommended.p_label_0}
        accent="mint"
        icon={CheckCircle2}
      />
      <OutcomeCard
        label="Label 1"
        meaning="Undesirable study outcome"
        probability={recommended.p_label_1}
        accent="rose"
        icon={XCircle}
      />
      <div className="md:col-span-2">
        <div className="rounded-xl border border-slate-500/20 bg-ink-900/60 px-4 py-3">
          <p className="text-xs leading-relaxed text-slate-400">
            Predicted outcome for the <span className="font-semibold text-slate-200">recommended treatment</span> ·{' '}
            <span className="text-slate-200">{recommended.name}</span>. Label 0 — desirable study outcome; Label 1 —
            undesirable study outcome.
          </p>
        </div>
      </div>
    </div>
  )
}

function OutcomeCard({
  label,
  meaning,
  probability,
  accent,
  icon: Icon,
}: {
  label: string
  meaning: string
  probability: number
  accent: 'mint' | 'rose'
  icon: typeof CheckCircle2
}) {
  const isMint = accent === 'mint'
  const color = isMint ? 'text-mint-300' : 'text-rose-risk'
  const barColor = isMint ? 'bg-mint-400' : 'bg-rose-risk'
  const borderColor = isMint ? 'border-mint-400/25' : 'border-rose-400/25'
  const bgColor = isMint ? 'bg-mint-400/10' : 'bg-rose-400/10'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className={`card-surface rounded-2xl border ${borderColor} p-5`}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className={`font-mono text-sm font-semibold tracking-wide ${color}`}>{label}</p>
          <p className="mt-0.5 text-sm text-slate-300">{meaning}</p>
        </div>
        <div className={`inline-flex h-9 w-9 items-center justify-center rounded-lg ${bgColor}`}>
          <Icon className={`h-5 w-5 ${color}`} strokeWidth={1.9} />
        </div>
      </div>
      <p className="mt-4 font-display text-4xl font-bold tracking-tight text-slate-50">
        {formatPercent(probability, 2)}
      </p>
      <p className="mt-1 font-mono text-xs text-slate-500">P = {formatProbability(probability, 4)}</p>
      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-ink-700">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${probability * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={`h-full rounded-full ${barColor}`}
        />
      </div>
    </motion.div>
  )
}
