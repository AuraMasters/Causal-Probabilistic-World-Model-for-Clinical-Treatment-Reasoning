import { motion } from 'framer-motion'
import { ArrowUpRight, BadgeCheck, Info } from 'lucide-react'
import type { AnalyzeResult } from '../lib/types'
import { formatPercent } from '../lib/format'

export function RecommendedTreatment({ result }: { result: AnalyzeResult }) {
  const recommended = result.recommended

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="relative overflow-hidden rounded-2xl border border-mint-400/30 bg-gradient-to-br from-mint-400/10 via-ink-850 to-ink-900 p-6 sm:p-8"
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-mint-400/60 to-transparent" />

      <div className="flex flex-wrap items-center gap-2">
        <BadgeCheck className="h-5 w-5 text-mint-300" strokeWidth={2} />
        <p className="font-mono text-[11px] font-semibold tracking-[0.24em] text-mint-300 uppercase">
          Recommended treatment
        </p>
      </div>

      <h3 className="mt-3 font-display text-3xl font-bold tracking-tight text-slate-50 sm:text-4xl">
        {recommended.name}
      </h3>
      <p className="mt-1 font-mono text-sm text-slate-400">
        Treatment {recommended.treatment} · {recommended.short_name}
      </p>

      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Metric label="Predicted desirable outcome" value={formatPercent(recommended.p_label_0, 2)} accent="mint" />
        <Metric label="Predicted undesirable outcome" value={formatPercent(recommended.p_label_1, 2)} accent="rose" />
        <Metric label="Expected utility" value={recommended.expected_utility.toFixed(4)} accent="cyan" />
      </div>

      <div className="mt-6 flex items-start gap-2.5 rounded-xl border border-slate-500/20 bg-ink-900/60 px-4 py-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" strokeWidth={2} />
        <div className="flex items-start gap-2 text-xs leading-relaxed text-slate-400">
          <p>
            <span className="text-slate-300">Reason:</span> highest predicted desirable outcome / lowest predicted
            undesirable outcome among the four treatments.
          </p>
          <ArrowUpRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-mint-300" />
        </div>
      </div>
    </motion.div>
  )
}

function Metric({ label, value, accent }: { label: string; value: string; accent: 'mint' | 'rose' | 'cyan' }) {
  const color =
    accent === 'mint'
      ? 'text-mint-300'
      : accent === 'rose'
        ? 'text-rose-risk'
        : 'text-cyan-300'
  return (
    <div className="rounded-xl border border-slate-500/20 bg-ink-900/70 px-4 py-3.5">
      <p className="text-[11px] font-medium tracking-wide text-slate-500 uppercase">{label}</p>
      <p className={`mt-1 font-display text-2xl font-bold ${color}`}>{value}</p>
    </div>
  )
}
