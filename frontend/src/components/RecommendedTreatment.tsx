import { motion } from 'framer-motion'
import { ArrowUpRight, CheckCircle2, ShieldCheck, Zap } from 'lucide-react'
import { formatPercent } from '../lib/format'
import type { AnalyzeResult } from '../lib/types'

export function RecommendedTreatment({ result }: { result: AnalyzeResult }) {
  const recommended = result.recommended

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="relative overflow-hidden rounded-2xl border border-mint-400/40 bg-gradient-to-br from-mint-500/15 via-ink-850 to-ink-900 p-6 sm:p-8 shadow-xl shadow-mint-500/5"
    >
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-mint-300 to-transparent" />

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 rounded-full border border-mint-400/30 bg-mint-400/10 px-3 py-1">
          <CheckCircle2 className="h-4 w-4 text-mint-300" strokeWidth={2.4} />
          <p className="font-mono text-[11px] font-bold tracking-[0.2em] text-mint-300 uppercase">
            Optimal Causal Recommendation
          </p>
        </div>
        <span className="font-mono text-xs font-semibold text-slate-400">
          Rank #1 &middot; Maximum Expected Utility
        </span>
      </div>

      <div className="mt-4 flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
        <div>
          <h3 className="font-display text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            {recommended.name}
          </h3>
          <p className="mt-1 font-mono text-sm text-cyan-300">
            Arm {recommended.treatment} &middot; {recommended.short_name}
          </p>
        </div>
      </div>

      {/* Outcome Probability Breakdown Bar */}
      <div className="mt-6 rounded-xl border border-slate-700/50 bg-ink-950/70 p-4">
        <div className="mb-2 flex items-center justify-between text-xs font-semibold">
          <span className="flex items-center gap-1.5 text-mint-300">
            <ShieldCheck className="h-4 w-4" /> Progression-Free Survival P(label=0)
          </span>
          <span className="font-mono text-mint-300 font-bold">{formatPercent(recommended.p_label_0, 2)}</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-800 flex">
          <div
            className="h-full bg-gradient-to-r from-mint-400 to-mint-500 transition-all duration-700"
            style={{ width: `${recommended.p_label_0 * 100}%` }}
          />
          <div
            className="h-full bg-gradient-to-r from-rose-500 to-rose-600 transition-all duration-700"
            style={{ width: `${recommended.p_label_1 * 100}%` }}
          />
        </div>
        <div className="mt-2 flex justify-between text-[11px] text-slate-400 font-mono">
          <span>Desirable: {formatPercent(recommended.p_label_0, 2)}</span>
          <span className="text-rose-400">Progression Risk: {formatPercent(recommended.p_label_1, 2)}</span>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Metric label="Progression-Free (P=0)" value={formatPercent(recommended.p_label_0, 2)} accent="mint" />
        <Metric label="Disease Progression (P=1)" value={formatPercent(recommended.p_label_1, 2)} accent="rose" />
        <Metric label="Pearlian Utility" value={recommended.expected_utility.toFixed(4)} accent="cyan" />
      </div>

      <div className="mt-6 flex items-start gap-2.5 rounded-xl border border-slate-600/30 bg-ink-900/80 px-4 py-3.5">
        <Zap className="mt-0.5 h-4 w-4 shrink-0 text-cyan-300" strokeWidth={2} />
        <div className="flex items-start gap-2 text-xs leading-relaxed text-slate-300">
          <p>
            <strong className="text-white">Interventional Rationale:</strong> Among all four evaluated arms,{' '}
            <span className="text-mint-300 font-semibold">{recommended.name}</span> delivers the lowest causal risk of
            CD4 decline or failure under interventional do-calculus.
          </p>
          <ArrowUpRight className="mt-0.5 h-4 w-4 shrink-0 text-mint-300" />
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
    <div className="rounded-xl border border-slate-700/40 bg-ink-900/80 px-4 py-3.5 shadow-sm">
      <p className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">{label}</p>
      <p className={`mt-1 font-display text-2xl font-extrabold tracking-tight ${color}`}>{value}</p>
    </div>
  )
}
