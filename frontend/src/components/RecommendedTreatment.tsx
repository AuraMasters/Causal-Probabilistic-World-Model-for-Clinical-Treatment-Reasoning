import { motion } from 'framer-motion'
import {
  Activity,
  Award,
  CheckCircle2,
  ShieldCheck,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { formatPercent, formatProbability } from '../lib/format'
import type { AnalyzeResult } from '../lib/types'

export function RecommendedTreatment({ result }: { result: AnalyzeResult }) {
  const recommended = result.recommended

  // Determine drug classification description
  const drugClass =
    recommended.treatment === 0
      ? 'Monotherapy (Zidovudine)'
      : recommended.treatment === 3
        ? 'Monotherapy (Didanosine)'
        : 'Combination Antiretroviral Therapy (Dual NRTI)'

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="relative overflow-hidden rounded-2xl border border-cyan-400/35 bg-gradient-to-br from-ink-850 via-ink-900 to-ink-950 p-6 sm:p-9 shadow-2xl shadow-cyan-950/40"
    >
      {/* Top glowing ambient highlight */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-cyan-400 via-mint-300 to-cyan-400 opacity-80" />

      {/* Decision Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-400/15 pb-5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-400/20 border border-cyan-400/40 shadow-sm shadow-cyan-400/20">
            <Award className="h-5 w-5 text-cyan-300" strokeWidth={2.2} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold tracking-[0.18em] text-cyan-300 uppercase">
                AI Optimal Regimen Selection
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-mint-300/15 border border-mint-300/30 px-2 py-0.5 font-mono text-[10px] font-bold text-mint-200 uppercase">
                <CheckCircle2 className="h-3 w-3 text-mint-300" /> Rank #1
              </span>
            </div>
            <p className="font-sans text-xs text-slate-400 mt-0.5">
              Maximizes Pearlian Expected Utility under interventional conditioning
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="rounded-lg border border-slate-700/60 bg-ink-950/80 px-3 py-1 font-mono text-xs font-semibold text-slate-300">
            Arm {recommended.treatment}
          </span>
          <span className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 font-mono text-xs font-bold text-cyan-200">
            {recommended.short_name}
          </span>
        </div>
      </div>

      {/* Regimen Hero Info */}
      <div className="mt-6 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="font-mono text-xs font-semibold tracking-wider text-slate-400 uppercase">
            Recommended Pharmaceutical Strategy
          </p>
          <h3 className="mt-1 font-display text-3xl font-extrabold tracking-tight text-white sm:text-4xl lg:text-5xl">
            {recommended.name}
          </h3>
          <p className="mt-2 flex items-center gap-2 text-sm text-cyan-200/90 font-medium">
            <span className="h-2 w-2 rounded-full bg-mint-300 inline-block animate-pulse" />
            {drugClass} &middot; Optimal counterfactual risk profile
          </p>
        </div>
      </div>

      {/* 3 Core Metric Pillar Cards */}
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="card-surface rounded-2xl border border-mint-300/30 bg-gradient-to-br from-mint-300/10 to-ink-900/90 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-mint-200 uppercase">
              Progression-Free
            </span>
            <ShieldCheck className="h-4 w-4 text-mint-300" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-mint-200 sm:text-5xl">
            {formatPercent(recommended.p_label_0, 1)}
          </p>
          <p className="mt-1 font-mono text-xs text-mint-300/80">
            P(label = 0) = {formatProbability(recommended.p_label_0, 4)}
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-950">
            <div
              className="h-full bg-gradient-to-r from-mint-300 to-mint-400 rounded-full"
              style={{ width: `${recommended.p_label_0 * 100}%` }}
            />
          </div>
        </div>

        <div className="card-surface rounded-2xl border border-rose-500/30 bg-gradient-to-br from-rose-500/10 to-ink-900/90 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-rose-300 uppercase">
              Progression Risk
            </span>
            <Activity className="h-4 w-4 text-rose-risk" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-rose-risk sm:text-5xl">
            {formatPercent(recommended.p_label_1, 1)}
          </p>
          <p className="mt-1 font-mono text-xs text-rose-300/80">
            P(label = 1) = {formatProbability(recommended.p_label_1, 4)}
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-950">
            <div
              className="h-full bg-rose-500 rounded-full"
              style={{ width: `${recommended.p_label_1 * 100}%` }}
            />
          </div>
        </div>

        <div className="card-surface rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-cyan-400/10 to-ink-900/90 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-cyan-200 uppercase">
              Expected Utility
            </span>
            <TrendingUp className="h-4 w-4 text-cyan-300" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-cyan-200 sm:text-5xl">
            {recommended.expected_utility.toFixed(4)}
          </p>
          <p className="mt-1 font-mono text-xs text-cyan-300/80">
            Scale: 0.0000 – 1.0000 max
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-950">
            <div
              className="h-full bg-cyan-400 rounded-full"
              style={{ width: `${recommended.expected_utility * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Full Segmented Probability Breakdown Bar */}
      <div className="mt-6 rounded-2xl border border-slate-700/60 bg-ink-950/80 p-5">
        <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2 text-xs font-semibold">
          <span className="flex items-center gap-1.5 text-mint-200 font-mono">
            <span className="h-2.5 w-2.5 rounded-full bg-mint-300" />
            Survival P(label = 0): {formatPercent(recommended.p_label_0, 2)}
          </span>
          <span className="flex items-center gap-1.5 text-rose-300 font-mono">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-risk" />
            Progression P(label = 1): {formatPercent(recommended.p_label_1, 2)}
          </span>
        </div>
        <div className="flex h-3.5 w-full overflow-hidden rounded-full bg-slate-900 p-0.5 border border-slate-700/50">
          <div
            className="h-full rounded-l-full bg-gradient-to-r from-mint-400 to-mint-300 transition-all duration-700"
            style={{ width: `${recommended.p_label_0 * 100}%` }}
          />
          <div
            className="h-full rounded-r-full bg-gradient-to-r from-rose-500 to-rose-600 transition-all duration-700"
            style={{ width: `${recommended.p_label_1 * 100}%` }}
          />
        </div>
      </div>

      {/* Pearlian Causal Rationale Callout */}
      <div className="mt-6 flex items-start gap-3 rounded-2xl border border-cyan-400/25 bg-gradient-to-r from-cyan-950/40 via-ink-900 to-ink-950 p-4.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan-400/15 border border-cyan-400/30">
          <Zap className="h-4 w-4 text-cyan-300" strokeWidth={2.2} />
        </div>
        <div className="flex-1 text-xs leading-relaxed text-slate-300">
          <p className="font-semibold text-white">Causal Interventional Decision Rationale</p>
          <p className="mt-1">
            Conditioning on this patient's specific baseline biomarkers (CD4, CD8, Karnofsky, and prior ART history),{' '}
            <strong className="text-mint-200">{recommended.name}</strong> yields the lowest posterior probability of disease progression among all four randomized treatment strategies in the ACTG175 Bayesian World Model.
          </p>
        </div>
      </div>
    </motion.div>
  )
}
