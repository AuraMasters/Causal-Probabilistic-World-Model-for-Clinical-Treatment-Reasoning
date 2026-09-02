import { motion } from 'framer-motion'
import { HeartPulse, ShieldAlert, ShieldCheck } from 'lucide-react'
import type { AnalyzeResult } from '../lib/types'
import { formatPercent, formatProbability } from '../lib/format'

export function OutcomeSection({ result }: { result: AnalyzeResult }) {
  const recommended = result.recommended

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <OutcomeCard
        label="Label 0"
        title="Progression-Free Survival"
        meaning="Preserved CD4 counts & absence of clinical AIDS endpoints"
        probability={recommended.p_label_0}
        accent="mint"
        icon={ShieldCheck}
      />
      <OutcomeCard
        label="Label 1"
        title="Disease Progression"
        meaning="50% CD4 decline, clinical AIDS-defining event, or mortality"
        probability={recommended.p_label_1}
        accent="rose"
        icon={ShieldAlert}
      />
      <div className="sm:col-span-2">
        <div className="rounded-xl border border-slate-700/60 bg-ink-950/70 px-4 py-3 text-xs leading-relaxed text-slate-300 flex items-center gap-2">
          <HeartPulse className="h-4 w-4 text-cyan-300 shrink-0" />
          <p>
            Predicted under <strong className="text-white">{recommended.name}</strong> (Arm {recommended.treatment}). Probabilities sum strictly to 1.0000 by Bayesian normalization.
          </p>
        </div>
      </div>
    </div>
  )
}

function OutcomeCard({
  label,
  title,
  meaning,
  probability,
  accent,
  icon: Icon,
}: {
  label: string
  title: string
  meaning: string
  probability: number
  accent: 'mint' | 'rose'
  icon: typeof ShieldCheck
}) {
  const isMint = accent === 'mint'
  const color = isMint ? 'text-mint-200' : 'text-rose-risk'
  const barColor = isMint ? 'bg-gradient-to-r from-mint-400 to-mint-300' : 'bg-rose-500'
  const borderColor = isMint ? 'border-mint-400/30' : 'border-rose-500/30'
  const bgColor = isMint ? 'bg-mint-400/10' : 'bg-rose-500/10'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className={`card-surface rounded-2xl border ${borderColor} p-5 shadow-lg`}
    >
      <div className="flex items-start justify-between">
        <div>
          <span className={`font-mono text-xs font-bold tracking-wider ${color} uppercase`}>
            {label} &middot; {isMint ? 'Desirable' : 'Risk'}
          </span>
          <p className="mt-1 font-display text-base font-bold text-white">{title}</p>
          <p className="mt-0.5 text-xs text-slate-400 leading-snug">{meaning}</p>
        </div>
        <div className={`inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${bgColor}`}>
          <Icon className={`h-5 w-5 ${color}`} strokeWidth={2} />
        </div>
      </div>

      <div className="mt-5 flex items-baseline justify-between">
        <p className="font-display text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          {formatPercent(probability, 2)}
        </p>
        <span className="font-mono text-xs text-slate-400">
          P = {formatProbability(probability, 4)}
        </span>
      </div>

      <div className="mt-3.5 h-2 w-full overflow-hidden rounded-full bg-slate-900 border border-slate-800">
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
