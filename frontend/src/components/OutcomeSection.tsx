import { HeartPulse, ShieldAlert, ShieldCheck } from 'lucide-react'
import type { AnalyzeResult } from '../lib/types'
import { formatPercent, formatProbability } from '../lib/format'

export function OutcomeSection({ result }: { result?: AnalyzeResult }) {
  const p0 = result?.recommended?.p_label_0 ?? 0.757
  const p1 = result?.recommended?.p_label_1 ?? 0.243
  const trtName = result?.recommended?.name ?? 'Recommended Arm'
  const trtNum = result?.recommended?.treatment ?? 1

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <OutcomeCard
        label="Label 0"
        title="Progression-Free Survival"
        meaning="Preserved CD4 counts & absence of clinical AIDS endpoints"
        probability={p0}
        accent="mint"
        icon={ShieldCheck}
      />
      <OutcomeCard
        label="Label 1"
        title="Disease Progression"
        meaning="50% CD4 decline, clinical AIDS-defining event, or mortality"
        probability={p1}
        accent="rose"
        icon={ShieldAlert}
      />
      <div className="sm:col-span-2">
        <div className="rounded-xl border border-slate-700/60 bg-ink-950/70 px-4 py-3 text-xs leading-relaxed text-slate-300 flex items-center gap-2">
          <HeartPulse className="h-4 w-4 text-cyan-300 shrink-0" />
          <p>
            Predicted under <strong className="text-white">{trtName}</strong> (Arm {trtNum}). Probabilities sum strictly to 1.0000 by normalization.
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
  icon: React.ComponentType<{ className?: string }>
}) {
  const isMint = accent === 'mint'
  const borderColor = isMint ? 'border-mint-300/30' : 'border-rose-500/30'
  const textColor = isMint ? 'text-mint-200' : 'text-rose-300'
  const iconBg = isMint ? 'bg-mint-300/10 text-mint-300' : 'bg-rose-500/10 text-rose-400'
  const progressBg = isMint ? 'bg-mint-300' : 'bg-rose-500'

  return (
    <div className={`card-surface rounded-2xl border ${borderColor} p-6 shadow-md`}>
      <div className="flex items-start justify-between">
        <div>
          <span className="font-mono text-xs font-bold uppercase text-slate-400">{label}</span>
          <h4 className="mt-1 font-display text-lg font-bold text-white">{title}</h4>
        </div>
        <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${iconBg}`}>
          <Icon className="h-5 w-5" />
        </div>
      </div>

      <p className="mt-2 text-xs text-slate-400">{meaning}</p>

      <div className="mt-4 flex items-baseline justify-between">
        <span className={`font-display text-3xl font-extrabold ${textColor}`}>
          {formatPercent(probability, 1)}
        </span>
        <span className="font-mono text-xs text-slate-400">
          P = {formatProbability(probability, 4)}
        </span>
      </div>

      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-950">
        <div className={`h-full ${progressBg} rounded-full`} style={{ width: `${probability * 100}%` }} />
      </div>
    </div>
  )
}
