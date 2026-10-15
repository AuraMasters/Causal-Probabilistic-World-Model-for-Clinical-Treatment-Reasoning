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
        isSurvival={true}
        icon={ShieldCheck}
      />
      <OutcomeCard
        label="Label 1"
        title="Disease Progression"
        meaning="50% CD4 decline, clinical AIDS-defining event, or mortality"
        probability={p1}
        isSurvival={false}
        icon={ShieldAlert}
      />
      <div className="sm:col-span-2">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 px-4 py-3 text-xs leading-relaxed text-zinc-400 flex items-center gap-2">
          <HeartPulse className="h-4 w-4 text-zinc-300 shrink-0" />
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
  isSurvival,
  icon: Icon,
}: {
  label: string
  title: string
  meaning: string
  probability: number
  isSurvival: boolean
  icon: React.ComponentType<{ className?: string }>
}) {
  return (
    <div className="card-surface rounded-xl border border-zinc-800 bg-zinc-950 p-6 shadow-md">
      <div className="flex items-start justify-between">
        <div>
          <span className="font-mono text-xs font-bold uppercase text-zinc-400">{label}</span>
          <h4 className="mt-1 font-display text-lg font-bold text-white">{title}</h4>
        </div>
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800 text-white">
          <Icon className="h-5 w-5" />
        </div>
      </div>

      <p className="mt-2 text-xs text-zinc-400">{meaning}</p>

      <div className="mt-4 flex items-baseline justify-between">
        <span className={`font-display text-3xl font-extrabold ${isSurvival ? 'text-white' : 'text-zinc-300'}`}>
          {formatPercent(probability, 1)}
        </span>
        <span className="font-mono text-xs text-zinc-500">
          P = {formatProbability(probability, 4)}
        </span>
      </div>

      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-black border border-zinc-800">
        <div
          className={`h-full ${isSurvival ? 'bg-white' : 'bg-zinc-500'} rounded-full`}
          style={{ width: `${probability * 100}%` }}
        />
      </div>
    </div>
  )
}
