import { motion } from 'framer-motion'
import { ArrowDown, Check } from 'lucide-react'
import { formatPercent } from '../lib/format'
import type { TreatmentResult } from '../lib/types'

const RANK_BADGES = [
  'bg-mint-400 text-ink-950 font-extrabold shadow-md shadow-mint-400/20',
  'bg-cyan-400 text-ink-950 font-extrabold',
  'bg-slate-700 text-slate-200 font-bold',
  'bg-slate-800 text-slate-400 font-semibold',
]

export function TreatmentRanking({ ranking }: { ranking: TreatmentResult[] }) {
  return (
    <div className="flex flex-col space-y-2">
      {ranking.map((row, index) => (
        <div key={row.treatment} className="flex flex-col">
          <motion.div
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.07, duration: 0.35 }}
            className={`flex items-center gap-3.5 rounded-xl border px-4 py-3.5 transition-all ${
              row.is_recommended
                ? 'border-mint-400/40 bg-gradient-to-r from-mint-500/10 via-ink-850 to-ink-900 shadow-md shadow-mint-500/5'
                : 'border-slate-700/40 bg-ink-900/60 hover:border-slate-600/50'
            }`}
          >
            <span
              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg font-mono text-sm ${
                RANK_BADGES[index] ?? 'bg-ink-700 text-slate-300'
              }`}
            >
              #{index + 1}
            </span>

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <p
                  className={`truncate text-sm font-bold ${
                    row.is_recommended ? 'text-mint-200' : 'text-slate-100'
                  }`}
                >
                  {row.name}
                </p>
                {row.is_recommended && (
                  <span className="hidden sm:inline-flex items-center gap-1 rounded-full bg-mint-400/15 border border-mint-400/30 px-2 py-0.5 text-[10px] font-bold tracking-wide text-mint-300 uppercase">
                    <Check className="h-3 w-3" /> Top Choice
                  </span>
                )}
              </div>
              <p className="font-mono text-[11px] text-slate-400">
                Arm {row.treatment} &middot; {row.short_name}
              </p>
            </div>

            {/* Probability preview bar */}
            <div className="hidden md:flex flex-col items-end w-32 shrink-0">
              <div className="flex items-center justify-between w-full text-[11px] font-mono text-slate-300">
                <span>P(surv):</span>
                <span className="font-bold text-mint-300">{formatPercent(row.p_label_0, 1)}</span>
              </div>
              <div className="h-1.5 w-full bg-slate-800 rounded-full mt-1 overflow-hidden">
                <div
                  className="h-full bg-mint-400 rounded-full"
                  style={{ width: `${row.p_label_0 * 100}%` }}
                />
              </div>
            </div>

            <div className="text-right shrink-0">
              <p className="font-mono text-sm font-bold text-white">{formatPercent(row.p_label_0, 2)}</p>
              <p className="text-[10px] font-semibold tracking-wider text-slate-400 uppercase">P(No Progression)</p>
            </div>
          </motion.div>

          {index < ranking.length - 1 && (
            <div className="flex justify-center py-1">
              <ArrowDown className="h-3 w-3 text-slate-600" />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
