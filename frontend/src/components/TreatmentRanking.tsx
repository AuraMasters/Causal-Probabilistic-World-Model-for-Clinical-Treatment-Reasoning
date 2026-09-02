import { motion } from 'framer-motion'
import { ArrowDown, CheckCircle2 } from 'lucide-react'
import { formatPercent, formatUtility } from '../lib/format'
import type { TreatmentResult } from '../lib/types'

export function TreatmentRanking({ ranking }: { ranking: TreatmentResult[] }) {
  // Baseline (Arm 0 - ZDV monotherapy) survival probability
  const arm0 = ranking.find((r) => r.treatment === 0)
  const arm0Surv = arm0 ? arm0.p_label_0 : 0.75

  return (
    <div className="flex flex-col space-y-3">
      {ranking.map((row, index) => {
        const isFirst = index === 0
        const isLast = index === ranking.length - 1
        const diffVsArm0 = row.p_label_0 - arm0Surv

        return (
          <div key={row.treatment} className="flex flex-col">
            <motion.div
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.08, duration: 0.35 }}
              className={`relative overflow-hidden rounded-2xl border p-4.5 transition-all shadow-md ${
                isFirst
                  ? 'border-cyan-400/50 bg-gradient-to-r from-ink-850 via-ink-900 to-ink-950 shadow-cyan-950/40 ring-1 ring-cyan-400/30'
                  : 'border-slate-700/60 bg-ink-900/80 hover:border-slate-600'
              }`}
            >
              {isFirst && (
                <div className="pointer-events-none absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-cyan-400 via-mint-300 to-cyan-400" />
              )}

              <div className="flex items-center justify-between gap-3">
                {/* Left: Rank & Name */}
                <div className="flex items-center gap-3.5 min-w-0 flex-1">
                  <span
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl font-mono text-xs font-extrabold shadow-sm ${
                      isFirst
                        ? 'bg-gradient-to-br from-cyan-400 to-mint-300 text-ink-950 shadow-cyan-400/30'
                        : index === 1
                          ? 'bg-ink-800 text-cyan-200 border border-cyan-400/30'
                          : index === 2
                            ? 'bg-ink-800 text-slate-300 border border-slate-700'
                            : 'bg-ink-950 text-slate-500 border border-slate-800'
                    }`}
                  >
                    #{index + 1}
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p
                        className={`truncate text-sm font-bold ${
                          isFirst ? 'text-white text-base' : 'text-slate-100'
                        }`}
                      >
                        {row.name}
                      </p>
                      {isFirst && (
                        <span className="hidden sm:inline-flex items-center gap-1 rounded-full bg-mint-300/15 border border-mint-300/35 px-2.5 py-0.5 text-[10px] font-bold text-mint-200 uppercase">
                          <CheckCircle2 className="h-3 w-3 text-mint-300" /> Optimal
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 font-mono text-xs text-slate-400 mt-0.5">
                      <span>Arm {row.treatment}</span>
                      <span>&middot;</span>
                      <span className="text-cyan-200/80 font-semibold">{row.short_name}</span>
                      {row.treatment !== 0 && diffVsArm0 !== 0 && (
                        <>
                          <span>&middot;</span>
                          <span
                            className={
                              diffVsArm0 > 0 ? 'text-mint-200 font-bold' : 'text-rose-risk font-bold'
                            }
                          >
                            {diffVsArm0 > 0 ? '+' : ''}
                            {(diffVsArm0 * 100).toFixed(1)}% vs ZDV
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right: Numbers */}
                <div className="text-right shrink-0">
                  <p
                    className={`font-mono text-base font-extrabold ${
                      isFirst ? 'text-mint-200 text-lg' : 'text-white'
                    }`}
                  >
                    {formatPercent(row.p_label_0, 1)}
                  </p>
                  <p className="font-mono text-[11px] text-slate-400">
                    Utility: {formatUtility(row.expected_utility, 4)}
                  </p>
                </div>
              </div>

              {/* Progress bar */}
              <div className="mt-3.5 h-1.5 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isFirst
                      ? 'bg-gradient-to-r from-cyan-400 to-mint-300'
                      : 'bg-slate-500'
                  }`}
                  style={{ width: `${row.p_label_0 * 100}%` }}
                />
              </div>
            </motion.div>

            {!isLast && (
              <div className="flex justify-center py-1">
                <ArrowDown className="h-3 w-3 text-slate-700" />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
