import { motion } from 'framer-motion'
import { Award, CheckCircle2 } from 'lucide-react'
import { formatPercent, formatProbability, formatUtility } from '../lib/format'
import type { TreatmentResult } from '../lib/types'

export function TreatmentComparison({ treatments }: { treatments: TreatmentResult[] }) {
  const sorted = [...treatments].sort((a, b) => a.rank - b.rank)

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-700/60 bg-ink-900/90 shadow-xl">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[700px] border-collapse text-left">
          <thead>
            <tr className="border-b border-slate-700/70 bg-ink-950/80">
              <th className="px-5 py-4 font-mono text-xs font-bold tracking-[0.16em] text-slate-300 uppercase">
                Treatment Regimen
              </th>
              <th className="px-5 py-4 text-right font-mono text-xs font-bold tracking-[0.16em] text-mint-200 uppercase">
                P(Survival · 0)
              </th>
              <th className="px-5 py-4 text-right font-mono text-xs font-bold tracking-[0.16em] text-rose-300 uppercase">
                P(Progression · 1)
              </th>
              <th className="px-5 py-4 text-right font-mono text-xs font-bold tracking-[0.16em] text-cyan-200 uppercase">
                Expected Utility
              </th>
              <th className="px-5 py-4 text-center font-mono text-xs font-bold tracking-[0.16em] text-slate-300 uppercase">
                Rank
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {sorted.map((row, index) => {
              const isRecommended = row.is_recommended

              return (
                <motion.tr
                  key={row.treatment}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05, duration: 0.3 }}
                  className={`transition-colors ${
                    isRecommended
                      ? 'bg-cyan-400/10 hover:bg-cyan-400/15'
                      : 'hover:bg-slate-800/40'
                  }`}
                >
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      {isRecommended ? (
                        <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-cyan-400/20 border border-cyan-400/40 shadow-sm shadow-cyan-400/20">
                          <Award className="h-4 w-4 text-cyan-300" />
                        </div>
                      ) : (
                        <span className="flex h-8 w-8 items-center justify-center rounded-xl border border-slate-700 bg-ink-950 font-mono text-xs font-bold text-slate-400">
                          {row.treatment}
                        </span>
                      )}
                      <div>
                        <div className="flex items-center gap-2">
                          <p className={`text-sm font-bold ${isRecommended ? 'text-white' : 'text-slate-200'}`}>
                            {row.name}
                          </p>
                          {isRecommended && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-mint-300/15 border border-mint-300/35 px-2 py-0.5 text-[10px] font-bold text-mint-200 uppercase font-mono">
                              <CheckCircle2 className="h-2.5 w-2.5 text-mint-300" /> Optimal
                            </span>
                          )}
                        </div>
                        <p className="font-mono text-xs text-slate-400">
                          Arm {row.treatment} &middot; {row.short_name}
                        </p>
                      </div>
                    </div>
                  </td>

                  <td className="px-5 py-4 text-right">
                    <div className="flex flex-col items-end">
                      <span className="font-mono text-sm font-bold text-mint-200">
                        {formatPercent(row.p_label_0, 2)}
                      </span>
                      <div className="mt-1 h-1.5 w-24 rounded-full bg-slate-800 overflow-hidden">
                        <div
                          className="h-full bg-mint-400 rounded-full"
                          style={{ width: `${row.p_label_0 * 100}%` }}
                        />
                      </div>
                    </div>
                  </td>

                  <td className="px-5 py-4 text-right">
                    <span
                      className={`font-mono text-sm font-semibold ${
                        row.p_label_1 === Math.min(...treatments.map((t) => t.p_label_1))
                          ? 'text-mint-200 font-bold'
                          : 'text-rose-risk'
                      }`}
                    >
                      {formatPercent(row.p_label_1, 2)}
                    </span>
                    <p className="font-mono text-[11px] text-slate-400">
                      P = {formatProbability(row.p_label_1, 4)}
                    </p>
                  </td>

                  <td className="px-5 py-4 text-right font-mono text-sm font-bold text-cyan-200">
                    {formatUtility(row.expected_utility, 4)}
                  </td>

                  <td className="px-5 py-4 text-center">
                    <span
                      className={`inline-flex h-7 w-7 items-center justify-center rounded-lg font-mono text-xs font-bold ${
                        isRecommended
                          ? 'bg-gradient-to-br from-cyan-400 to-mint-300 text-ink-950 shadow-md shadow-cyan-400/20'
                          : 'bg-slate-800 text-slate-300 border border-slate-700'
                      }`}
                    >
                      #{row.rank}
                    </span>
                  </td>
                </motion.tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
