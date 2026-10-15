import { motion } from 'framer-motion'
import { Award, CheckCircle2 } from 'lucide-react'
import { formatPercent, formatProbability, formatUtility } from '../lib/format'
import type { TreatmentResult } from '../lib/types'

export function TreatmentComparison({ treatments }: { treatments: TreatmentResult[] }) {
  const sorted = [...treatments].sort((a, b) => a.rank - b.rank)

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 shadow-xl">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[700px] border-collapse text-left">
          <thead>
            <tr className="border-b border-zinc-800 bg-black">
              <th className="px-5 py-3.5 font-mono text-xs font-bold tracking-[0.16em] text-zinc-400 uppercase">
                Treatment Regimen
              </th>
              <th className="px-5 py-3.5 text-right font-mono text-xs font-bold tracking-[0.16em] text-zinc-300 uppercase">
                P(Survival · 0)
              </th>
              <th className="px-5 py-3.5 text-right font-mono text-xs font-bold tracking-[0.16em] text-zinc-400 uppercase">
                P(Progression · 1)
              </th>
              <th className="px-5 py-3.5 text-right font-mono text-xs font-bold tracking-[0.16em] text-zinc-300 uppercase">
                Expected Utility
              </th>
              <th className="px-5 py-3.5 text-center font-mono text-xs font-bold tracking-[0.16em] text-zinc-400 uppercase">
                Rank
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/80">
            {sorted.map((row, index) => {
              const isRecommended = row.is_recommended

              return (
                <motion.tr
                  key={row.treatment}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.04, duration: 0.25 }}
                  className={`transition-colors ${
                    isRecommended
                      ? 'bg-zinc-900/80 hover:bg-zinc-900'
                      : 'hover:bg-zinc-900/30'
                  }`}
                >
                  <td className="px-5 py-3.5">
                    <div className="flex items-center gap-3">
                      {isRecommended ? (
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800 text-white shadow-sm">
                          <Award className="h-4 w-4 text-white" />
                        </div>
                      ) : (
                        <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-800 bg-black font-mono text-xs font-bold text-zinc-400">
                          {row.treatment}
                        </span>
                      )}
                      <div>
                        <div className="flex items-center gap-2">
                          <p className={`text-sm font-bold ${isRecommended ? 'text-white' : 'text-zinc-200'}`}>
                            {row.name}
                          </p>
                          {isRecommended && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-zinc-800 border border-zinc-700 px-2 py-0.5 text-[10px] font-bold text-white uppercase font-mono">
                              <CheckCircle2 className="h-2.5 w-2.5 text-white" /> Optimal
                            </span>
                          )}
                        </div>
                        <p className="font-mono text-xs text-zinc-400">
                          Arm {row.treatment} &middot; {row.short_name}
                        </p>
                      </div>
                    </div>
                  </td>

                  <td className="px-5 py-3.5 text-right">
                    <div className="flex flex-col items-end">
                      <span className="font-mono text-sm font-bold text-white">
                        {formatPercent(row.p_label_0, 2)}
                      </span>
                      <div className="mt-1 h-1.5 w-24 rounded-full bg-black border border-zinc-800 overflow-hidden">
                        <div
                          className="h-full bg-white rounded-full"
                          style={{ width: `${row.p_label_0 * 100}%` }}
                        />
                      </div>
                    </div>
                  </td>

                  <td className="px-5 py-3.5 text-right">
                    <span className="font-mono text-sm font-semibold text-zinc-300">
                      {formatPercent(row.p_label_1, 2)}
                    </span>
                    <p className="font-mono text-[11px] text-zinc-500">
                      P = {formatProbability(row.p_label_1, 4)}
                    </p>
                  </td>

                  <td className="px-5 py-3.5 text-right font-mono text-sm font-bold text-white">
                    {formatUtility(row.expected_utility, 4)}
                  </td>

                  <td className="px-5 py-3.5 text-center">
                    <span
                      className={`inline-flex h-7 w-7 items-center justify-center rounded-lg font-mono text-xs font-bold ${
                        isRecommended
                          ? 'bg-white text-black shadow-sm'
                          : 'bg-zinc-900 text-zinc-400 border border-zinc-800'
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
