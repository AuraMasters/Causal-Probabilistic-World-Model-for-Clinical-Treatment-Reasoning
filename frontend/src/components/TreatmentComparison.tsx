import { motion } from 'framer-motion'
import { Award } from 'lucide-react'
import { formatPercent, formatProbability, formatUtility } from '../lib/format'
import type { TreatmentResult } from '../lib/types'

export function TreatmentComparison({ treatments }: { treatments: TreatmentResult[] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-700/50 bg-ink-950/60 shadow-md">
      <table className="w-full min-w-[660px] border-collapse text-left">
        <thead>
          <tr className="border-b border-slate-700/60 bg-ink-900/80">
            <th className="px-4 py-3.5 font-mono text-[11px] font-bold tracking-[0.16em] text-slate-400 uppercase">
              Treatment Arm
            </th>
            <th className="px-4 py-3.5 text-right font-mono text-[11px] font-bold tracking-[0.16em] text-slate-400 uppercase">
              P(Survival · 0)
            </th>
            <th className="px-4 py-3.5 text-right font-mono text-[11px] font-bold tracking-[0.16em] text-slate-400 uppercase">
              P(Progression · 1)
            </th>
            <th className="px-4 py-3.5 text-right font-mono text-[11px] font-bold tracking-[0.16em] text-slate-400 uppercase">
              Expected Utility
            </th>
            <th className="px-4 py-3.5 text-center font-mono text-[11px] font-bold tracking-[0.16em] text-slate-400 uppercase">
              Rank
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/60">
          {[...treatments]
            .sort((a, b) => a.rank - b.rank)
            .map((row, index) => (
              <motion.tr
                key={row.treatment}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05, duration: 0.3 }}
                className={`transition-colors ${
                  row.is_recommended
                    ? 'bg-mint-400/10 hover:bg-mint-400/15'
                    : 'hover:bg-slate-800/40'
                }`}
              >
                <td className="px-4 py-3.5">
                  <div className="flex items-center gap-3">
                    {row.is_recommended ? (
                      <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-mint-400/20 border border-mint-400/40">
                        <Award className="h-4 w-4 text-mint-300" />
                      </div>
                    ) : (
                      <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700/60 bg-ink-900/60 text-slate-500 font-mono text-xs">
                        {row.treatment}
                      </span>
                    )}
                    <div>
                      <p className={`text-sm font-bold ${row.is_recommended ? 'text-mint-200' : 'text-slate-100'}`}>
                        {row.name}
                      </p>
                      <p className="font-mono text-[11px] text-slate-400">Arm {row.treatment} &middot; {row.short_name}</p>
                    </div>
                  </div>
                </td>

                <td className="px-4 py-3.5 text-right">
                  <span className="font-mono text-sm font-bold text-mint-300">
                    {formatPercent(row.p_label_0, 2)}
                  </span>
                  <span className="ml-1.5 font-mono text-[11px] text-slate-400">
                    ({formatProbability(row.p_label_0, 4)})
                  </span>
                </td>

                <td className="px-4 py-3.5 text-right">
                  <span
                    className={`font-mono text-sm font-semibold ${
                      row.p_label_1 === Math.min(...treatments.map((t) => t.p_label_1))
                        ? 'text-mint-300'
                        : 'text-rose-300'
                    }`}
                  >
                    {formatPercent(row.p_label_1, 2)}
                  </span>
                </td>

                <td className="px-4 py-3.5 text-right font-mono text-sm font-semibold text-slate-200">
                  {formatUtility(row.expected_utility, 4)}
                </td>

                <td className="px-4 py-3.5 text-center">
                  <span
                    className={`inline-flex h-7 w-7 items-center justify-center rounded-lg font-mono text-xs font-bold ${
                      row.is_recommended
                        ? 'bg-mint-400 text-ink-950 shadow-md shadow-mint-400/20'
                        : 'bg-slate-800 text-slate-300 border border-slate-700'
                    }`}
                  >
                    {row.rank}
                  </span>
                </td>
              </motion.tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}
