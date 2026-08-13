import { motion } from 'framer-motion'
import { Award } from 'lucide-react'
import type { TreatmentResult } from '../lib/types'
import { formatPercent, formatProbability, formatUtility } from '../lib/format'

export function TreatmentComparison({ treatments }: { treatments: TreatmentResult[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-left">
        <thead>
          <tr className="border-b border-slate-600/30">
            <th className="px-4 py-3 font-mono text-[11px] font-medium tracking-[0.18em] text-slate-500 uppercase">
              Treatment
            </th>
            <th className="px-4 py-3 text-right font-mono text-[11px] font-medium tracking-[0.18em] text-slate-500 uppercase">
              P(Label 0)
            </th>
            <th className="px-4 py-3 text-right font-mono text-[11px] font-medium tracking-[0.18em] text-slate-500 uppercase">
              P(Label 1)
            </th>
            <th className="px-4 py-3 text-right font-mono text-[11px] font-medium tracking-[0.18em] text-slate-500 uppercase">
              Expected utility
            </th>
            <th className="px-4 py-3 text-center font-mono text-[11px] font-medium tracking-[0.18em] text-slate-500 uppercase">
              Rank
            </th>
          </tr>
        </thead>
        <tbody>
          {[...treatments]
            .sort((a, b) => a.rank - b.rank)
            .map((row, index) => (
              <motion.tr
                key={row.treatment}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.06, duration: 0.35 }}
                className={`border-b border-slate-600/15 transition-colors ${
                  row.is_recommended
                    ? 'bg-mint-400/10 hover:bg-mint-400/15'
                    : 'hover:bg-ink-800/60'
                }`}
              >
                <td className="px-4 py-3.5">
                  <div className="flex items-center gap-3">
                    {row.is_recommended ? (
                      <Award className="h-4 w-4 shrink-0 text-mint-300" />
                    ) : (
                      <span className="h-4 w-4 shrink-0 rounded-full border border-slate-600/50" />
                    )}
                    <div>
                      <p className={`text-sm font-medium ${row.is_recommended ? 'text-mint-200' : 'text-slate-200'}`}>
                        {row.name}
                      </p>
                      <p className="font-mono text-[11px] text-slate-500">trt {row.treatment}</p>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3.5 text-right">
                  <span className="font-mono text-sm text-slate-200">
                    {formatPercent(row.p_label_0, 2)}
                    <span className="ml-1.5 text-[11px] text-slate-500">({formatProbability(row.p_label_0, 4)})</span>
                  </span>
                </td>
                <td className="px-4 py-3.5 text-right">
                  <span
                    className={`font-mono text-sm ${
                      row.p_label_1 === Math.min(...treatments.map((t) => t.p_label_1))
                        ? 'text-mint-300'
                        : 'text-slate-200'
                    }`}
                  >
                    {formatPercent(row.p_label_1, 2)}
                  </span>
                </td>
                <td className="px-4 py-3.5 text-right font-mono text-sm text-slate-200">
                  {formatUtility(row.expected_utility, 4)}
                </td>
                <td className="px-4 py-3.5 text-center">
                  <span
                    className={`inline-flex h-7 w-7 items-center justify-center rounded-full font-mono text-xs font-semibold ${
                      row.is_recommended ? 'bg-mint-400 text-ink-950' : 'bg-ink-700 text-slate-300'
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
