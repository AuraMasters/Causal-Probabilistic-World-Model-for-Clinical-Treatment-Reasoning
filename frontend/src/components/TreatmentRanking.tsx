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
              transition={{ delay: index * 0.06, duration: 0.3 }}
              className={`relative overflow-hidden rounded-xl border p-4.5 transition-all shadow-md ${
                isFirst
                  ? 'border-zinc-500 bg-zinc-900/90'
                  : 'border-zinc-800 bg-zinc-950/70 hover:border-zinc-700'
              }`}
            >
              <div className="flex items-center justify-between gap-3">
                {/* Left: Rank & Name */}
                <div className="flex items-center gap-3.5 min-w-0 flex-1">
                  <span
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg font-mono text-xs font-extrabold shadow-sm ${
                      isFirst
                        ? 'bg-white text-black'
                        : 'bg-zinc-900 text-zinc-400 border border-zinc-800'
                    }`}
                  >
                    #{index + 1}
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p
                        className={`truncate text-sm font-bold ${
                          isFirst ? 'text-white text-base' : 'text-zinc-200'
                        }`}
                      >
                        {row.name}
                      </p>
                      {isFirst && (
                        <span className="hidden sm:inline-flex items-center gap-1 rounded-full bg-zinc-800 border border-zinc-700 px-2.5 py-0.5 text-[10px] font-bold text-white uppercase">
                          <CheckCircle2 className="h-3 w-3 text-white" /> Optimal
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 font-mono text-xs text-zinc-400 mt-0.5">
                      <span>Arm {row.treatment}</span>
                      <span>&middot;</span>
                      <span className="text-zinc-300 font-medium">{row.short_name}</span>
                      {row.treatment !== 0 && diffVsArm0 !== 0 && (
                        <>
                          <span>&middot;</span>
                          <span className="text-white font-bold">
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
                      isFirst ? 'text-white text-lg' : 'text-zinc-300'
                    }`}
                  >
                    {formatPercent(row.p_label_0, 1)}
                  </p>
                  <p className="font-mono text-[11px] text-zinc-500">
                    Utility: {formatUtility(row.expected_utility, 4)}
                  </p>
                </div>
              </div>

              {/* Progress bar */}
              <div className="mt-3.5 h-1.5 w-full bg-black rounded-full overflow-hidden border border-zinc-800">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isFirst ? 'bg-white' : 'bg-zinc-600'
                  }`}
                  style={{ width: `${row.p_label_0 * 100}%` }}
                />
              </div>
            </motion.div>

            {!isLast && (
              <div className="flex justify-center py-1">
                <ArrowDown className="h-3 w-3 text-zinc-700" />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
