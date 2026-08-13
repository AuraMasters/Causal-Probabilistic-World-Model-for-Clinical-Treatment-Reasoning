import { motion } from 'framer-motion'
import { ArrowDown } from 'lucide-react'
import type { TreatmentResult } from '../lib/types'
import { formatPercent } from '../lib/format'

const RANK_COLORS = ['bg-mint-400 text-ink-950', 'bg-cyan-glow text-ink-950', 'bg-slate-600 text-slate-100', 'bg-ink-700 text-slate-400']

export function TreatmentRanking({ ranking }: { ranking: TreatmentResult[] }) {
  return (
    <div className="flex flex-col">
      {ranking.map((row, index) => (
        <div key={row.treatment} className="flex flex-col">
          <motion.div
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.08, duration: 0.4 }}
            className="flex items-center gap-4 rounded-xl border border-slate-500/15 bg-ink-900/50 px-4 py-3.5"
          >
            <span
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl font-mono text-sm font-bold ${RANK_COLORS[index] ?? 'bg-ink-700 text-slate-300'}`}
            >
              {index + 1}
            </span>
            <div className="min-w-0 flex-1">
              <p className={`truncate text-sm font-medium ${row.is_recommended ? 'text-mint-200' : 'text-slate-200'}`}>
                {row.name}
              </p>
              <p className="font-mono text-[11px] text-slate-500">
                trt {row.treatment} · {row.short_name}
              </p>
            </div>
            <div className="hidden text-right sm:block">
              <p className="font-mono text-sm text-slate-200">{formatPercent(row.p_label_0, 2)}</p>
              <p className="text-[10px] tracking-wide text-slate-500 uppercase">P(label 0)</p>
            </div>
            {row.is_recommended && (
              <span className="rounded-full bg-mint-400/15 px-2.5 py-1 text-[10px] font-semibold tracking-wide text-mint-300 uppercase">
                Recommended
              </span>
            )}
          </motion.div>
          {index < ranking.length - 1 && (
            <div className="flex justify-center py-0.5">
              <ArrowDown className="h-3.5 w-3.5 text-slate-600" />
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
