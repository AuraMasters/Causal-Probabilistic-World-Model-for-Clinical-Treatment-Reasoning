import { motion } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'

interface StatCardProps {
  icon: LucideIcon
  label: string
  value: string
  detail?: string
  accent?: 'cyan' | 'violet' | 'mint' | 'amber'
}

const ACCENTS = {
  cyan: 'from-cyan-400/20 to-transparent text-cyan-300 border-cyan-400/20',
  violet: 'from-violet-400/20 to-transparent text-violet-300 border-violet-400/20',
  mint: 'from-mint-400/20 to-transparent text-mint-300 border-mint-400/20',
  amber: 'from-amber-400/20 to-transparent text-amber-300 border-amber-400/20',
}

export function StatCard({ icon: Icon, label, value, detail, accent = 'cyan' }: StatCardProps) {
  return (
    <motion.div
      whileHover={{ y: -3 }}
      transition={{ type: 'spring', stiffness: 320, damping: 22 }}
      className="card-surface relative overflow-hidden rounded-2xl p-5"
    >
      <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${ACCENTS[accent]} opacity-60`} />
      <div className="relative">
        <div className={`mb-3 inline-flex h-10 w-10 items-center justify-center rounded-xl border bg-ink-900/70`}>
          <Icon className="h-5 w-5" strokeWidth={1.8} />
        </div>
        <p className="font-mono text-2xl font-semibold tracking-tight text-slate-50">{value}</p>
        <p className="mt-1 text-sm font-medium text-slate-300">{label}</p>
        {detail && <p className="mt-0.5 text-xs text-slate-500">{detail}</p>}
      </div>
    </motion.div>
  )
}
