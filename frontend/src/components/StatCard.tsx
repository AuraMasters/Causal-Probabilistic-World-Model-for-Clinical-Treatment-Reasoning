import { motion } from 'framer-motion'
import type { LucideIcon } from 'lucide-react'

interface StatCardProps {
  icon: LucideIcon
  label: string
  value: string
  detail?: string
}

export function StatCard({ icon: Icon, label, value, detail }: StatCardProps) {
  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ type: 'spring', stiffness: 320, damping: 22 }}
      className="card-surface relative overflow-hidden rounded-xl p-5"
    >
      <div className="relative">
        <div className="mb-3 inline-flex h-10 w-10 items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900 text-white">
          <Icon className="h-5 w-5" strokeWidth={1.8} />
        </div>
        <p className="font-mono text-2xl font-bold tracking-tight text-white">{value}</p>
        <p className="mt-1 text-sm font-medium text-zinc-300">{label}</p>
        {detail && <p className="mt-0.5 text-xs text-zinc-500 font-mono">{detail}</p>}
      </div>
    </motion.div>
  )
}
