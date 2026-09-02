import { motion } from 'framer-motion'
import { BrainCircuit, Database, GitBranch, HeartPulse } from 'lucide-react'
import type { Overview } from '../lib/types'

export function OverviewCards({ overview }: { overview: Overview }) {
  const stats = [
    {
      icon: Database,
      label: 'Trial Cohort',
      value: overview.dataset.patients.toLocaleString(),
      unit: 'Patients',
      detail: `${overview.dataset.development_rows} development · ${overview.dataset.test_rows} held-out test`,
      accent: 'cyan' as const,
    },
    {
      icon: HeartPulse,
      label: 'Treatment Arms',
      value: String(overview.treatments.length),
      unit: 'Interventions',
      detail: 'ZDV, ZDV+ddI, ZDV+ddC, ddI',
      accent: 'mint' as const,
    },
    {
      icon: GitBranch,
      label: 'Bayesian DAG',
      value: `${overview.model.dag_edges}`,
      unit: 'Causal Edges',
      detail: `${overview.model.nodes} clinical nodes · temporal structure`,
      accent: 'cyan' as const,
    },
    {
      icon: BrainCircuit,
      label: 'CPT Estimation',
      value: 'BDeu',
      unit: 'ESS = 10',
      detail: 'Exact Variable Elimination inference',
      accent: 'mint' as const,
    },
  ]

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat, index) => (
        <OverviewCard key={stat.label} index={index} {...stat} />
      ))}
    </div>
  )
}

function OverviewCard({
  index,
  icon: Icon,
  label,
  value,
  unit,
  detail,
  accent,
}: {
  index: number
  icon: typeof Database
  label: string
  value: string
  unit: string
  detail: string
  accent: 'cyan' | 'mint'
}) {
  const isMint = accent === 'mint'
  const iconBorder = isMint ? 'border-mint-300/35 bg-mint-300/10 text-mint-200' : 'border-cyan-400/35 bg-cyan-400/10 text-cyan-300'
  const valueColor = isMint ? 'text-mint-200' : 'text-cyan-200'

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.06, duration: 0.45, ease: 'easeOut' }}
      className="card-surface relative overflow-hidden rounded-2xl p-5 shadow-lg hover:border-cyan-400/30 transition-all"
    >
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs font-bold tracking-wider text-slate-400 uppercase">
          {label}
        </span>
        <div className={`inline-flex h-9 w-9 items-center justify-center rounded-xl border ${iconBorder}`}>
          <Icon className="h-4.5 w-4.5" strokeWidth={2} />
        </div>
      </div>

      <div className="mt-3 flex items-baseline gap-2">
        <p className={`font-display text-3xl font-extrabold tracking-tight ${valueColor}`}>
          {value}
        </p>
        <span className="font-mono text-xs text-slate-400 font-semibold">{unit}</span>
      </div>

      <p className="mt-1.5 font-mono text-xs text-slate-400 leading-snug">{detail}</p>
    </motion.div>
  )
}
