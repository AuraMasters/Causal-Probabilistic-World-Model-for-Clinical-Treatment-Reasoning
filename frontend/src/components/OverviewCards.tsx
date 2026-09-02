import { motion } from 'framer-motion'
import { BrainCircuit, Database, GitBranch, HeartPulse } from 'lucide-react'
import type { Overview } from '../lib/types'

export function OverviewCards({ overview }: { overview: Overview }) {
  const stats = [
    {
      icon: Database,
      label: 'Patients',
      value: overview.dataset.patients.toLocaleString(),
      detail: `${overview.dataset.development_rows} development · ${overview.dataset.test_rows} held-out test`,
      accent: 'cyan' as const,
    },
    {
      icon: HeartPulse,
      label: 'Treatments',
      value: String(overview.treatments.length),
      detail: 'ACTG175 interventions',
      accent: 'mint' as const,
    },
    {
      icon: GitBranch,
      label: 'Bayesian Network',
      value: `${overview.model.dag_edges} edges`,
      detail: `${overview.model.nodes} nodes · final DAG`,
      accent: 'violet' as const,
    },
    {
      icon: BrainCircuit,
      label: 'Parameter learning',
      value: overview.model.parameter_learning.split(' ')[0],
      detail: overview.model.parameter_learning,
      accent: 'violet' as const,
    },
  ]

  return (
    <div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <OverviewCard key={stat.label} {...stat} />
        ))}
      </div>
    </div>
  )
}

function OverviewCard({
  icon: Icon,
  label,
  value,
  detail,
  accent,
}: {
  icon: typeof Database
  label: string
  value: string
  detail: string
  accent: 'cyan' | 'mint' | 'violet'
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="card-surface relative overflow-hidden rounded-2xl p-5"
    >
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-white/[0.03] to-transparent" />
      <div className="relative">
        <div
          className={`mb-3 inline-flex h-10 w-10 items-center justify-center rounded-xl border ${
            accent === 'cyan'
              ? 'border-cyan-400/25 bg-cyan-400/10 text-cyan-300'
              : accent === 'mint'
                ? 'border-mint-400/25 bg-mint-400/10 text-mint-300'
                : 'border-violet-400/25 bg-violet-400/10 text-violet-300'
          }`}
        >
          <Icon className="h-5 w-5" strokeWidth={1.8} />
        </div>
        <p className="font-display text-2xl font-semibold tracking-tight text-slate-50">{value}</p>
        <p className="mt-1 text-sm font-medium text-slate-300">{label}</p>
        <p className="mt-0.5 text-xs text-slate-500">{detail}</p>
      </div>
    </motion.div>
  )
}
