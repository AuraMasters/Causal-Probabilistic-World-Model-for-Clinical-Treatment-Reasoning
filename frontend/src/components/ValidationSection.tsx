import { motion } from 'framer-motion'
import {
  BarChart3,
  Gauge,
  Scale,
  ShieldCheck,
} from 'lucide-react'
import type { Overview } from '../lib/types'
import { formatPercent } from '../lib/format'

export function ValidationSection({ overview }: { overview: Overview }) {
  const metrics = overview.validation
  const decision = overview.treatment_decision_validation

  const cards = [
    { label: 'Log Loss', value: metrics.log_loss.toFixed(4), hint: 'Lower is better', target: '0.5284', isGood: true },
    { label: 'Brier Score', value: metrics.brier_score.toFixed(4), hint: 'Lower is better', target: '0.1770', isGood: true },
    { label: 'ROC-AUC', value: metrics.roc_auc.toFixed(4), hint: 'Higher is better', target: '0.6558', isGood: true },
    { label: 'Accuracy', value: formatPercent(metrics.accuracy, 1), hint: 'Prediction agreement', target: '76.4%', isGood: true },
    { label: 'Calibration (ECE)', value: metrics.ece.toFixed(4), hint: 'Calibration error', target: '0.0460', isGood: true },
    { label: 'Held-Out Test Set', value: String(metrics.test_patients), hint: 'Strict out-of-sample', target: '428', isGood: true },
  ]

  return (
    <div className="space-y-6">
      {/* 6 Key Model Performance Metrics */}
      <div className="grid grid-cols-2 gap-3.5 sm:grid-cols-3 lg:grid-cols-6">
        {cards.map((card, index) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.05 }}
            className="card-surface rounded-2xl border border-slate-700/60 p-4 shadow-md hover:border-cyan-400/30 transition-all"
          >
            <p className="font-mono text-[11px] font-bold tracking-wider text-slate-400 uppercase">{card.label}</p>
            <p className="mt-2 font-display text-2xl font-extrabold tracking-tight text-white">{card.value}</p>
            <p className="mt-1 text-[11px] text-cyan-200/70 font-mono">{card.hint}</p>
          </motion.div>
        ))}
      </div>

      {/* Decision Validation Hero Card */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="card-surface relative overflow-hidden rounded-2xl border border-cyan-400/35 bg-gradient-to-br from-cyan-400/10 via-ink-900 to-ink-950 p-6 sm:p-8 shadow-xl"
      >
        <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-cyan-400/35 bg-cyan-400/20 shadow-sm shadow-cyan-400/20">
              <Scale className="h-6 w-6 text-cyan-300" strokeWidth={2} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs font-bold tracking-[0.18em] text-cyan-300 uppercase">
                  Counterfactual Treatment Decision Validation
                </span>
                <span className="rounded-md bg-mint-300/20 border border-mint-300/40 px-2 py-0.5 font-mono text-[10px] font-bold text-mint-200 uppercase">
                  75% Success Rate
                </span>
              </div>
              <h3 className="mt-1 font-display text-xl font-bold text-white">
                Causal Superiority in 321 of 428 Held-Out Test Patients
              </h3>
              <p className="mt-1.5 max-w-2xl text-xs text-slate-300 leading-relaxed">
                For <strong className="text-mint-200">{decision.better_count} of {decision.total}</strong> test patients ({formatPercent(decision.rate, 0)}), the model's recommended regimen demonstrated a strictly lower predicted progression probability P(label = 1) than the historically assigned treatment arm.
              </p>
            </div>
          </div>

          <div className="flex flex-col items-center justify-center rounded-2xl border border-mint-300/30 bg-ink-950/80 px-6 py-4 text-center shrink-0">
            <p className="font-display text-4xl font-extrabold tracking-tight text-mint-200">{formatPercent(decision.rate, 0)}</p>
            <p className="font-mono text-xs text-slate-400 mt-0.5">Test cohort agreement</p>
          </div>
        </div>
      </motion.div>

      {/* 3 Research Methodology Cards */}
      <div className="grid grid-cols-1 gap-3.5 sm:grid-cols-3">
        <Callout
          icon={ShieldCheck}
          title="Strict Out-Of-Sample Separation"
          text="CPTs learned strictly on the 1,711 development patients with BDeu prior (ESS = 10). Test cohort remained completely unobserved during training."
        />
        <Callout
          icon={Gauge}
          title="Decision-Theoretic Evaluation"
          text="Evaluates whether model-recommended regimens achieve superior counterfactual utility compared to historical monotherapy assignments."
        />
        <Callout
          icon={BarChart3}
          title="Exact Pearlian Inference"
          text="Interventional predictions computed through exact Bayesian Variable Elimination without sampling variance or MCMC approximations."
        />
      </div>
    </div>
  )
}

function Callout({
  icon: Icon,
  title,
  text,
}: {
  icon: typeof ShieldCheck
  title: string
  text: string
}) {
  return (
    <div className="card-surface rounded-2xl border border-slate-700/60 p-4.5 hover:border-slate-600 transition-colors">
      <div className="flex items-center gap-2.5">
        <Icon className="h-4 w-4 text-cyan-300 shrink-0" strokeWidth={2.2} />
        <p className="font-display text-xs font-bold text-white uppercase tracking-wider">{title}</p>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">{text}</p>
    </div>
  )
}
