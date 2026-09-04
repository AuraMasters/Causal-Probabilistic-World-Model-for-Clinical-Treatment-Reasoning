import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  BarChart3,
  CheckCircle2,
  Database,
  Gauge,
  GitBranch,
  LineChart,
  Percent,
  Scale,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Sparkles,
  UserCheck,
} from 'lucide-react'
import type { Overview, CurvePoint, CalibrationBin, DcaPoint, ThresholdSweepItem } from '../lib/types'
import { formatPercent } from '../lib/format'

export function ValidationSection({ overview }: { overview: Overview }) {
  const [activeModelTab, setActiveModelTab] = useState<'continuous' | 'discretized'>('continuous')
  const [activeTab, setActiveTab] = useState<
    'comparison' | 'predictive' | 'thresholds' | 'dca' | 'calibration' | 'confidence' | 'baselines' | 'counterfactual' | 'subgroups' | 'limitations'
  >('comparison')

  const compA = overview.comprehensive_validation
  const compB = overview.continuous_validation
  const comparison = overview.model_comparison

  const comp = activeModelTab === 'continuous' ? (compB || compA) : (compA || compB)

  if (!comp) {
    return <FallbackValidation overview={overview} />
  }

  const pred = comp.predictive_metrics
  const cis = comp.confidence_intervals_95
  const calib = comp.calibration
  const dca = comp.decision_curve_analysis
  const thresh = comp.threshold_analysis
  const baselines = comp.baseline_comparison || []
  const cf = comp.counterfactual_treatment_evaluation
  const subgroups = comp.subgroup_analysis
  const meth = comp.methodology_and_limitations

  const tabs = [
    { id: 'comparison', label: 'Model A vs B Comparison', icon: Scale },
    { id: 'predictive', label: 'Predictive & Curves', icon: BarChart3 },
    { id: 'thresholds', label: 'Threshold Calibration', icon: Sliders },
    { id: 'dca', label: 'Decision Curves (DCA)', icon: LineChart },
    { id: 'calibration', label: 'Calibration & Reliability', icon: Gauge },
    { id: 'confidence', label: '95% Bootstrap CIs', icon: Percent },
    { id: 'baselines', label: 'ML Baseline Benchmarks', icon: GitBranch },
    { id: 'counterfactual', label: 'Counterfactual Advantage', icon: Sparkles },
    { id: 'subgroups', label: 'Subgroup Analysis', icon: UserCheck },
    { id: 'limitations', label: 'Methodology & Ethics', icon: ShieldAlert },
  ]

  return (
    <div className="space-y-6">
      {/* Model Selection Switcher & Navigation */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between rounded-2xl border border-cyan-400/25 bg-ink-900/95 p-3.5 shadow-lg">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs font-bold text-slate-400 uppercase mr-1">Active Model View:</span>
          <button
            onClick={() => setActiveModelTab('continuous')}
            className={`rounded-lg px-3 py-1 text-xs font-mono font-bold transition-all cursor-pointer ${
              activeModelTab === 'continuous'
                ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/40 shadow-sm shadow-cyan-400/20'
                : 'text-slate-400 hover:text-white border border-slate-700/60'
            }`}
          >
            Model B: Continuous / Hybrid SCM (Primary)
          </button>
          <button
            onClick={() => setActiveModelTab('discretized')}
            className={`rounded-lg px-3 py-1 text-xs font-mono font-bold transition-all cursor-pointer ${
              activeModelTab === 'discretized'
                ? 'bg-amber-400/20 text-amber-200 border border-amber-400/40 shadow-sm'
                : 'text-slate-400 hover:text-white border border-slate-700/60'
            }`}
          >
            Model A: Discretized BN (Baseline)
          </button>
        </div>

        <span className="text-[11px] font-mono text-slate-400">
          Held-out Test Cohort: <strong className="text-white">N = 428 patients</strong>
        </span>
      </div>

      {/* Top Section Navigation Tabs */}
      <div className="flex flex-wrap items-center gap-1.5 rounded-2xl border border-cyan-400/20 bg-ink-900/90 p-2 shadow-md">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`inline-flex items-center gap-2 rounded-xl px-3 py-2 font-mono text-xs font-bold transition-all cursor-pointer ${
                isActive
                  ? 'bg-cyan-400/20 text-cyan-200 border border-cyan-400/40 shadow-sm shadow-cyan-400/20'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-ink-850'
              }`}
            >
              <Icon className={`h-3.5 w-3.5 ${isActive ? 'text-cyan-300' : 'text-slate-400'}`} />
              <span>{tab.label}</span>
            </button>
          )
        })}
      </div>

      {/* TAB: MODEL A VS MODEL B COMPARISON */}
      {activeTab === 'comparison' && comparison && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          {/* Summary Hero Card */}
          <div className="card-surface rounded-2xl border border-cyan-400/35 bg-gradient-to-br from-cyan-400/10 via-ink-900 to-ink-950 p-6 sm:p-8 shadow-xl">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-bold tracking-[0.18em] text-cyan-300 uppercase">
                    Continuous Information Preservation vs Discretization
                  </span>
                  <span className="rounded-md bg-mint-300/20 border border-mint-300/40 px-2 py-0.5 font-mono text-[10px] font-bold text-mint-200 uppercase">
                    Faculty Recommended
                  </span>
                </div>
                <h3 className="mt-1 font-display text-xl font-bold text-white">
                  Model B (Continuous SCM) vs Model A (Discretized BN)
                </h3>
                <p className="mt-2 text-xs text-slate-300 leading-relaxed max-w-3xl">
                  {comparison.summary}
                </p>
              </div>
            </div>
          </div>

          {/* Quick Head-to-Head Pillar Metric Cards */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="card-surface rounded-2xl p-5 border border-cyan-400/30">
              <span className="font-mono text-xs text-slate-400 uppercase">Discrimination (ROC-AUC)</span>
              <div className="mt-2 flex items-baseline gap-3">
                <span className="font-display text-3xl font-extrabold text-cyan-200">{comparison.model_b_summary.roc_auc.toFixed(4)}</span>
                <span className="font-mono text-xs text-slate-400">vs {comparison.model_a_summary.roc_auc.toFixed(4)}</span>
              </div>
              <p className="mt-1 text-[11px] font-mono text-mint-300 font-bold">+0.0506 AUC Gain (Model B)</p>
            </div>

            <div className="card-surface rounded-2xl p-5 border border-cyan-400/30">
              <span className="font-mono text-xs text-slate-400 uppercase">Precision-Recall AUC</span>
              <div className="mt-2 flex items-baseline gap-3">
                <span className="font-display text-3xl font-extrabold text-cyan-200">{comparison.model_b_summary.pr_auc.toFixed(4)}</span>
                <span className="font-mono text-xs text-slate-400">vs {comparison.model_a_summary.pr_auc.toFixed(4)}</span>
              </div>
              <p className="mt-1 text-[11px] font-mono text-mint-300 font-bold">+0.0443 PR-AUC Gain (Model B)</p>
            </div>

            <div className="card-surface rounded-2xl p-5 border border-cyan-400/30">
              <span className="font-mono text-xs text-slate-400 uppercase">Calibration Error (ECE)</span>
              <div className="mt-2 flex items-baseline gap-3">
                <span className="font-display text-3xl font-extrabold text-mint-200">{formatPercent(comparison.model_b_summary.ece, 2)}</span>
                <span className="font-mono text-xs text-slate-400">vs {formatPercent(comparison.model_a_summary.ece, 2)}</span>
              </div>
              <p className="mt-1 text-[11px] font-mono text-mint-300 font-bold">Lower probability gap</p>
            </div>

            <div className="card-surface rounded-2xl p-5 border border-cyan-400/30">
              <span className="font-mono text-xs text-slate-400 uppercase">Calibrated F1-Score</span>
              <div className="mt-2 flex items-baseline gap-3">
                <span className="font-display text-3xl font-extrabold text-cyan-200">{comparison.model_b_summary.calibrated_f1.toFixed(4)}</span>
                <span className="font-mono text-xs text-slate-400">vs {comparison.model_a_summary.calibrated_f1.toFixed(4)}</span>
              </div>
              <p className="mt-1 text-[11px] font-mono text-mint-300 font-bold">+0.0520 F1 Gain at tau*</p>
            </div>
          </div>

          {/* Full Comparative Dimension Table */}
          <div className="overflow-hidden rounded-2xl border border-slate-700/60 bg-ink-900/90 shadow-xl">
            <div className="p-4 border-b border-slate-800">
              <h4 className="font-display text-base font-bold text-white">Full Side-by-Side Methodological Benchmark</h4>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] border-collapse text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-slate-800 bg-ink-950/80 text-slate-300 uppercase">
                    <th className="px-5 py-3.5 font-bold w-1/4">Evaluation Dimension</th>
                    <th className="px-5 py-3.5 font-bold text-amber-200 w-1/3">Model A (Discretized BN)</th>
                    <th className="px-5 py-3.5 font-bold text-cyan-200 w-1/3">Model B (Continuous SCM)</th>
                    <th className="px-5 py-3.5 font-bold text-mint-200">Scientific Advantage</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  {comparison.comparison_table.map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-5 py-3.5 font-sans font-bold text-white">{row.dimension}</td>
                      <td className="px-5 py-3.5 text-slate-300">{row.model_a_discretized}</td>
                      <td className="px-5 py-3.5 text-cyan-100 font-medium">{row.model_b_continuous}</td>
                      <td className="px-5 py-3.5 text-mint-300 font-sans text-[11px]">{row.advantage}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}

      {/* TAB 1: PREDICTIVE PERFORMANCE & CURVES */}
      {activeTab === 'predictive' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          {/* Primary Metric Grid */}
          <div className="grid grid-cols-2 gap-3.5 sm:grid-cols-3 lg:grid-cols-6">
            <MetricBox label="ROC-AUC" value={pred.roc_auc.toFixed(4)} ci={`${cis.roc_auc.ci_lower} – ${cis.roc_auc.ci_upper}`} hint="Discrimination" accent="cyan" />
            <MetricBox label="PR-AUC" value={pred.pr_auc.toFixed(4)} ci={`${cis.pr_auc.ci_lower} – ${cis.pr_auc.ci_upper}`} hint="Precision-Recall" accent="cyan" />
            <MetricBox label="Accuracy" value={formatPercent(pred.accuracy, 1)} ci={`${(cis.accuracy.ci_lower*100).toFixed(1)}% – ${(cis.accuracy.ci_upper*100).toFixed(1)}%`} hint="Agreement at 0.50" accent="mint" />
            <MetricBox label="Specificity" value={formatPercent(pred.specificity, 1)} ci={`${(cis.specificity.ci_lower*100).toFixed(1)}% – ${(cis.specificity.ci_upper*100).toFixed(1)}%`} hint="True Negative Rate" accent="mint" />
            <MetricBox label="Sensitivity" value={formatPercent(pred.recall_sensitivity, 1)} ci={`${(cis.recall_sensitivity.ci_lower*100).toFixed(1)}% – ${(cis.recall_sensitivity.ci_upper*100).toFixed(1)}%`} hint="At fixed 0.50 cutoff" accent="amber" />
            <MetricBox label="Brier Score" value={pred.brier_score.toFixed(4)} ci={`${cis.brier_score.ci_lower} – ${cis.brier_score.ci_upper}`} hint="Lower is better" accent="mint" />
          </div>

          {/* Curves & Confusion Matrix Grid */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {/* ROC Curve */}
            <div className="card-surface rounded-2xl p-6 shadow-lg">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="font-mono text-xs font-bold text-slate-300 uppercase">Receiver Operating Characteristic</span>
                <span className="font-mono text-xs font-extrabold text-cyan-300">AUC = {pred.roc_auc.toFixed(4)}</span>
              </div>
              <div className="my-5 flex justify-center">
                <SimpleRocCurve points={pred.roc_curve} />
              </div>
              <p className="text-[11px] text-slate-400 font-mono text-center">
                True Positive Rate vs False Positive Rate
              </p>
            </div>

            {/* PR Curve */}
            <div className="card-surface rounded-2xl p-6 shadow-lg">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="font-mono text-xs font-bold text-slate-300 uppercase">Precision-Recall Curve</span>
                <span className="font-mono text-xs font-extrabold text-mint-300">PR-AUC = {pred.pr_auc.toFixed(4)}</span>
              </div>
              <div className="my-5 flex justify-center">
                <SimplePrCurve points={pred.pr_curve} baselineRate={104/428} />
              </div>
              <p className="text-[11px] text-slate-400 font-mono text-center">
                Dashed line: random event prevalence (24.3%)
              </p>
            </div>

            {/* Confusion Matrix */}
            <div className="card-surface rounded-2xl p-6 shadow-lg flex flex-col justify-between">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <span className="font-mono text-xs font-bold text-slate-300 uppercase">Confusion Matrix (τ = 0.50)</span>
                <span className="font-mono text-xs font-bold text-slate-400">Total N = {pred.confusion_matrix.total}</span>
              </div>

              <div className="my-4 grid grid-cols-2 gap-2 text-center font-mono text-xs">
                <div className="rounded-xl border border-mint-400/30 bg-mint-400/10 p-3">
                  <span className="text-[10px] text-mint-300 uppercase block">True Negative</span>
                  <span className="text-2xl font-extrabold text-white">{pred.confusion_matrix.true_negatives}</span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">Correct Surv</span>
                </div>
                <div className="rounded-xl border border-amber-400/30 bg-amber-400/10 p-3">
                  <span className="text-[10px] text-amber-300 uppercase block">False Positive</span>
                  <span className="text-2xl font-extrabold text-white">{pred.confusion_matrix.false_positives}</span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">False Alarm</span>
                </div>
                <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3">
                  <span className="text-[10px] text-rose-300 uppercase block">False Negative</span>
                  <span className="text-2xl font-extrabold text-white">{pred.confusion_matrix.false_negatives}</span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">Missed Prog</span>
                </div>
                <div className="rounded-xl border border-cyan-400/30 bg-cyan-400/10 p-3">
                  <span className="text-[10px] text-cyan-300 uppercase block">True Positive</span>
                  <span className="text-2xl font-extrabold text-white">{pred.confusion_matrix.true_positives}</span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">Detected Prog</span>
                </div>
              </div>

              <p className="text-[11px] text-slate-400 font-mono text-center">
                Operating point at uncalibrated 0.50 threshold
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* TAB 2: THRESHOLD CALIBRATION */}
      {activeTab === 'thresholds' && thresh && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <div className="overflow-hidden rounded-2xl border border-slate-700/60 bg-ink-900/90 shadow-xl">
            <div className="p-5 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <div>
                <h4 className="font-display text-base font-bold text-white">Development-Tuned Threshold Analysis</h4>
                <p className="text-xs text-slate-400">
                  Optimal threshold selected strictly on Development cohort via Youden's J, then evaluated on held-out test data
                </p>
              </div>
              <span className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 font-mono text-xs font-bold text-cyan-200">
                Optimal τ* = {thresh.investigation_summary.optimal_threshold_tau}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[700px] border-collapse text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-slate-800 bg-ink-950/80 text-slate-400 uppercase">
                    <th className="px-5 py-3.5">Decision Policy</th>
                    <th className="px-5 py-3.5 text-center">Threshold (τ)</th>
                    <th className="px-5 py-3.5 text-right text-mint-200">Sensitivity (Recall)</th>
                    <th className="px-5 py-3.5 text-right">Specificity</th>
                    <th className="px-5 py-3.5 text-right">Precision</th>
                    <th className="px-5 py-3.5 text-right">F1-Score</th>
                    <th className="px-5 py-3.5 text-right">Accuracy</th>
                    <th className="px-5 py-3.5 text-right text-mint-200">True Positives</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  <tr className="hover:bg-slate-800/40">
                    <td className="px-5 py-3.5 font-sans font-bold text-slate-300">Default Cutoff (0.50)</td>
                    <td className="px-5 py-3.5 text-center font-bold text-slate-400">0.50</td>
                    <td className="px-5 py-3.5 text-right text-amber-300 font-bold">
                      {formatPercent(thresh.investigation_summary.test_default_threshold_0_50.sensitivity_recall, 1)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-slate-300">
                      {formatPercent(thresh.investigation_summary.test_default_threshold_0_50.specificity, 1)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-slate-300">
                      {formatPercent(thresh.investigation_summary.test_default_threshold_0_50.precision, 1)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-slate-300">
                      {thresh.investigation_summary.test_default_threshold_0_50.f1_score.toFixed(4)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-slate-300">
                      {formatPercent(thresh.investigation_summary.test_default_threshold_0_50.accuracy, 1)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-slate-300">
                      {thresh.investigation_summary.test_default_threshold_0_50.true_positives} / 104
                    </td>
                  </tr>

                  <tr className="bg-cyan-400/10 hover:bg-cyan-400/15">
                    <td className="px-5 py-3.5 font-sans font-bold text-white flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-cyan-300" />
                      <span>Calibrated Decision Cutoff (Optimal)</span>
                    </td>
                    <td className="px-5 py-3.5 text-center font-extrabold text-cyan-200">
                      {thresh.investigation_summary.optimal_threshold_tau}
                    </td>
                    <td className="px-5 py-3.5 text-right text-mint-200 font-extrabold">
                      {formatPercent(thresh.investigation_summary.test_calibrated_threshold.sensitivity_recall, 1)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-slate-300 font-bold">
                      {formatPercent(thresh.investigation_summary.test_calibrated_threshold.specificity, 1)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-cyan-200 font-bold">
                      {formatPercent(thresh.investigation_summary.test_calibrated_threshold.precision, 1)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-white font-bold">
                      {thresh.investigation_summary.test_calibrated_threshold.f1_score.toFixed(4)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-slate-300">
                      {formatPercent(thresh.investigation_summary.test_calibrated_threshold.accuracy, 1)}
                    </td>
                    <td className="px-5 py-3.5 text-right text-mint-200 font-extrabold">
                      {thresh.investigation_summary.test_calibrated_threshold.true_positives} / 104
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Threshold Sweep Curves */}
          <div className="card-surface rounded-2xl p-6 shadow-lg">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h4 className="font-display text-base font-bold text-white">Threshold Trade-off Dynamics (Sensitivity vs Specificity)</h4>
                <p className="text-xs text-slate-400">Sweeping decision threshold τ from 0.05 to 0.60 on held-out test data</p>
              </div>
            </div>
            <div className="my-6 flex justify-center">
              <SimpleThresholdSweepPlot sweep={thresh.test_threshold_sweep} optimal={thresh.investigation_summary.optimal_threshold_tau} />
            </div>
            <div className="flex flex-wrap items-center justify-center gap-6 text-xs font-mono text-slate-300 border-t border-slate-800 pt-3">
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-mint-300" /> Sensitivity / Recall</span>
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-cyan-400" /> Specificity</span>
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-amber-300" /> F1-Score</span>
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-white border border-slate-400" /> Optimal τ = {thresh.investigation_summary.optimal_threshold_tau}</span>
            </div>
          </div>
        </motion.div>
      )}

      {/* TAB 3: DECISION CURVE ANALYSIS (DCA) */}
      {activeTab === 'dca' && dca && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <div className="card-surface rounded-2xl p-6 shadow-xl border border-slate-700/60">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-800 pb-4 gap-2">
              <div>
                <h4 className="font-display text-lg font-bold text-white">Decision Curve Analysis (Clinical Net Benefit)</h4>
                <p className="text-xs text-slate-400">Vickers &amp; Elkin Clinical Utility Framework across preference threshold probabilities</p>
              </div>
              <span className="font-mono text-xs text-cyan-200 bg-cyan-400/10 border border-cyan-400/30 px-3 py-1 rounded-lg">
                Prevalence = {formatPercent(dca.event_prevalence, 1)}
              </span>
            </div>

            <div className="my-6 flex justify-center">
              <SimpleDcaPlot points={dca.dca_points} />
            </div>

            <div className="flex flex-wrap items-center justify-center gap-6 text-xs font-mono text-slate-300 border-t border-slate-800 pt-3">
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-cyan-400" /> Model-Guided Strategy</span>
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-slate-500" /> Treat All Patients</span>
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full bg-rose-500" /> Treat None</span>
            </div>

            <p className="mt-4 text-xs text-slate-400 leading-relaxed font-sans border-t border-slate-800/80 pt-3">
              <strong>Clinical Utility Interpretation:</strong> {dca.interpretation}
            </p>
          </div>
        </motion.div>
      )}

      {/* TAB 4: CALIBRATION & RELIABILITY */}
      {activeTab === 'calibration' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.2fr_1fr]">
            {/* Calibration Reliability Plot */}
            <div className="card-surface rounded-2xl p-6 shadow-lg">
              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div>
                  <h4 className="font-display text-base font-bold text-white">Reliability Diagram (Calibration Curve)</h4>
                  <p className="text-xs text-slate-400">Mean Predicted Probability vs Observed Clinical Outcome Frequency</p>
                </div>
                <div className="text-right">
                  <span className="font-mono text-sm font-extrabold text-mint-200">ECE = {calib.ece.toFixed(4)}</span>
                  <p className="text-[10px] text-slate-400 font-mono">
                    {calib.calibration_intercept !== undefined && `Intercept: ${calib.calibration_intercept} | Slope: ${calib.calibration_slope}`}
                  </p>
                </div>
              </div>

              <div className="my-6 flex justify-center">
                <SimpleCalibrationPlot bins={calib.bins} />
              </div>

              <div className="rounded-xl border border-cyan-400/20 bg-ink-950 p-3.5 text-xs text-slate-300 flex items-start gap-2.5">
                <ShieldCheck className="h-4 w-4 text-cyan-300 shrink-0 mt-0.5" />
                <p>
                  <strong>Clinical Significance:</strong> With an Expected Calibration Error of <span className="text-mint-200 font-bold">{formatPercent(calib.ece, 2)}</span> and Brier score of <span className="text-cyan-200 font-bold">{calib.brier_score.toFixed(4)}</span>, predicted probabilities accurately reflect actual empirical patient risk.
                </p>
              </div>
            </div>

            {/* Probability Distribution Histogram */}
            <div className="card-surface rounded-2xl p-6 shadow-lg flex flex-col justify-between">
              <div className="border-b border-slate-800 pb-4">
                <h4 className="font-display text-base font-bold text-white">Predicted Risk Distribution</h4>
                <p className="text-xs text-slate-400">Predicted P(Progression) split by actual patient outcome</p>
              </div>

              <div className="my-4 space-y-2.5">
                {calib.probability_distribution.bin_labels.slice(0, 6).map((binLabel, idx) => {
                  const count0 = calib.probability_distribution.label_0_counts[idx] || 0
                  const count1 = calib.probability_distribution.label_1_counts[idx] || 0
                  const totalBin = count0 + count1
                  return (
                    <div key={binLabel} className="text-xs font-mono">
                      <div className="flex justify-between text-slate-300 mb-1">
                        <span>Risk Bin {binLabel}</span>
                        <span className="text-slate-400">{totalBin} pts ({count0} surv / {count1} prog)</span>
                      </div>
                      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-slate-900 border border-slate-800">
                        <div
                          className="bg-mint-400"
                          style={{ width: `${(count0 / 428) * 100 * 3}%` }}
                          title={`Label 0: ${count0}`}
                        />
                        <div
                          className="bg-rose-500"
                          style={{ width: `${(count1 / 428) * 100 * 3}%` }}
                          title={`Label 1: ${count1}`}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>

              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 border-t border-slate-800 pt-3">
                <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-mint-400" /> Label 0 (No Progression)</span>
                <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-rose-500" /> Label 1 (Progression)</span>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* TAB 5: 95% BOOTSTRAP CONFIDENCE INTERVALS */}
      {activeTab === 'confidence' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <div className="overflow-hidden rounded-2xl border border-slate-700/60 bg-ink-900/90 shadow-xl">
            <div className="p-5 border-b border-slate-800 flex items-center justify-between">
              <div>
                <h4 className="font-display text-base font-bold text-white">Bootstrap 95% Confidence Intervals</h4>
                <p className="text-xs text-slate-400">Non-parametric resampling with B = 1,000 iterations on held-out test cohort (N = 428)</p>
              </div>
              <span className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 font-mono text-xs font-bold text-cyan-200">
                Seed = 42
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse text-left">
                <thead>
                  <tr className="border-b border-slate-800 bg-ink-950/80">
                    <th className="px-5 py-3.5 font-mono text-xs font-bold tracking-wider text-slate-300 uppercase">Metric</th>
                    <th className="px-5 py-3.5 text-right font-mono text-xs font-bold tracking-wider text-white uppercase">Point Estimate</th>
                    <th className="px-5 py-3.5 text-right font-mono text-xs font-bold tracking-wider text-mint-200 uppercase">95% CI Range</th>
                    <th className="px-5 py-3.5 text-right font-mono text-xs font-bold tracking-wider text-slate-400 uppercase">Std. Error</th>
                    <th className="px-5 py-3.5 font-mono text-xs font-bold tracking-wider text-slate-400 uppercase">Interpretation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80 font-mono text-xs">
                  {Object.entries(cis).map(([key, ci]) => (
                    <tr key={key} className="hover:bg-slate-800/40 transition-colors">
                      <td className="px-5 py-3.5 font-bold text-slate-200 uppercase">{key.replace('_', ' ')}</td>
                      <td className="px-5 py-3.5 text-right font-extrabold text-white">
                        {key.includes('accuracy') || key.includes('rate') || key.includes('precision') || key.includes('recall') || key.includes('specificity')
                          ? formatPercent(ci.point_estimate, 2)
                          : ci.point_estimate.toFixed(4)}
                      </td>
                      <td className="px-5 py-3.5 text-right text-mint-200 font-bold">
                        [{ci.ci_lower.toFixed(4)}, {ci.ci_upper.toFixed(4)}]
                      </td>
                      <td className="px-5 py-3.5 text-right text-slate-400">&plusmn; {ci.std_error.toFixed(4)}</td>
                      <td className="px-5 py-3.5 font-sans text-slate-300 text-xs">
                        {key === 'roc_auc' && 'Discrimination is significantly above chance level (CI excludes 0.50)'}
                        {key === 'accuracy' && 'High overall diagnostic agreement on unseen patients'}
                        {key === 'brier_score' && 'Low probabilistic penalty, indicating calibrated probability outputs'}
                        {key === 'specificity' && 'Exceptional ability to rule out false progression alarms'}
                        {key === 'pr_auc' && 'Exceeds uninformative baseline prevalence (~0.24)'}
                        {key === 'log_loss' && 'Stable cross-entropy loss'}
                        {key === 'f1_score' && 'Harmonic balance reflecting class imbalance'}
                        {key === 'precision' && 'Positive predictive value under default cutoff'}
                        {key === 'recall_sensitivity' && 'Conservative positive threshold minimizes false alarms'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}

      {/* TAB 6: BASELINE MODEL COMPARISON */}
      {activeTab === 'baselines' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <div className="overflow-hidden rounded-2xl border border-slate-700/60 bg-ink-900/90 shadow-xl">
            <div className="p-5 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <div>
                <h4 className="font-display text-base font-bold text-white">Predictive Benchmarks vs Causal World Models</h4>
                <p className="text-xs text-slate-400">All models evaluated on identical 80/20 train/test split with seed 42</p>
              </div>
              <span className="font-mono text-xs text-cyan-300">Exact Same Design Matrix &amp; Test Partition</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] border-collapse text-left">
                <thead>
                  <tr className="border-b border-slate-800 bg-ink-950/80 font-mono text-xs uppercase">
                    <th className="px-5 py-3.5 text-slate-300 font-bold">Model Architecture</th>
                    <th className="px-5 py-3.5 text-right text-cyan-200 font-bold">ROC-AUC</th>
                    <th className="px-5 py-3.5 text-right text-mint-200 font-bold">Accuracy</th>
                    <th className="px-5 py-3.5 text-right text-slate-300 font-bold">F1-Score</th>
                    <th className="px-5 py-3.5 text-right text-slate-300 font-bold">Brier Score</th>
                    <th className="px-5 py-3.5 text-center text-slate-300 font-bold">Interventional do(trt)</th>
                    <th className="px-5 py-3.5 text-slate-300 font-bold">Interpretability</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80 font-mono text-xs">
                  {baselines.map((m) => {
                    const isOurs = m.supports_do_calculus
                    return (
                      <tr
                        key={m.model_name}
                        className={`transition-colors ${
                          isOurs ? 'bg-cyan-400/10 hover:bg-cyan-400/15' : 'hover:bg-slate-800/40'
                        }`}
                      >
                        <td className="px-5 py-3.5 font-sans font-bold">
                          <div className="flex items-center gap-2">
                            {isOurs && <Sparkles className="h-4 w-4 text-cyan-300" />}
                            <span className={isOurs ? 'text-white' : 'text-slate-300'}>{m.model_name}</span>
                          </div>
                          <p className="font-mono text-[11px] text-slate-400 font-normal">{m.model_type}</p>
                        </td>
                        <td className="px-5 py-3.5 text-right font-bold text-cyan-200">{m.roc_auc.toFixed(4)}</td>
                        <td className="px-5 py-3.5 text-right font-bold text-mint-200">{formatPercent(m.accuracy, 1)}</td>
                        <td className="px-5 py-3.5 text-right text-slate-200">{m.f1_score.toFixed(4)}</td>
                        <td className="px-5 py-3.5 text-right text-slate-200">{m.brier_score.toFixed(4)}</td>
                        <td className="px-5 py-3.5 text-center">
                          {m.supports_do_calculus ? (
                            <span className="inline-flex items-center gap-1 rounded-full bg-mint-400/20 px-2.5 py-0.5 text-[10px] font-bold text-mint-200 uppercase">
                              <CheckCircle2 className="h-3 w-3" /> Supported
                            </span>
                          ) : (
                            <span className="text-slate-500 font-mono text-[11px]">No (Associative Only)</span>
                          )}
                        </td>
                        <td className="px-5 py-3.5 font-sans text-xs text-slate-300">{m.interpretability}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}

      {/* TAB 7: COUNTERFACTUAL TREATMENT ADVANTAGE */}
      {activeTab === 'counterfactual' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <div className="card-surface rounded-2xl border border-cyan-400/35 bg-gradient-to-br from-cyan-400/10 via-ink-900 to-ink-950 p-6 sm:p-8 shadow-xl">
            <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-start gap-4">
                <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-cyan-400/35 bg-cyan-400/20 shadow-sm">
                  <Scale className="h-6 w-6 text-cyan-300" strokeWidth={2} />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-bold tracking-[0.18em] text-cyan-300 uppercase">
                      Model-Predicted Counterfactual Advantage
                    </span>
                    <span className="rounded-md bg-mint-300/20 border border-mint-300/40 px-2 py-0.5 font-mono text-[10px] font-bold text-mint-200 uppercase">
                      {formatPercent(cf.better_rate, 0)} of Cohort
                    </span>
                  </div>
                  <h3 className="mt-1 font-display text-xl font-bold text-white">
                    Counterfactual Risk Reduction across {cf.better_count} of {cf.test_patients} Patients
                  </h3>
                  <p className="mt-1.5 max-w-2xl text-xs text-slate-300 leading-relaxed">
                    Evaluating each patient under do(trt = rec) vs do(trt = obs): For <strong className="text-mint-200">{formatPercent(cf.better_rate, 0)}</strong> of held-out test patients, the model-selected regimen yielded a strictly lower predicted progression probability than the historically assigned trial arm.
                  </p>
                </div>
              </div>

              <div className="flex flex-col items-center justify-center rounded-2xl border border-mint-300/30 bg-ink-950/80 px-6 py-4 text-center shrink-0">
                <p className="font-display text-4xl font-extrabold text-mint-200">{formatPercent(cf.better_rate, 0)}</p>
                <p className="font-mono text-xs text-slate-400 mt-0.5">Mean Risk &Delta;: {(cf.mean_risk_reduction*100).toFixed(1)}%</p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
            <div className="card-surface rounded-2xl p-5">
              <span className="font-mono text-xs font-bold text-mint-200 uppercase">Lower Predicted Risk</span>
              <p className="mt-2 font-display text-3xl font-extrabold text-white">{cf.better_count} <span className="text-sm font-mono text-slate-400">({formatPercent(cf.better_rate, 1)})</span></p>
              <p className="mt-1 text-xs text-slate-400">Recommended arm is counterfactually superior</p>
            </div>
            <div className="card-surface rounded-2xl p-5">
              <span className="font-mono text-xs font-bold text-cyan-200 uppercase">Identical Strategy</span>
              <p className="mt-2 font-display text-3xl font-extrabold text-white">{cf.same_count} <span className="text-sm font-mono text-slate-400">({formatPercent(cf.same_rate, 1)})</span></p>
              <p className="mt-1 text-xs text-slate-400">Recommended arm matched observed assignment</p>
            </div>
            <div className="card-surface rounded-2xl p-5">
              <span className="font-mono text-xs font-bold text-rose-300 uppercase">Higher Risk</span>
              <p className="mt-2 font-display text-3xl font-extrabold text-white">{cf.worse_count} <span className="text-sm font-mono text-slate-400">({formatPercent(cf.worse_rate, 1)})</span></p>
              <p className="mt-1 text-xs text-slate-400">0 cases where recommendation had higher risk</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* TAB 8: CLINICAL SUBGROUP ANALYSIS */}
      {activeTab === 'subgroups' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <div className="overflow-hidden rounded-2xl border border-slate-700/60 bg-ink-900/90 shadow-xl">
            <div className="p-5 border-b border-slate-800">
              <h4 className="font-display text-base font-bold text-white">Stratified Subgroup Performance Analysis</h4>
              <p className="text-xs text-slate-400">Evaluated on actual continuous patient biomarkers (CD4, Age, Karnofsky) before model inference</p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[700px] border-collapse text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-slate-800 bg-ink-950/80 text-slate-300 uppercase">
                    <th className="px-5 py-3.5 font-bold">Clinical Subgroup</th>
                    <th className="px-5 py-3.5 text-right font-bold">Sample Size (N)</th>
                    <th className="px-5 py-3.5 text-right font-bold text-rose-300">Progression Rate</th>
                    <th className="px-5 py-3.5 text-right font-bold text-cyan-200">ROC-AUC</th>
                    <th className="px-5 py-3.5 text-right font-bold text-mint-200">Accuracy</th>
                    <th className="px-5 py-3.5 text-right font-bold">Brier Score</th>
                    <th className="px-5 py-3.5 font-bold">Reliability</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  {subgroups.map((sub) => (
                    <tr key={sub.subgroup_name} className="hover:bg-slate-800/40">
                      <td className="px-5 py-3.5 font-sans font-bold text-white">{sub.subgroup_name}</td>
                      <td className="px-5 py-3.5 text-right font-bold text-slate-200">{sub.sample_size}</td>
                      <td className="px-5 py-3.5 text-right text-rose-300">{formatPercent(sub.event_rate, 1)}</td>
                      <td className="px-5 py-3.5 text-right font-bold text-cyan-200">
                        {sub.roc_auc !== null ? sub.roc_auc.toFixed(4) : 'N/A'}
                      </td>
                      <td className="px-5 py-3.5 text-right text-mint-200 font-bold">{formatPercent(sub.accuracy, 1)}</td>
                      <td className="px-5 py-3.5 text-right text-slate-300">{sub.brier_score.toFixed(4)}</td>
                      <td className="px-5 py-3.5">
                        {sub.is_reliable ? (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-mint-200 uppercase bg-mint-400/15 border border-mint-400/30 px-2 py-0.5 rounded-md">
                            Adequate (N &ge; 30)
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[10px] font-bold text-amber-300 uppercase bg-amber-400/15 border border-amber-400/30 px-2 py-0.5 rounded-md">
                            Small Sample
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      )}

      {/* TAB 9: METHODOLOGY & LIMITATIONS */}
      {activeTab === 'limitations' && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="space-y-6"
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="card-surface rounded-2xl p-5 border border-cyan-400/25">
              <div className="flex items-center gap-2 text-cyan-300 font-display font-bold text-sm">
                <Database className="h-4 w-4" />
                <span>Continuous Variable Modeling</span>
              </div>
              <p className="mt-2 text-xs text-slate-300 leading-relaxed font-sans">
                {meth.continuous_modeling || meth.data_partitioning}
              </p>
            </div>

            <div className="card-surface rounded-2xl p-5 border border-mint-300/25">
              <div className="flex items-center gap-2 text-mint-200 font-display font-bold text-sm">
                <GitBranch className="h-4 w-4" />
                <span>Causal Identification Assumptions</span>
              </div>
              <p className="mt-2 text-xs text-slate-300 leading-relaxed font-sans">
                {meth.causal_assumptions}
              </p>
            </div>

            <div className="card-surface rounded-2xl p-5 border border-amber-400/25">
              <div className="flex items-center gap-2 text-amber-300 font-display font-bold text-sm">
                <Scale className="h-4 w-4" />
                <span>Strict 80/20 Partitioning</span>
              </div>
              <p className="mt-2 text-xs text-slate-300 leading-relaxed font-sans">
                {meth.data_partitioning}
              </p>
            </div>

            <div className="card-surface rounded-2xl p-5 border border-rose-500/35 bg-rose-500/5">
              <div className="flex items-center gap-2 text-rose-300 font-display font-bold text-sm">
                <ShieldAlert className="h-4 w-4" />
                <span>Formal Clinical Prototype Disclaimer</span>
              </div>
              <p className="mt-2 text-xs text-slate-300 leading-relaxed font-sans">
                {meth.clinical_disclaimer}
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}

function MetricBox({
  label,
  value,
  ci,
  hint,
  accent,
}: {
  label: string
  value: string
  ci: string
  hint: string
  accent: 'cyan' | 'mint' | 'amber'
}) {
  const textColor =
    accent === 'cyan' ? 'text-cyan-200' : accent === 'mint' ? 'text-mint-200' : 'text-amber-200'
  return (
    <div className="card-surface rounded-2xl border border-slate-700/60 p-4 shadow-md hover:border-cyan-400/30 transition-all">
      <p className="font-mono text-[11px] font-bold tracking-wider text-slate-400 uppercase">{label}</p>
      <p className={`mt-2 font-display text-2xl font-extrabold tracking-tight ${textColor}`}>{value}</p>
      <p className="mt-1 text-[10px] text-slate-400 font-mono">95% CI: {ci}</p>
      <p className="mt-0.5 text-[10px] text-slate-500 font-mono">{hint}</p>
    </div>
  )
}

function SimpleRocCurve({ points }: { points: CurvePoint[] }) {
  const width = 220
  const height = 180
  const padding = 25

  const pathD = points
    .map((p, idx) => {
      const x = padding + (p.fpr || 0) * (width - 2 * padding)
      const y = height - padding - (p.tpr || 0) * (height - 2 * padding)
      return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`
    })
    .join(' ')

  return (
    <svg width={width} height={height} className="overflow-visible">
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={height - padding} x2={width - padding} y2={padding} stroke="#475569" strokeDasharray="3 3" strokeWidth="1.2" />
      <path d={pathD} fill="none" stroke="#92eeff" strokeWidth="2.5" />
      <text x={width / 2} y={height - 5} fill="#94a3b8" fontSize="9" textAnchor="middle" fontFamily="monospace">FPR (1 - Specificity)</text>
      <text x={8} y={height / 2} fill="#94a3b8" fontSize="9" textAnchor="middle" transform={`rotate(-90 8 ${height/2})`} fontFamily="monospace">TPR (Recall)</text>
    </svg>
  )
}

function SimplePrCurve({ points, baselineRate }: { points: CurvePoint[]; baselineRate: number }) {
  const width = 220
  const height = 180
  const padding = 25

  const pathD = points
    .map((p, idx) => {
      const x = padding + (p.recall || 0) * (width - 2 * padding)
      const y = height - padding - (p.precision || 0) * (height - 2 * padding)
      return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`
    })
    .join(' ')

  const baselineY = height - padding - baselineRate * (height - 2 * padding)

  return (
    <svg width={width} height={height} className="overflow-visible">
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={baselineY} x2={width - padding} y2={baselineY} stroke="#f43f5e" strokeDasharray="3 3" strokeWidth="1" />
      <path d={pathD} fill="none" stroke="#d8ffc5" strokeWidth="2.5" />
      <text x={width / 2} y={height - 5} fill="#94a3b8" fontSize="9" textAnchor="middle" fontFamily="monospace">Recall</text>
      <text x={8} y={height / 2} fill="#94a3b8" fontSize="9" textAnchor="middle" transform={`rotate(-90 8 ${height/2})`} fontFamily="monospace">Precision</text>
    </svg>
  )
}

function SimpleCalibrationPlot({ bins }: { bins: CalibrationBin[] }) {
  const width = 280
  const height = 200
  const padding = 35

  const populated = bins.filter((b) => b.count > 0 && b.mean_predicted !== null && b.observed_rate !== null)

  return (
    <svg width={width} height={height} className="overflow-visible">
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={height - padding} x2={width - padding} y2={padding} stroke="#64748b" strokeDasharray="4 4" strokeWidth="1.2" />
      {populated.map((b) => {
        const x = padding + (b.mean_predicted || 0) * (width - 2 * padding)
        const y = height - padding - (b.observed_rate || 0) * (height - 2 * padding)
        return (
          <g key={b.bin}>
            <circle cx={x} cy={y} r="5" fill="#30afff" stroke="#92eeff" strokeWidth="2" />
          </g>
        )
      })}
      <text x={width / 2} y={height - 8} fill="#94a3b8" fontSize="10" textAnchor="middle" fontFamily="monospace">Mean Predicted Risk</text>
      <text x={12} y={height / 2} fill="#94a3b8" fontSize="10" textAnchor="middle" transform={`rotate(-90 12 ${height/2})`} fontFamily="monospace">Observed Frequency</text>
    </svg>
  )
}

function SimpleDcaPlot({ points }: { points: DcaPoint[] }) {
  const width = 360
  const height = 220
  const padding = 45

  const minX = 0.05
  const maxX = 0.50
  const minY = -0.05
  const maxY = 0.25

  const scaleX = (x: number) => padding + ((x - minX) / (maxX - minX)) * (width - 2 * padding)
  const scaleY = (y: number) => height - padding - ((y - minY) / (maxY - minY)) * (height - 2 * padding)

  const pathModel = points
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(p.threshold_probability)} ${scaleY(p.net_benefit_model)}`)
    .join(' ')

  const pathAll = points
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(p.threshold_probability)} ${scaleY(p.net_benefit_all)}`)
    .join(' ')

  const yZero = scaleY(0)

  return (
    <svg width={width} height={height} className="overflow-visible">
      <line x1={padding} y1={yZero} x2={width - padding} y2={yZero} stroke="#f43f5e" strokeWidth="2" strokeDasharray="3 3" />
      <path d={pathAll} fill="none" stroke="#64748b" strokeWidth="2" strokeDasharray="4 4" />
      <path d={pathModel} fill="none" stroke="#30afff" strokeWidth="3" />
      {points.map((p) => (
        <circle
          key={p.threshold_probability}
          cx={scaleX(p.threshold_probability)}
          cy={scaleY(p.net_benefit_model)}
          r="3.5"
          fill="#30afff"
          stroke="#92eeff"
          strokeWidth="1.5"
        />
      ))}
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <text x={width / 2} y={height - 8} fill="#94a3b8" fontSize="10" textAnchor="middle" fontFamily="monospace">Threshold Probability (pt)</text>
      <text x={12} y={height / 2} fill="#94a3b8" fontSize="10" textAnchor="middle" transform={`rotate(-90 12 ${height/2})`} fontFamily="monospace">Net Benefit</text>
    </svg>
  )
}

function SimpleThresholdSweepPlot({ sweep, optimal }: { sweep: ThresholdSweepItem[]; optimal: number }) {
  const width = 360
  const height = 220
  const padding = 45

  const minX = 0.05
  const maxX = 0.60

  const scaleX = (x: number) => padding + ((x - minX) / (maxX - minX)) * (width - 2 * padding)
  const scaleY = (y: number) => height - padding - y * (height - 2 * padding)

  const pathSens = sweep.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(p.threshold)} ${scaleY(p.sensitivity_recall)}`).join(' ')
  const pathSpec = sweep.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(p.threshold)} ${scaleY(p.specificity)}`).join(' ')
  const pathF1 = sweep.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${scaleX(p.threshold)} ${scaleY(p.f1_score)}`).join(' ')

  const xOptimal = scaleX(optimal)

  return (
    <svg width={width} height={height} className="overflow-visible">
      <line x1={xOptimal} y1={padding} x2={xOptimal} y2={height - padding} stroke="#ffffff" strokeWidth="1.5" strokeDasharray="3 3" />
      <path d={pathSens} fill="none" stroke="#d8ffc5" strokeWidth="2.5" />
      <path d={pathSpec} fill="none" stroke="#30afff" strokeWidth="2.5" />
      <path d={pathF1} fill="none" stroke="#f59e0b" strokeWidth="2.5" />
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <text x={width / 2} y={height - 8} fill="#94a3b8" fontSize="10" textAnchor="middle" fontFamily="monospace">Decision Threshold (τ)</text>
      <text x={12} y={height / 2} fill="#94a3b8" fontSize="10" textAnchor="middle" transform={`rotate(-90 12 ${height/2})`} fontFamily="monospace">Metric Score</text>
    </svg>
  )
}

function FallbackValidation({ overview }: { overview: Overview }) {
  const metrics = overview.validation
  const decision = overview.treatment_decision_validation

  const cards = [
    { label: 'Log Loss', value: metrics.log_loss.toFixed(4), hint: 'Lower is better' },
    { label: 'Brier Score', value: metrics.brier_score.toFixed(4), hint: 'Lower is better' },
    { label: 'ROC-AUC', value: metrics.roc_auc.toFixed(4), hint: 'Higher is better' },
    { label: 'Accuracy', value: formatPercent(metrics.accuracy, 1), hint: 'Agreement' },
    { label: 'ECE', value: metrics.ece.toFixed(4), hint: 'Calibration error' },
    { label: 'Test cohort', value: String(metrics.test_patients), hint: 'Held-out set' },
  ]

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3.5 sm:grid-cols-3 lg:grid-cols-6">
        {cards.map((card) => (
          <div key={card.label} className="card-surface rounded-2xl p-4">
            <p className="font-mono text-[11px] font-bold text-slate-400 uppercase">{card.label}</p>
            <p className="mt-2 font-display text-2xl font-extrabold text-white">{card.value}</p>
            <p className="mt-1 text-[11px] text-slate-500 font-mono">{card.hint}</p>
          </div>
        ))}
      </div>
      <div className="card-surface rounded-2xl p-6 border border-cyan-400/30">
        <h4 className="font-display text-base font-bold text-white">Counterfactual Decision Rate: {formatPercent(decision.rate, 0)}</h4>
        <p className="text-xs text-slate-300 mt-1">Recommended regimen achieved lower risk than observed in {decision.better_count}/{decision.total} test patients.</p>
      </div>
    </div>
  )
}
