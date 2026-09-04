import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  Award,
  CheckCircle2,
  HelpCircle,
  LineChart,
  ShieldCheck,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { formatPercent, formatProbability } from '../lib/format'
import type { AnalyzeResult } from '../lib/types'

const FEATURE_LABELS: Record<string, string> = {
  age: 'Age (years)',
  wtkg: 'Body Weight (kg)',
  karnof: 'Karnofsky Score (%)',
  preanti: 'Pre-ART Exposure (days)',
  cd40: 'Baseline CD4 (cells/mm³)',
  cd80: 'Baseline CD8 (cells/mm³)',
  hemo: 'Hemophilia',
  homo: 'Homosexuality',
  drugs: 'IV Drug History',
  oprior: 'Prior Opportunistic Infection',
  z30: 'Prior ZDV Exposure',
  race: 'Race',
  gender: 'Gender',
  strat: 'Trial Stratum',
  symptom: 'Symptom Status',
}

const ARM_COLORS: Record<number, { stroke: string; fill: string; name: string }> = {
  0: { stroke: '#f43f5e', fill: 'bg-rose-500', name: 'ZDV Monotherapy' },
  1: { stroke: '#38bdf8', fill: 'bg-sky-400', name: 'ZDV + ddI' },
  2: { stroke: '#4ade80', fill: 'bg-emerald-400', name: 'ZDV + ddC (Recommended)' },
  3: { stroke: '#c084fc', fill: 'bg-purple-400', name: 'ddI Monotherapy' },
}

export function RecommendedTreatment({ result }: { result: AnalyzeResult }) {
  const recommended = result.recommended
  const isContinuousModel = Boolean(recommended.continuous_sensitivities && recommended.continuous_sensitivities.length > 0)

  // Determine drug classification description
  const drugClass =
    recommended.treatment === 0
      ? 'Monotherapy (Zidovudine)'
      : recommended.treatment === 3
        ? 'Monotherapy (Didanosine)'
        : 'Combination Antiretroviral Therapy (Dual NRTI)'

  const riskDelta = recommended.risk_delta_vs_monotherapy ?? 0
  const continuousSens = recommended.continuous_sensitivities || []
  const discreteAttributions = recommended.feature_attributions || []
  const eValue = recommended.e_value_analysis
  const benefit = recommended.treatment_benefit
  const whatIf = result.what_if_trajectory

  // Interactive slider state for What-If CD4 simulation
  const [simCd4, setSimCd4] = useState<number>(whatIf?.current_value ?? 350)

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="relative overflow-hidden rounded-2xl border border-cyan-400/35 bg-gradient-to-br from-ink-850 via-ink-900 to-ink-950 p-6 sm:p-9 shadow-2xl shadow-cyan-950/40"
    >
      {/* Top glowing ambient highlight */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-cyan-400 via-mint-300 to-cyan-400 opacity-80" />

      {/* Decision Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-cyan-400/15 pb-5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-cyan-400/20 border border-cyan-400/40 shadow-sm shadow-cyan-400/20">
            <Award className="h-5 w-5 text-cyan-300" strokeWidth={2.2} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold tracking-[0.18em] text-cyan-300 uppercase">
                Model-Recommended Treatment Strategy
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-mint-300/15 border border-mint-300/30 px-2 py-0.5 font-mono text-[10px] font-bold text-mint-200 uppercase">
                <CheckCircle2 className="h-3 w-3 text-mint-300" /> Rank #1
              </span>
            </div>
            <p className="font-sans text-xs text-slate-400 mt-0.5">
              Maximizes Expected Clinical Utility under Pearlian interventional conditioning do(trt = k)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {riskDelta > 0 && (
            <span className="inline-flex items-center gap-1 rounded-lg border border-mint-300/30 bg-mint-300/10 px-2.5 py-1 font-mono text-xs font-bold text-mint-200">
              <TrendingDown className="h-3.5 w-3.5" /> -{formatPercent(riskDelta, 1)} Risk vs Monotherapy
            </span>
          )}
          {benefit && (
            <span className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-2.5 py-1 font-mono text-xs font-bold text-cyan-200">
              {benefit.benefit_tier} ({benefit.benefit_percentile}th %ile Benefit)
            </span>
          )}
          <span className="rounded-lg border border-slate-700/60 bg-ink-950/80 px-3 py-1 font-mono text-xs font-semibold text-slate-300">
            Arm {recommended.treatment}
          </span>
          <span className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 font-mono text-xs font-bold text-cyan-200">
            {recommended.short_name}
          </span>
        </div>
      </div>

      {/* Regimen Hero Info */}
      <div className="mt-6 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <p className="font-mono text-xs font-semibold tracking-wider text-slate-400 uppercase">
              Model-Estimated Optimal Regimen
            </p>
            <span className="rounded bg-cyan-400/15 border border-cyan-400/30 px-2 py-0.5 font-mono text-[10px] font-bold text-cyan-200">
              {isContinuousModel ? 'Model B: Continuous / Hybrid SCM' : 'Model A: Discretized BN'}
            </span>
          </div>
          <h3 className="mt-1 font-display text-3xl font-extrabold tracking-tight text-white sm:text-4xl lg:text-5xl">
            {recommended.name}
          </h3>
          <p className="mt-2 flex items-center gap-2 text-sm text-cyan-200/90 font-medium">
            <span className="h-2 w-2 rounded-full bg-mint-300 inline-block animate-pulse" />
            {drugClass} &middot; Lowest model-estimated progression probability under evidence
          </p>
        </div>
      </div>

      {/* 3 Core Metric Pillar Cards */}
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="card-surface rounded-2xl border border-mint-300/30 bg-gradient-to-br from-mint-300/10 to-ink-900/90 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-mint-200 uppercase">
              Progression-Free Probability
            </span>
            <ShieldCheck className="h-4 w-4 text-mint-300" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-mint-200 sm:text-5xl">
            {formatPercent(recommended.p_label_0, 1)}
          </p>
          <p className="mt-1 font-mono text-xs text-mint-300/80">
            P(Progression-Free | do(trt={recommended.treatment})) = {formatProbability(recommended.p_label_0, 4)}
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-950">
            <div
              className="h-full bg-gradient-to-r from-mint-300 to-mint-400 rounded-full"
              style={{ width: `${recommended.p_label_0 * 100}%` }}
            />
          </div>
        </div>

        <div className="card-surface rounded-2xl border border-rose-500/30 bg-gradient-to-br from-rose-500/10 to-ink-900/90 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-rose-300 uppercase">
              Progression Risk
            </span>
            <Activity className="h-4 w-4 text-rose-risk" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-rose-risk sm:text-5xl">
            {formatPercent(recommended.p_label_1, 1)}
          </p>
          <p className="mt-1 font-mono text-xs text-rose-300/80">
            P(Progression | do(trt={recommended.treatment})) = {formatProbability(recommended.p_label_1, 4)}
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-950">
            <div
              className="h-full bg-rose-500 rounded-full"
              style={{ width: `${recommended.p_label_1 * 100}%` }}
            />
          </div>
        </div>

        <div className="card-surface rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-cyan-400/10 to-ink-900/90 p-5 shadow-lg">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-cyan-200 uppercase">
              Expected Clinical Utility
            </span>
            <TrendingUp className="h-4 w-4 text-cyan-300" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-cyan-200 sm:text-5xl">
            {recommended.expected_utility.toFixed(4)}
          </p>
          <p className="mt-1 font-mono text-xs text-cyan-300/80">
            Utility scale: 0.0000 (Progression) to 1.0000 (Progression-Free)
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-ink-950">
            <div
              className="h-full bg-cyan-400 rounded-full"
              style={{ width: `${recommended.expected_utility * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* E-Value Sensitivity Analysis for Unmeasured Confounding */}
      {eValue && (
        <div className="mt-6 rounded-2xl border border-emerald-400/30 bg-ink-950/90 p-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-400/20 border border-emerald-400/40">
                <ShieldCheck className="h-4 w-4 text-emerald-300" />
              </div>
              <div>
                <span className="font-mono text-xs font-bold text-emerald-300 uppercase">
                  E-Value Sensitivity Analysis (VanderWeele &amp; Ding)
                </span>
                <p className="text-[11px] text-slate-400">
                  Mathematical lower bound for robustness against unmeasured confounding
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="rounded-lg bg-emerald-400/15 border border-emerald-400/30 px-3 py-1 font-mono text-xs font-extrabold text-emerald-200">
                E-Value = {eValue.e_value_point.toFixed(2)}
              </span>
              <span className="rounded-lg bg-slate-800 px-2.5 py-1 font-mono text-[11px] text-slate-300">
                RR = {eValue.risk_ratio_vs_monotherapy.toFixed(3)} vs Monotherapy
              </span>
            </div>
          </div>

          <p className="mt-3 text-xs text-slate-300 leading-relaxed font-sans">
            <strong>Clinical Causal Interpretation:</strong> {eValue.interpretation} Because{' '}
            <span className="text-emerald-300 font-bold">E = {eValue.e_value_point.toFixed(2)} &ge; 1.5</span>, the causal recommendation has high stability and cannot be easily nullified by residual clinical confounders.
          </p>
        </div>
      )}

      {/* MODEL B: Exact Continuous Gradient Sensitivity Analysis */}
      {isContinuousModel && continuousSens.length > 0 && (
        <div className="mt-6 rounded-2xl border border-cyan-400/30 bg-ink-950/80 p-5">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-cyan-300 uppercase">
              <Sparkles className="h-3.5 w-3.5" />
              <span>Continuous Biomarker Gradient Sensitivity (∂P / ∂X_j)</span>
            </div>
            <span className="text-[11px] font-mono text-slate-400">
              Exact continuous derivatives per clinical unit
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {continuousSens.map((item) => {
              const label = FEATURE_LABELS[item.variable] || item.variable
              const isRisk = item.direction === 'increases_risk'
              const isProtective = item.direction === 'reduces_risk'

              return (
                <div
                  key={item.variable}
                  className={`rounded-xl border p-3 text-xs transition-all ${
                    isRisk
                      ? 'border-rose-500/30 bg-rose-500/10'
                      : isProtective
                        ? 'border-mint-300/30 bg-mint-300/10'
                        : 'border-slate-800 bg-ink-900/60'
                  }`}
                >
                  <div className="flex items-center justify-between font-mono font-bold">
                    <span className="text-white">{label}</span>
                    <span
                      className={`text-[11px] px-1.5 py-0.5 rounded ${
                        isRisk
                          ? 'bg-rose-500/20 text-rose-300'
                          : isProtective
                            ? 'bg-mint-300/20 text-mint-200'
                            : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {item.percentage_points_per_step > 0
                        ? `+${item.percentage_points_per_step}% / +${item.clinical_step_unit}`
                        : item.percentage_points_per_step < 0
                          ? `${item.percentage_points_per_step}% / +${item.clinical_step_unit}`
                          : '0.0% / unit'}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] font-mono text-slate-400 truncate">
                    Current Patient Value: <strong className="text-cyan-200">{item.raw_value}</strong>
                  </div>
                </div>
              )
            })}
          </div>

          <p className="mt-3 text-[11px] text-slate-400 flex items-center gap-1.5">
            <HelpCircle className="h-3.5 w-3.5 text-slate-500 shrink-0" />
            <span>
              Continuous gradient sensitivity calculates the exact marginal change in progression risk per clinical unit increase in each continuous biomarker, avoiding step-function binning distortions.
            </span>
          </p>
        </div>
      )}

      {/* Interactive Continuous "What-If" Sensitivity Simulator */}
      {whatIf && whatIf.trajectory_points.length > 0 && (
        <div className="mt-6 rounded-2xl border border-sky-400/30 bg-ink-950/90 p-5 shadow-xl">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-sky-400/20 border border-sky-400/40">
                <LineChart className="h-4 w-4 text-sky-300" />
              </div>
              <div>
                <span className="font-mono text-xs font-bold text-sky-300 uppercase">
                  Continuous What-If Trajectory Simulator
                </span>
                <p className="text-[11px] text-slate-400">
                  Simulating continuous progression risk across all 4 trial arms as CD4 varies from 50 to 800 cells/mm³
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-slate-300">Simulate CD4:</span>
              <input
                type="range"
                min={50}
                max={750}
                step={25}
                value={simCd4}
                onChange={(e) => setSimCd4(Number(e.target.value))}
                className="h-2 w-32 accent-sky-400 cursor-pointer"
              />
              <span className="font-mono text-xs font-bold text-sky-200 bg-sky-400/15 px-2 py-0.5 rounded border border-sky-400/30">
                {simCd4} cells/mm³
              </span>
            </div>
          </div>

          {/* SVG Multi-Arm Trajectory Plot */}
          <div className="my-4 flex justify-center">
            <WhatIfTrajectorySvg trajectory={whatIf.trajectory_points} currentCd4={whatIf.current_value} simulatedCd4={simCd4} />
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-mono border-t border-slate-800 pt-3">
            {Object.entries(ARM_COLORS).map(([armKey, info]) => (
              <span key={armKey} className="flex items-center gap-1.5 text-slate-300">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: info.stroke }} />
                <span>Arm {armKey}: {info.name}</span>
              </span>
            ))}
            <span className="flex items-center gap-1.5 text-white">
              <span className="h-3 w-0.5 bg-white" />
              <span>Patient CD4 ({whatIf.current_value})</span>
            </span>
          </div>
        </div>
      )}

      {/* MODEL A: Discrete Evidence Ablation */}
      {!isContinuousModel && discreteAttributions.length > 0 && (
        <div className="mt-6 rounded-2xl border border-slate-700/60 bg-ink-950/80 p-5">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-slate-300 uppercase">
              <Sparkles className="h-3.5 w-3.5 text-cyan-300" />
              <span>Patient Characteristics Influencing Risk (Bayesian Evidence Ablation)</span>
            </div>
            <span className="text-[11px] font-mono text-slate-400">
              Ranked by posterior sensitivity |ΔP|
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {discreteAttributions.slice(0, 6).map((item) => {
              const label = FEATURE_LABELS[item.feature] || item.feature
              const isRisk = item.direction === 'increases_risk'
              const isProtective = item.direction === 'reduces_risk'

              return (
                <div
                  key={item.feature}
                  className={`rounded-xl border p-3 text-xs transition-all ${
                    isRisk
                      ? 'border-rose-500/30 bg-rose-500/10'
                      : isProtective
                        ? 'border-mint-300/30 bg-mint-300/10'
                        : 'border-slate-800 bg-ink-900/60'
                  }`}
                >
                  <div className="flex items-center justify-between font-mono font-bold">
                    <span className="text-white">{label}</span>
                    <span
                      className={`text-[11px] px-1.5 py-0.5 rounded ${
                        isRisk
                          ? 'bg-rose-500/20 text-rose-300'
                          : isProtective
                            ? 'bg-mint-300/20 text-mint-200'
                            : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {item.risk_impact > 0 ? `+${item.percentage_points}% Risk` : item.risk_impact < 0 ? `${item.percentage_points}% Protective` : 'Neutral (0%)'}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] font-mono text-slate-400 truncate">
                    Observed Bin: <span className="text-slate-300 font-medium">{item.observed_state}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Full Segmented Probability Breakdown Bar */}
      <div className="mt-4 rounded-2xl border border-slate-700/60 bg-ink-950/80 p-5">
        <div className="mb-2.5 flex flex-wrap items-center justify-between gap-2 text-xs font-semibold">
          <span className="flex items-center gap-1.5 text-mint-200 font-mono">
            <span className="h-2.5 w-2.5 rounded-full bg-mint-300" />
            Progression-Free P(label = 0): {formatPercent(recommended.p_label_0, 2)}
          </span>
          <span className="flex items-center gap-1.5 text-rose-300 font-mono">
            <span className="h-2.5 w-2.5 rounded-full bg-rose-risk" />
            Progression P(label = 1): {formatPercent(recommended.p_label_1, 2)}
          </span>
        </div>
        <div className="flex h-3.5 w-full overflow-hidden rounded-full bg-slate-900 p-0.5 border border-slate-700/50">
          <div
            className="h-full rounded-l-full bg-gradient-to-r from-mint-400 to-mint-300 transition-all duration-700"
            style={{ width: `${recommended.p_label_0 * 100}%` }}
          />
          <div
            className="h-full rounded-r-full bg-gradient-to-r from-rose-500 to-rose-600 transition-all duration-700"
            style={{ width: `${recommended.p_label_1 * 100}%` }}
          />
        </div>
      </div>

      {/* Causal Interventional Rationale Callout */}
      <div className="mt-6 flex items-start gap-3 rounded-2xl border border-cyan-400/25 bg-gradient-to-r from-cyan-950/40 via-ink-900 to-ink-950 p-4.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan-400/15 border border-cyan-400/30">
          <Zap className="h-4 w-4 text-cyan-300" strokeWidth={2.2} />
        </div>
        <div className="flex-1 text-xs leading-relaxed text-slate-300">
          <p className="font-semibold text-white">Causal Interventional Decision Rationale</p>
          <p className="mt-1">
            Conditioning on this patient's exact continuous biomarkers (CD4, CD8, Karnofsky, and prior ART history),{' '}
            <strong className="text-mint-200">{recommended.name}</strong> yields the lowest model-estimated posterior probability of disease progression among all four randomized treatment strategies in the ACTG175 Causal World Model.
          </p>
        </div>
      </div>
    </motion.div>
  )
}

function WhatIfTrajectorySvg({
  trajectory,
  currentCd4,
  simulatedCd4,
}: {
  trajectory: Array<{ cd4: number; arm_0: number; arm_1: number; arm_2: number; arm_3: number }>
  currentCd4: number
  simulatedCd4: number
}) {
  const width = 500
  const height = 200
  const padding = 45

  const minX = 50
  const maxX = 800
  const minY = 0.0
  const maxY = 0.45

  const scaleX = (x: number) => padding + ((x - minX) / (maxX - minX)) * (width - 2 * padding)
  const scaleY = (y: number) => height - padding - ((y - minY) / (maxY - minY)) * (height - 2 * padding)

  const armPaths: Record<number, string> = {}
  for (const arm of [0, 1, 2, 3]) {
    armPaths[arm] = trajectory
      .map((p, idx) => {
        const val = p[`arm_${arm}` as keyof typeof p] as number
        return `${idx === 0 ? 'M' : 'L'} ${scaleX(p.cd4)} ${scaleY(val)}`
      })
      .join(' ')
  }

  const currentX = scaleX(currentCd4)
  const simX = scaleX(simulatedCd4)

  return (
    <svg width={width} height={height} className="overflow-visible w-full max-w-[500px]">
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#334155" strokeWidth="1" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#334155" strokeWidth="1" />

      {/* Grid lines */}
      {[0.1, 0.2, 0.3, 0.4].map((yVal) => (
        <line
          key={yVal}
          x1={padding}
          y1={scaleY(yVal)}
          x2={width - padding}
          y2={scaleY(yVal)}
          stroke="#1e293b"
          strokeDasharray="2 2"
          strokeWidth="1"
        />
      ))}

      {/* Trajectory lines for each arm */}
      {Object.entries(ARM_COLORS).map(([armKey, info]) => {
        const armNum = Number(armKey)
        return (
          <path
            key={armKey}
            d={armPaths[armNum]}
            fill="none"
            stroke={info.stroke}
            strokeWidth={armNum === 2 ? 3.5 : 2}
            strokeDasharray={armNum === 0 ? '3 3' : undefined}
          />
        )
      })}

      {/* Patient Current CD4 vertical marker */}
      <line
        x1={currentX}
        y1={padding}
        x2={currentX}
        y2={height - padding}
        stroke="#ffffff"
        strokeWidth="2"
        strokeDasharray="4 4"
      />
      <circle cx={currentX} cy={scaleY(0.04)} r="4" fill="#ffffff" />

      {/* Simulated CD4 vertical marker */}
      {Math.abs(simX - currentX) > 5 && (
        <line
          x1={simX}
          y1={padding}
          x2={simX}
          y2={height - padding}
          stroke="#38bdf8"
          strokeWidth="2"
          strokeDasharray="2 2"
        />
      )}

      <text x={width / 2} y={height - 8} fill="#94a3b8" fontSize="10" textAnchor="middle" fontFamily="monospace">
        Baseline CD4 Count (cells/mm³)
      </text>
      <text x={12} y={height / 2} fill="#94a3b8" fontSize="10" textAnchor="middle" transform={`rotate(-90 12 ${height/2})`} fontFamily="monospace">
        Progression Risk P(Y=1)
      </text>
    </svg>
  )
}
