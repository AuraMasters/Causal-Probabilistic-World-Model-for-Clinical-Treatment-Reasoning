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
  0: { stroke: '#71717a', fill: 'bg-zinc-500', name: 'ZDV Monotherapy' },
  1: { stroke: '#a1a1aa', fill: 'bg-zinc-400', name: 'ZDV + ddI' },
  2: { stroke: '#ffffff', fill: 'bg-white', name: 'ZDV + ddC (Recommended)' },
  3: { stroke: '#d4d4d8', fill: 'bg-zinc-300', name: 'ddI Monotherapy' },
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
      transition={{ duration: 0.45, ease: 'easeOut' }}
      className="card-surface relative overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 p-6 sm:p-8 shadow-xl"
    >
      {/* Decision Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-5">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800 shadow-sm">
            <Award className="h-5 w-5 text-white" strokeWidth={2} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold tracking-[0.18em] text-zinc-300 uppercase">
                Model-Recommended Treatment Strategy
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-white text-black px-2 py-0.5 font-mono text-[10px] font-bold uppercase shadow-sm">
                <CheckCircle2 className="h-3 w-3 text-black" /> Rank #1
              </span>
            </div>
            <p className="font-sans text-xs text-zinc-400 mt-0.5">
              Maximizes Expected Clinical Utility under Pearlian interventional conditioning do(trt = k)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {riskDelta > 0 && (
            <span className="inline-flex items-center gap-1 rounded-md border border-zinc-700 bg-zinc-900 px-2.5 py-1 font-mono text-xs font-bold text-white">
              <TrendingDown className="h-3.5 w-3.5 text-zinc-300" /> -{formatPercent(riskDelta, 1)} Risk vs Monotherapy
            </span>
          )}
          {benefit && (
            <span
              className="rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1 font-mono text-xs font-medium text-zinc-300 cursor-help"
              title="Individualized treatment benefit relative to ZDV monotherapy."
            >
              {benefit.benefit_tier} ({benefit.benefit_percentile}th %ile Benefit)
            </span>
          )}
          <span className="rounded-md border border-zinc-800 bg-black px-3 py-1 font-mono text-xs font-medium text-zinc-400">
            Arm {recommended.treatment}
          </span>
          <span className="rounded-md border border-zinc-800 bg-zinc-900 px-3 py-1 font-mono text-xs font-bold text-white">
            {recommended.short_name}
          </span>
        </div>
      </div>

      {/* Regimen Hero Info */}
      <div className="mt-6 flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <p className="font-mono text-xs font-semibold tracking-wider text-zinc-400 uppercase">
              Model-Estimated Optimal Regimen
            </p>
            <span className="rounded bg-zinc-900 border border-zinc-800 px-2 py-0.5 font-mono text-[10px] font-bold text-zinc-300">
              {isContinuousModel ? 'Model B: Continuous / Hybrid SCM' : 'Model A: Discretized BN'}
            </span>
          </div>
          <h3 className="mt-1 font-display text-3xl font-extrabold tracking-tight text-white sm:text-4xl lg:text-5xl">
            {recommended.name}
          </h3>
          <p className="mt-2 flex items-center gap-2 text-sm text-zinc-400 font-medium">
            <span className="h-2 w-2 rounded-full bg-white inline-block" />
            {drugClass} &middot; Lowest model-estimated progression probability under evidence
          </p>
        </div>
      </div>

      {/* 3 Core Metric Pillar Cards */}
      <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-zinc-300 uppercase">
              Progression-Free Probability
            </span>
            <ShieldCheck className="h-4 w-4 text-white" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
            {formatPercent(recommended.p_label_0, 1)}
          </p>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            P(Progression-Free | do(trt={recommended.treatment})) = {formatProbability(recommended.p_label_0, 4)}
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-black border border-zinc-800">
            <div
              className="h-full bg-white rounded-full"
              style={{ width: `${recommended.p_label_0 * 100}%` }}
            />
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-zinc-400 uppercase">
              Progression Risk
            </span>
            <Activity className="h-4 w-4 text-zinc-400" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-zinc-300 sm:text-5xl">
            {formatPercent(recommended.p_label_1, 1)}
          </p>
          <p className="mt-1 font-mono text-xs text-zinc-500">
            P(Progression | do(trt={recommended.treatment})) = {formatProbability(recommended.p_label_1, 4)}
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-black border border-zinc-800">
            <div
              className="h-full bg-zinc-500 rounded-full"
              style={{ width: `${recommended.p_label_1 * 100}%` }}
            />
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs font-bold tracking-wider text-zinc-300 uppercase">
              Expected Clinical Utility
            </span>
            <TrendingUp className="h-4 w-4 text-white" />
          </div>
          <p className="mt-2 font-display text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
            {recommended.expected_utility.toFixed(4)}
          </p>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            Utility scale: 0.0000 (Progression) to 1.0000 (Progression-Free)
          </p>
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-black border border-zinc-800">
            <div
              className="h-full bg-white rounded-full"
              style={{ width: `${recommended.expected_utility * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* E-Value Sensitivity Analysis for Unmeasured Confounding */}
      {eValue && (
        <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-zinc-800 pb-3">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800">
                <ShieldCheck className="h-4 w-4 text-white" />
              </div>
              <div>
                <span className="font-mono text-xs font-bold text-white uppercase">
                  E-Value Sensitivity Analysis (VanderWeele &amp; Ding)
                </span>
                <p className="text-[11px] text-zinc-400">
                  Mathematical lower bound for robustness against unmeasured confounding
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="rounded-md bg-white text-black px-3 py-1 font-mono text-xs font-extrabold shadow-sm">
                E-Value = {eValue.e_value_point.toFixed(2)}
              </span>
              <span className="rounded-md bg-zinc-900 border border-zinc-800 px-2.5 py-1 font-mono text-[11px] text-zinc-300">
                RR = {eValue.risk_ratio_vs_monotherapy.toFixed(3)} vs Monotherapy
              </span>
            </div>
          </div>

          <p className="mt-3 text-xs text-zinc-300 leading-relaxed font-sans">
            <strong>Clinical Causal Interpretation:</strong> {eValue.interpretation}{' '}
            {eValue.risk_ratio_vs_monotherapy < 1.0 ? (
              eValue.is_robust ? (
                <span>
                  Because <span className="text-white font-bold">E = {eValue.e_value_point.toFixed(2)} &ge; 1.5</span>, the estimated protective benefit demonstrates moderate-to-high stability against plausible unmeasured confounding.
                </span>
              ) : (
                <span>
                  Because <span className="text-zinc-300 font-bold">E = {eValue.e_value_point.toFixed(2)} &lt; 1.5</span>, residual confounding of modest strength could potentially explain away the observed risk reduction.
                </span>
              )
            ) : (
              <span>
                Sensitivity analysis for protective confounding is not applicable when risk reduction is absent relative to monotherapy.
              </span>
            )}
          </p>
        </div>
      )}

      {/* MODEL B: Exact Continuous Gradient Sensitivity Analysis */}
      {isContinuousModel && continuousSens.length > 0 && (
        <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-white uppercase">
              <Sparkles className="h-3.5 w-3.5 text-zinc-400" />
              <span>Continuous Biomarker Gradient Sensitivity (∂P / ∂X_j)</span>
            </div>
            <span className="text-[11px] font-mono text-zinc-400">
              Exact continuous derivatives per clinical unit
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {continuousSens.map((item) => {
              const label = FEATURE_LABELS[item.variable] || item.variable
              return (
                <div
                  key={item.variable}
                  className="rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-xs"
                >
                  <div className="flex items-center justify-between font-mono font-bold">
                    <span className="text-white">{label}</span>
                    <span className="text-[11px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-200">
                      {item.percentage_points_per_step > 0
                        ? `+${item.percentage_points_per_step}% / +${item.clinical_step_unit}`
                        : item.percentage_points_per_step < 0
                          ? `${item.percentage_points_per_step}% / +${item.clinical_step_unit}`
                          : '0.0% / unit'}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] font-mono text-zinc-400 truncate">
                    Current Patient Value: <strong className="text-white">{item.raw_value}</strong>
                  </div>
                </div>
              )
            })}
          </div>

          <p className="mt-3 text-[11px] text-zinc-500 flex items-center gap-1.5">
            <HelpCircle className="h-3.5 w-3.5 text-zinc-600 shrink-0" />
            <span>
              Continuous gradient sensitivity calculates the exact marginal change in progression risk per clinical unit increase in each continuous biomarker, avoiding step-function binning distortions.
            </span>
          </p>
        </div>
      )}

      {/* Interactive Continuous "What-If" Sensitivity Simulator */}
      {whatIf && whatIf.trajectory_points.length > 0 && (
        <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/40 p-5 shadow-lg">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-zinc-800 pb-3">
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-zinc-900 border border-zinc-800">
                <LineChart className="h-4 w-4 text-white" />
              </div>
              <div>
                <span className="font-mono text-xs font-bold text-white uppercase">
                  Continuous What-If Trajectory Simulator
                </span>
                <p className="text-[11px] text-zinc-400">
                  Simulating continuous progression risk across all 4 trial arms as CD4 varies from 50 to 800 cells/mm³
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-zinc-300">Simulate CD4:</span>
              <input
                type="range"
                min={50}
                max={750}
                step={25}
                value={simCd4}
                onChange={(e) => setSimCd4(Number(e.target.value))}
                className="h-2 w-32 accent-white cursor-pointer"
              />
              <span className="font-mono text-xs font-bold text-white bg-zinc-800 px-2 py-0.5 rounded border border-zinc-700">
                {simCd4} cells/mm³
              </span>
            </div>
          </div>

          {/* SVG Multi-Arm Trajectory Plot */}
          <div className="my-4 flex justify-center">
            <WhatIfTrajectorySvg trajectory={whatIf.trajectory_points} currentCd4={whatIf.current_value} simulatedCd4={simCd4} />
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-mono border-t border-zinc-800 pt-3">
            {Object.entries(ARM_COLORS).map(([armKey, info]) => (
              <span key={armKey} className="flex items-center gap-1.5 text-zinc-300">
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
        <div className="mt-6 rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
          <div className="flex items-center justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-zinc-300 uppercase">
              <Sparkles className="h-3.5 w-3.5 text-zinc-400" />
              <span>Patient Characteristics Influencing Risk (Bayesian Evidence Ablation)</span>
            </div>
            <span className="text-[11px] font-mono text-zinc-400">
              Ranked by posterior sensitivity |ΔP|
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
            {discreteAttributions.slice(0, 6).map((item) => {
              const label = FEATURE_LABELS[item.feature] || item.feature
              return (
                <div
                  key={item.feature}
                  className="rounded-lg border border-zinc-800 bg-zinc-950 p-3 text-xs"
                >
                  <div className="flex items-center justify-between font-mono font-bold">
                    <span className="text-white">{label}</span>
                    <span className="text-[11px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-200">
                      {item.risk_impact > 0 ? `+${item.percentage_points}% Risk` : item.risk_impact < 0 ? `${item.percentage_points}% Protective` : 'Neutral (0%)'}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] font-mono text-zinc-400">
                    Observed Bin: <span className="text-zinc-200 font-medium">{item.observed_state}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Causal Interventional Rationale Callout */}
      <div className="mt-6 flex items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
        <div className="flex-1 text-xs leading-relaxed text-zinc-400 font-sans">
          <p className="font-semibold text-white">Causal Interventional Decision Rationale</p>
          <p className="mt-1">
            Conditioning on this patient's exact continuous biomarkers (CD4, CD8, Karnofsky, and prior ART history),{' '}
            <strong className="text-white">{recommended.name}</strong> yields the lowest model-estimated posterior probability of disease progression among all four randomized treatment strategies in the ACTG175 Causal World Model.
          </p>
          <p className="mt-2 text-[11px] text-zinc-500 italic">
            * Causal estimates reflect model-estimated potential outcomes under ACTG175 trial assumptions and do not replace personalized physician evaluation or constitute autonomous clinical prescribing decisions.
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
  trajectory: { cd4: number; arm_0: number; arm_1: number; arm_2: number; arm_3: number }[]
  currentCd4: number
  simulatedCd4: number
}) {
  const width = 480
  const height = 210
  const padding = 40

  const minX = 50
  const maxX = 750
  const minY = 0.0
  const maxY = 0.50

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
      <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#27272a" strokeWidth="1" />
      <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#27272a" strokeWidth="1" />

      {/* Grid lines */}
      {[0.1, 0.2, 0.3, 0.4].map((yVal) => (
        <line
          key={yVal}
          x1={padding}
          y1={scaleY(yVal)}
          x2={width - padding}
          y2={scaleY(yVal)}
          stroke="#18181b"
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
            strokeWidth={armNum === 2 ? 3 : 1.8}
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
        strokeWidth="1.5"
        strokeDasharray="4 4"
      />
      <circle cx={currentX} cy={scaleY(0.04)} r="3.5" fill="#ffffff" />

      {/* Simulated CD4 vertical marker */}
      {Math.abs(simX - currentX) > 5 && (
        <line
          x1={simX}
          y1={padding}
          x2={simX}
          y2={height - padding}
          stroke="#a1a1aa"
          strokeWidth="1.5"
          strokeDasharray="2 2"
        />
      )}

      <text x={width / 2} y={height - 8} fill="#a1a1aa" fontSize="10" textAnchor="middle" fontFamily="monospace">
        Baseline CD4 Count (cells/mm³)
      </text>
      <text x={12} y={height / 2} fill="#a1a1aa" fontSize="10" textAnchor="middle" transform={`rotate(-90 12 ${height/2})`} fontFamily="monospace">
        Progression Risk P(Y=1)
      </text>
    </svg>
  )
}
