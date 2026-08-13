import type { DiscretizationMetadata } from './types'

/**
 * Converts a real numerical patient value into the exact discretized
 * state used by the final model. The bin boundaries come from the
 * development-fitted discretization metadata served by the API — they
 * are never hard-coded here.
 */
export function discretizeValue(
  variable: string,
  rawValue: string | number,
  metadata: DiscretizationMetadata,
): string | null {
  const value = Number(rawValue)
  if (!Number.isFinite(value)) return null

  const variableMetadata = metadata.variables[variable]
  if (!variableMetadata) return null

  const [, e1, e2] = variableMetadata.edges
  const mid = e1
  const hi = e2

  if (variable === 'preanti') {
    if (value === 0) return 'zero'
    if (mid !== null && value <= mid) return 'positive_1'
    if (hi !== null && value <= hi) return 'positive_2'
    return 'positive_3'
  }

  if (variable === 'karnof') {
    if (mid !== null && value <= mid) return 'karnof_1'
    return 'karnof_2'
  }

  if (variable === 'age' || variable === 'wtkg' || variable === 'cd40' || variable === 'cd80') {
    if (mid !== null && value <= mid) return `${variable}_1`
    if (hi !== null && value <= hi) return `${variable}_2`
    return `${variable}_3`
  }

  return null
}

/** Human-readable description of the bin ranges for a variable. */
export function describeRanges(variable: string, metadata: DiscretizationMetadata): string[] {
  const variableMetadata = metadata.variables[variable]
  if (!variableMetadata) return []

  const [, e1, e2] = variableMetadata.edges
  const mid = e1
  const hi = e2

  const fmt = (value: number) => (Number.isInteger(value) ? String(value) : String(value))

  if (variable === 'preanti') {
    return [
      `0 → zero`,
      `0 < value ≤ ${fmt(mid ?? 0)} → positive_1`,
      `${fmt(mid ?? 0)} < value ≤ ${fmt(hi ?? 0)} → positive_2`,
      `> ${fmt(hi ?? 0)} → positive_3`,
    ]
  }

  if (variable === 'karnof') {
    return [
      `≤ ${fmt(mid ?? 0)} → karnof_1`,
      `> ${fmt(mid ?? 0)} → karnof_2`,
    ]
  }

  return [
    `≤ ${fmt(mid ?? 0)} → ${variable}_1`,
    `${fmt(mid ?? 0)} < value ≤ ${fmt(hi ?? 0)} → ${variable}_2`,
    `> ${fmt(hi ?? 0)} → ${variable}_3`,
  ]
}
