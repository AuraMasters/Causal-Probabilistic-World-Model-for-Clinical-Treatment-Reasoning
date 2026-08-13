export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`
}

export function formatProbability(value: number, digits = 4): string {
  return value.toFixed(digits)
}

export function formatUtility(value: number, digits = 4): string {
  return value.toFixed(digits)
}
