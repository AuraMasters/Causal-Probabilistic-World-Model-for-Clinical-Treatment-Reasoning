import type { AnalyzeResult, Overview, PatientInputs } from './types'

const BASE_URL = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })

  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new Error(`Unexpected response from the analysis server (${response.status}).`)
  }

  if (!response.ok) {
    const message =
      payload && typeof payload === 'object' && 'error' in payload
        ? String((payload as { error: string }).error)
        : `Request failed with status ${response.status}.`
    throw new Error(message)
  }

  return payload as T
}

export async function fetchOverview(): Promise<Overview> {
  return request<Overview>('/overview')
}

export async function analyzePatient(inputs: PatientInputs): Promise<AnalyzeResult> {
  return request<AnalyzeResult>('/analyze', {
    method: 'POST',
    body: JSON.stringify({ inputs }),
  })
}
