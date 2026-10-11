import { USE_MOCK } from '../config'
import type { AnalyzeResponse, ApiErrorBody, RenderResponse } from './types'

/** An API failure with a message that's safe to show to the user as-is. */
export class ApiError extends Error {
  code: string
  status: number
  constructor(code: string, message: string, status: number) {
    super(message)
    this.code = code
    this.status = status
  }
}

const FALLBACK_MESSAGES: Record<string, string> = {
  not_dicom: "One of the files isn't a DICOM file or a .zip.",
  not_mr: 'No MRI series was found in the upload.',
  no_axial_series: "There's nothing in the upload we can show as an axial slice.",
  too_large: 'The upload is too large.',
  bad_slice_index: "That slice doesn't exist in this scan.",
  internal: 'Something went wrong on the server.',
}

async function toApiError(res: Response): Promise<ApiError> {
  let body: Partial<ApiErrorBody> | null = null
  try {
    body = await res.json()
  } catch {
    // not JSON (e.g. a proxy error page)
  }
  const code = body?.error?.code ?? (res.status === 404 ? 'not_found' : 'internal')
  const message =
    body?.error?.message ??
    FALLBACK_MESSAGES[code] ??
    (res.status === 404 ? 'The analysis server was not found.' : `The server answered ${res.status}.`)
  return new ApiError(code, message, res.status)
}

async function request(url: string, init: RequestInit): Promise<Response> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ApiError('network', "Couldn't reach the analysis server.", 0)
  }
  if (!res.ok) throw await toApiError(res)
  return res
}

/** POST /api/analyze: upload the files, get layers and metrics back (contract C3). */
export async function analyze(files: File[], signal?: AbortSignal): Promise<AnalyzeResponse> {
  if (USE_MOCK) return (await import('./mock')).mockAnalyze(files, signal)
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const res = await request('/api/analyze', { method: 'POST', body: form, signal })
  return res.json()
}

/** GET /api/render/{job_id}: poll for the illustration (contract C3). */
export async function fetchRender(url: string, signal?: AbortSignal): Promise<RenderResponse> {
  if (USE_MOCK) return (await import('./mock')).mockRender(url, signal)
  const res = await request(url, { signal })
  return res.json()
}
