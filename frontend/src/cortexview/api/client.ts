import type { AnalyzeResponse, ApiErrorBody, RenderResponse } from './types'

// Empty in local development: Vite proxies relative /api calls. In a Vercel
// build, set this to the HTTPS URL of the laptop's tunnel (no trailing slash).
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
const apiUrl = (path: string) => (path.startsWith('http://') || path.startsWith('https://') ? path : `${API_BASE_URL}${path}`)

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

function errorFrom(status: number, bodyText: string): ApiError {
  let body: Partial<ApiErrorBody> | null = null
  try {
    body = JSON.parse(bodyText)
  } catch {
    // not JSON (e.g. a proxy error page)
  }
  const code = body?.error?.code ?? (status === 404 ? 'not_found' : 'internal')
  const message =
    body?.error?.message ??
    FALLBACK_MESSAGES[code] ??
    (status === 404 ? 'The analysis server was not found.' : `The server answered ${status}.`)
  return new ApiError(code, message, status)
}

const aborted = () => new DOMException('Aborted', 'AbortError')

/**
 * POST /api/analyze: upload the files, get layers and metrics back (contract C3).
 * Uses XMLHttpRequest rather than fetch because only XHR reports upload progress.
 * `onUpload(fraction)` goes 0 → 1 while the bytes are sent; after that the server is analyzing.
 */
export function analyze(files: File[], signal?: AbortSignal, onUpload?: (fraction: number) => void): Promise<AnalyzeResponse> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(aborted())
    const xhr = new XMLHttpRequest()
    xhr.open('POST', apiUrl('/api/analyze'))
    xhr.upload.onprogress = (e) => e.lengthComputable && onUpload?.(e.loaded / e.total)
    xhr.upload.onload = () => onUpload?.(1)
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) return reject(errorFrom(xhr.status, xhr.responseText))
      try {
        resolve(JSON.parse(xhr.responseText))
      } catch {
        reject(new ApiError('internal', 'The server sent an answer we couldn’t read.', xhr.status))
      }
    }
    xhr.onerror = () => reject(new ApiError('network', "Couldn't reach the analysis server.", 0))
    xhr.onabort = () => reject(aborted())
    signal?.addEventListener('abort', () => xhr.abort())
    const form = new FormData()
    for (const f of files) form.append('files', f)
    xhr.send(form)
  })
}

/** GET /api/render/{job_id}: poll for the illustration (contract C3). */
export async function fetchRender(url: string, signal?: AbortSignal): Promise<RenderResponse> {
  let res: Response
  try {
    res = await fetch(apiUrl(url), { signal })
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    throw new ApiError('network', "Couldn't reach the analysis server.", 0)
  }
  if (!res.ok) throw errorFrom(res.status, await res.text())
  return res.json()
}
