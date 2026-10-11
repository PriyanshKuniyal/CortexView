import { useEffect, useRef, useState } from 'react'
import { LAYERS, N_LAYERS, byId, css, type LayerId } from '../layers'
import { ApiError, analyze, fetchRender } from '../api/client'
import type { AnalyzeResponse, LayerMetrics } from '../api/types'
import { ANALYZE_TIMEOUT_MS, RENDER_TIMEOUT_MS } from '../config'
import type { Film } from '../film/useScrollFilm'
import { describeFiles, sequenceLabel, spacingLabel } from '../format'

type Mode = 'ill' | 'flat' | 'raw'
type Step = '' | 'run' | 'ok'
type RenderState = 'idle' | 'pending' | 'done' | 'failed'

const STEPS = ['Read DICOM', 'Find five layers', 'Measure each layer', 'Paint the illustration']
const IDLE_NOTE = 'Files are analyzed on the CortexView server, in memory only, and never written to disk.'

const isAbort = (e: unknown) => e instanceof DOMException && e.name === 'AbortError'
const errorText = (e: unknown) => (e instanceof ApiError ? e.message : 'Something went wrong. Please try again.')
const secs = (ms: number) => `${(ms / 1000).toFixed(1)} s`

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(t)
      reject(new DOMException('Aborted', 'AbortError'))
    })
  })
}

/** Where the slice sits inside the square frame, so non-square slices keep their aspect and the pins stay put. */
function boxStyle(w: number, h: number): React.CSSProperties {
  const padX = w >= h ? 0 : ((1 - w / h) / 2) * 100
  const padY = h >= w ? 0 : ((1 - h / w) / 2) * 100
  return { position: 'absolute', left: `${padX}%`, right: `${padX}%`, top: `${padY}%`, bottom: `${padY}%` }
}

function defaultSelection(res: AnalyzeResponse): LayerId {
  if (res.metrics.layers.lesion?.present) return 'lesion'
  return LAYERS.find((L) => res.metrics.layers[L.id]?.present)?.id ?? 'lesion'
}

function MeasurementCard({ res, sel }: { res: AnalyzeResponse | null; sel: LayerId }) {
  if (!res)
    return (
      <div className="card">
        <div className="hd">Measurements</div>
        <dl>
          <dt>Drop a scan to measure each layer.</dt>
        </dl>
      </div>
    )
  const L = byId(sel)
  const m: LayerMetrics | undefined = res.metrics.layers[sel]
  const source = res.layers.find((l) => l.id === sel)?.source
  return (
    <div className="card">
      <div className="hd">
        <i style={{ background: css(L.rgb) }} />
        {L.name}
      </div>
      {!m || !m.present ? (
        <dl>
          <dt>Not found on this slice</dt>
        </dl>
      ) : (
        <dl>
          <dt>Area</dt>
          <dd>{m.area_mm2 != null ? `${m.area_mm2.toFixed(0)} mm²` : `${m.area_px.toLocaleString()} px`}</dd>
          {m.pct_intracranial != null && (
            <>
              <dt>Share inside skull</dt>
              <dd>{m.pct_intracranial.toFixed(1)}%</dd>
            </>
          )}
          <dt>Separate pieces</dt>
          <dd>{m.n_components}</dd>
          {m.max_diameter_mm != null && (
            <>
              <dt>Longest span</dt>
              <dd>{m.max_diameter_mm.toFixed(1)} mm</dd>
              {m.hemisphere && (
                <>
                  <dt>Side</dt>
                  <dd>{m.hemisphere}</dd>
                </>
              )}
            </>
          )}
          <dt>Source</dt>
          <dd>{source === 'doctor' ? 'doctor label' : 'mask pixels'}</dd>
        </dl>
      )}
    </div>
  )
}

export function TryScene({ film }: { film: Film }) {
  const [res, setRes] = useState<AnalyzeResponse | null>(null)
  const [illustrated, setIllustrated] = useState<Partial<Record<LayerId, string>>>({})
  const [render, setRender] = useState<RenderState>('idle')
  const [busy, setBusy] = useState(false)
  const [steps, setSteps] = useState<Step[]>(['', '', '', ''])
  const [times, setTimes] = useState(['', '', '', ''])
  const [note, setNote] = useState(IDLE_NOTE)
  const [renderNote, setRenderNote] = useState('')
  const [error, setError] = useState('')
  const [mode, setMode] = useState<Mode>('ill')
  const [pl, setPl] = useState(0)
  const [sel, setSel] = useState<LayerId>('lesion')
  const [over, setOver] = useState(false)
  const [lastFiles, setLastFiles] = useState<File[] | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const run = useRef<AbortController | null>(null)

  useEffect(() => () => run.current?.abort(), [])

  // Seconds since the upload started, so a long analysis never looks frozen.
  useEffect(() => {
    if (!busy) return
    const t0 = performance.now()
    setElapsed(0)
    const id = setInterval(() => setElapsed(Math.floor((performance.now() - t0) / 1000)), 1000)
    return () => clearInterval(id)
  }, [busy])

  const setStep = (i: number, step: Step, time: string) => {
    setSteps((s) => s.map((v, k) => (k === i ? step : v)))
    setTimes((t) => t.map((v, k) => (k === i ? time : v)))
  }

  async function poll(r: AnalyzeResponse, signal: AbortSignal, t0: number) {
    const url = r.render.poll_url ?? `/api/render/${r.job_id}`
    const interval = r.render.poll_interval_ms ?? 1000
    const fail = (why: string) => {
      setRender('failed')
      setStep(3, '', 'failed')
      setRenderNote(`${why} Showing flat layers.`)
    }
    try {
      for (;;) {
        await sleep(interval, signal)
        if (performance.now() - t0 > RENDER_TIMEOUT_MS) return fail('The illustration is taking too long.')
        let out
        try {
          out = await fetchRender(url, signal)
        } catch (e) {
          if (isAbort(e)) throw e
          return fail(
            e instanceof ApiError && e.status === 404 ? 'This result expired before the illustration arrived.' : `Couldn’t get the illustration: ${errorText(e)}`,
          )
        }
        if (out.status === 'pending') continue
        if (out.status === 'failed') return fail(`The illustration couldn’t be made (${out.reason}).`)
        setIllustrated(Object.fromEntries(out.layers.map((l) => [l.id, l.illustrated_png])))
        setRender('done')
        setStep(3, 'ok', secs(performance.now() - t0))
        setRenderNote(`Illustration by ${out.backend}${out.fidelity ? (out.fidelity.passed ? ', fidelity check passed' : ', fidelity check failed') : ''}.`)
        return
      }
    } catch (e) {
      if (!isAbort(e)) throw e
    }
  }

  async function start(files: File[]) {
    if (!files.length) return
    run.current?.abort()
    const ac = new AbortController()
    run.current = ac
    setLastFiles(files)

    setBusy(true)
    setError('')
    setRenderNote('')
    setSteps(['run', 'run', 'run', ''])
    setTimes(['…', '…', '…', ''])
    setNote(`Analyzing ${describeFiles(files)}…`)
    const t0 = performance.now()

    let r: AnalyzeResponse
    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      ac.abort()
    }, ANALYZE_TIMEOUT_MS)
    try {
      r = await analyze(files, ac.signal)
    } catch (e) {
      if (isAbort(e) && !timedOut) return
      setBusy(false)
      setRender((st) => (st === 'pending' ? 'failed' : st)) // the previous scan's polling was cancelled
      setSteps(['', '', '', ''])
      setTimes(['', '', '', ''])
      setNote(res ? 'Still showing the previous scan.' : IDLE_NOTE)
      setError(timedOut ? `The analysis took longer than ${ANALYZE_TIMEOUT_MS / 60_000} minutes and was stopped. Try again, or upload a single series.` : errorText(e))
      return
    } finally {
      clearTimeout(timer)
    }
    const tDone = performance.now()

    const ready = r.layers.every((l) => l.illustrated_png)
    setRes(r)
    setBusy(false)
    setIllustrated(ready ? Object.fromEntries(r.layers.map((l) => [l.id, l.illustrated_png!])) : {})
    setMode('ill')
    setPl(0)
    setSel(defaultSelection(r))
    // The three server steps happen in one request, so there's one measured time for them, in the note.
    setSteps(['ok', 'ok', 'ok', ready ? 'ok' : r.render.status === 'pending' ? 'run' : ''])
    setTimes(['', '', '', ready ? '' : r.render.status === 'pending' ? '…' : 'failed'])
    setNote(`Analyzed ${describeFiles(files)} in ${secs(tDone - t0)}, upload included.`)

    if (ready) setRender('done')
    else if (r.render.status === 'pending') {
      setRender('pending')
      poll(r, ac.signal, tDone)
    } else {
      setRender('failed')
      setRenderNote('The illustration couldn’t be made. Showing flat layers.')
    }
  }

  const painting = render === 'pending'
  const hidden = (order: number) => busy || mode === 'raw' || order <= pl
  const showIll = mode === 'ill'
  const slice = res?.slice
  const tr = !res ? '' : mode === 'raw' ? 'MRI' : showIll && painting ? 'Flat · pending' : mode === 'flat' || !Object.keys(illustrated).length ? 'Flat' : 'Illustration'

  return (
    <section className="sc s6" data-name="Try it" id="try" ref={film.bindScene(6)}>
      <div className="in">
        <div className="mono" data-d="0.3">
          Scene 7 · Your turn
        </div>
        <h2 data-d="0.5">Drop a scan. Peel it.</h2>
        <div className="app" data-d="0.8">
          <div className="frame">
            {res && slice && (
              <div style={boxStyle(slice.width, slice.height)}>
                <img src={slice.mri_png} alt="MRI base slice" />
                <div className="layers">
                  {LAYERS.map((L) => {
                    const lr = res.layers.find((l) => l.id === L.id)
                    if (!lr) return null
                    const src = showIll && illustrated[L.id] ? illustrated[L.id] : lr.flat_png
                    return <img key={L.id} src={src} className={hidden(L.peel_order) ? 'off' : ''} alt={`${L.name} layer`} />
                  })}
                </div>
                <div>
                  {LAYERS.map((L) => {
                    const a = res.metrics.layers[L.id]?.anchor_px
                    if (!a || !res.metrics.layers[L.id]?.present) return null
                    return (
                      <button
                        key={L.id}
                        className={'pin' + (hidden(L.peel_order) ? ' off' : '') + (sel === L.id ? ' sel' : '')}
                        type="button"
                        style={{ left: `${(a[0] / slice.width) * 100}%`, top: `${(a[1] / slice.height) * 100}%` }}
                        aria-label={`${L.name} measurements`}
                        aria-pressed={sel === L.id}
                        onClick={() => setSel(L.id)}
                      >
                        <i style={{ background: css(L.rgb) }} />
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
            <div className="tick" />
            <div className="ov tl">
              {slice ? (
                <>
                  Your scan
                  <br />
                  {sequenceLabel(slice)}
                </>
              ) : (
                'No scan yet'
              )}
            </div>
            <div className="ov tr">{tr}</div>
            {slice && (
              <div className="ov bl">
                {/* slice_index is 0-based (backend/cortexview/pipeline.py); show it 1-based */}
                SL {slice.slice_index + 1} / {slice.n_slices}
                <br />
                {spacingLabel(slice.pixel_spacing_mm)}
              </div>
            )}
            <div className="ov br">
              Educational aid
              <br />
              not for diagnosis
            </div>
            {busy ? (
              <div className="badge">
                <span className="sp" />
                Reading the scan · {elapsed} s
              </div>
            ) : painting ? (
              <div className="badge">
                <span className="sp" />
                Painting illustration
              </div>
            ) : (
              !res && <div className="badge">Drop a scan to start</div>
            )}
          </div>
          <div className="panel">
            <label
              className={'drop' + (over ? ' over' : '')}
              onDragEnter={(e) => {
                e.preventDefault()
                setOver(true)
              }}
              onDragOver={(e) => {
                e.preventDefault()
                setOver(true)
              }}
              onDragLeave={(e) => {
                e.preventDefault()
                setOver(false)
              }}
              onDrop={(e) => {
                e.preventDefault()
                setOver(false)
                start([...e.dataTransfer.files])
              }}
            >
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
                <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 15v4h16v-4" />
              </svg>
              <div>
                <b>Drop .dcm files or a .zip</b>
                <span>or click to choose · a whole scan folder is fine</span>
              </div>
              <input
                type="file"
                accept=".dcm,.zip"
                multiple
                aria-label="Choose DICOM files"
                onChange={(e) => {
                  const files = [...(e.currentTarget.files ?? [])]
                  e.currentTarget.value = ''
                  start(files)
                }}
              />
            </label>
            <ol className="pipe" aria-live="polite">
              {STEPS.map((s, i) => (
                <li key={s} className={steps[i]}>
                  <span className="st" />
                  <span>{s}</span>
                  <span className="tm">{times[i]}</span>
                </li>
              ))}
            </ol>
            <div className="ctl">
              <div className="row">
                <label className="mono" htmlFor="peelR">
                  Peel
                </label>
                <span>{pl === 0 ? 'All layers' : pl === N_LAYERS ? 'Raw scan' : `${LAYERS.find((l) => l.peel_order === pl)?.name} off`}</span>
              </div>
              <input id="peelR" type="range" min={0} max={N_LAYERS} step={1} value={pl} disabled={!res} onChange={(e) => setPl(+e.currentTarget.value)} />
            </div>
            <div className="ctl">
              <span className="mono">View</span>
              <div className="seg" role="group" aria-label="View">
                {(
                  [
                    ['ill', 'Illustration'],
                    ['flat', 'Flat'],
                    ['raw', 'MRI'],
                  ] as const
                ).map(([m, label]) => (
                  <button key={m} type="button" aria-pressed={mode === m} disabled={!res} onClick={() => setMode(m)}>
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <MeasurementCard res={res} sel={sel} />
            {error && (
              <p className="note" role="alert">
                <b>{error}</b>
              </p>
            )}
            {res?.warnings.map((w) => (
              <p className="note" key={w.code}>
                {w.message}
              </p>
            ))}
            <p className="note">
              {note}
              {renderNote && ` ${renderNote}`}{' '}
              {lastFiles && !busy && (
                <button className="cta" type="button" style={{ fontSize: 12 }} onClick={() => start(lastFiles)}>
                  Run it again
                </button>
              )}
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
