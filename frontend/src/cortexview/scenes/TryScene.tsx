import { useEffect, useRef, useState } from 'react'
import '../try/try.css'
import { LAYERS, N_LAYERS, css, type LayerId } from '../layers'
import { ApiError, analyze, fetchRender } from '../api/client'
import type { AnalyzeResponse } from '../api/types'
import { ANALYZE_TIMEOUT_MS, RENDER_TIMEOUT_MS } from '../config'
import type { Film } from '../film/useScrollFilm'
import { Progress } from '../try/Progress'
import { Report, type RenderInfo } from '../try/Report'
import { Viewer, type View } from '../try/Viewer'
import { SampleList, sampleFile, useSamples, type Sample } from '../try/Samples'
import { Credits } from '../try/Credits'

type RenderState = 'idle' | 'pending' | 'done' | 'failed'

const isAbort = (e: unknown) => e instanceof DOMException && e.name === 'AbortError'
const errorText = (e: unknown) => (e instanceof ApiError ? e.message : 'Something went wrong. Please try again.')

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    signal.addEventListener('abort', () => {
      clearTimeout(t)
      reject(new DOMException('Aborted', 'AbortError'))
    })
  })
}

function DropZone({ big, onFiles }: { big?: boolean; onFiles: (f: File[]) => void }) {
  const [over, setOver] = useState(false)
  const on = (e: React.DragEvent) => {
    e.preventDefault()
    setOver(true)
  }
  return (
    <label
      className={'drop' + (big ? ' big' : '') + (over ? ' over' : '')}
      onDragEnter={on}
      onDragOver={on}
      onDragLeave={(e) => {
        e.preventDefault()
        setOver(false)
      }}
      onDrop={(e) => {
        e.preventDefault()
        e.stopPropagation()
        setOver(false)
        onFiles([...e.dataTransfer.files])
      }}
    >
      <svg width={big ? 34 : 22} height={big ? 34 : 22} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
        <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 15v4h16v-4" />
      </svg>
      <div>
        <b>{big ? 'Drop a brain MRI here' : 'Analyze another scan'}</b>
        <span>{big ? '.dcm files from one scan, or a .zip of the folder. Doctor’s outline files (DICOM-SEG) are welcome too.' : 'drop files or click'}</span>
      </div>
      <input
        type="file"
        accept=".dcm,.zip"
        multiple
        aria-label="Choose DICOM files"
        onChange={(e) => {
          const files = [...(e.currentTarget.files ?? [])]
          e.currentTarget.value = ''
          onFiles(files)
        }}
      />
    </label>
  )
}

export function TryScene({ film }: { film: Film }) {
  const [res, setRes] = useState<AnalyzeResponse | null>(null)
  const [illustrated, setIllustrated] = useState<Partial<Record<LayerId, string>>>({})
  const [render, setRender] = useState<RenderState>('idle')
  const [renderInfo, setRenderInfo] = useState<RenderInfo | null>(null)
  const [job, setJob] = useState<{ count: number; mb: number; upload: number; analyzeStart: number | null } | null>(null)
  const [error, setError] = useState('')
  const [view, setView] = useState<View>('3d')
  const [peel, setPeel] = useState(0)
  const [sel, setSel] = useState<LayerId | null>(null)
  const [hover, setHover] = useState<LayerId | null>(null)
  const [source, setSource] = useState<Sample | null>(null)
  const [picking, setPicking] = useState<string | null>(null)
  const samples = useSamples()
  const run = useRef<AbortController | null>(null)

  useEffect(() => () => run.current?.abort(), [])

  async function poll(r: AnalyzeResponse, signal: AbortSignal) {
    const url = r.render.poll_url ?? `/api/render/${r.job_id}`
    const interval = r.render.poll_interval_ms ?? 1000
    const t0 = performance.now()
    try {
      for (;;) {
        await sleep(interval, signal)
        if (performance.now() - t0 > RENDER_TIMEOUT_MS) return setRender('failed')
        let out
        try {
          out = await fetchRender(url, signal)
        } catch (e) {
          if (isAbort(e)) throw e
          return setRender('failed')
        }
        if (out.status === 'pending') continue
        if (out.status === 'failed') return setRender('failed')
        setIllustrated(Object.fromEntries(out.layers.map((l) => [l.id, l.illustrated_png])))
        setRenderInfo({ backend: out.backend, fidelity: out.fidelity })
        setRender('done')
        return
      }
    } catch (e) {
      if (!isAbort(e)) throw e
    }
  }

  async function pick(s: Sample) {
    setPicking(s.id)
    setError('')
    let f: File
    try {
      f = await sampleFile(s)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Couldn’t load the sample.')
      return
    } finally {
      setPicking(null)
    }
    start([f], s)
  }

  async function start(files: File[], from: Sample | null = null) {
    if (!files.length) return
    run.current?.abort()
    const ac = new AbortController()
    run.current = ac
    setError('')
    setJob({ count: files.length, mb: files.reduce((s, f) => s + f.size, 0) / 2 ** 20, upload: 0, analyzeStart: null })

    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      ac.abort()
    }, ANALYZE_TIMEOUT_MS)
    let r: AnalyzeResponse
    try {
      r = await analyze(files, ac.signal, (f) =>
        setJob((j) => j && { ...j, upload: f, analyzeStart: f >= 1 ? (j.analyzeStart ?? performance.now()) : null }),
      )
    } catch (e) {
      if (isAbort(e) && !timedOut) return
      setJob(null)
      setError(
        timedOut
          ? `The analysis took longer than ${ANALYZE_TIMEOUT_MS / 60_000} minutes and was stopped. Try again, or upload a single series.`
          : errorText(e),
      )
      return
    } finally {
      clearTimeout(timer)
    }

    const ready = r.layers.every((l) => l.illustrated_png)
    setSource(from)
    setRes(r)
    setJob(null)
    setIllustrated(ready ? Object.fromEntries(r.layers.map((l) => [l.id, l.illustrated_png!])) : {})
    setRenderInfo(null)
    setPeel(0)
    setView('3d')
    setHover(null)
    setSel(r.metrics.layers.lesion?.present ? 'lesion' : null)
    if (ready) setRender('done')
    else if (r.render.status === 'pending') {
      setRender('pending')
      poll(r, ac.signal)
    } else setRender('failed')
  }

  function cancel() {
    run.current?.abort()
    setJob(null)
  }

  const peeled = LAYERS.find((l) => l.peel_order === peel)
  const showRes = res != null && job == null

  return (
    <section
      className={'sc s8 try2' + (showRes ? ' has-res' : '')}
      data-name="Try it"
      id="try"
      ref={film.bindScene(8)}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault()
        if (!job) start([...e.dataTransfer.files])
      }}
    >
      <div className="in">
        <div className="mono" data-d="0.3">
          Scene 9 · Your turn
        </div>
        <h2 data-d="0.5">{showRes ? 'Your scan, layer by layer.' : 'Add a scan to begin.'}</h2>

        {job ? (
          <div className="try-busy">
            <Progress fileCount={job.count} totalMB={job.mb} upload={job.upload} analyzeStart={job.analyzeStart} />
            <button type="button" className="cta" onClick={cancel}>
              Cancel
            </button>
          </div>
        ) : !res ? (
          <div className="try-idle" data-d="0.8">
            <DropZone big onFiles={(f) => start(f)} />
            <ol className="try-how">
              <li>
                <b>Five layers, mapped by AI.</b> A trained nnU-Net model looks for the skull, gray matter, white matter, fluid and any suspected lesion.
              </li>
              <li>
                <b>Measured from the scan.</b> Areas, widths and volumes come from the outlines and the pixel size stored in the file.
              </li>
              <li>
                <b>Shown with care.</b> The illustration follows the measured outlines, so it cannot invent anatomy outside them.
              </li>
            </ol>
            <p className="note">Files are analyzed in memory on the CortexView server and never written to disk. Educational aid, not for diagnosis.</p>
            <SampleList samples={samples} onPick={pick} busyId={picking} />
          </div>
        ) : (
          <div className="try-res">
            <div className="try-view">
              <div className="try-bar">
                <div className="seg" role="group" aria-label="View">
                  {(
                    [
                      ['3d', '3D layers'],
                      ['slice', 'Slice'],
                      ['mri', 'Raw MRI'],
                    ] as const
                  ).map(([v, label]) => (
                    <button key={v} type="button" aria-pressed={view === v} onClick={() => setView(v)}>
                      {label}
                    </button>
                  ))}
                </div>
                <div className="try-peel">
                  <label className="mono" htmlFor="peelR">
                    Peel
                  </label>
                  <input
                    id="peelR"
                    type="range"
                    min={0}
                    max={N_LAYERS}
                    step={1}
                    value={peel}
                    disabled={view === 'mri'}
                    onChange={(e) => setPeel(+e.currentTarget.value)}
                  />
                  <span>
                    {peeled ? (
                      <>
                        <i style={{ background: css(peeled.rgb) }} />
                        {peel === N_LAYERS ? 'All lifted' : `${peeled.name} lifted`}
                      </>
                    ) : (
                      'All layers'
                    )}
                  </span>
                </div>
              </div>
              <Viewer
                res={res}
                illustrated={illustrated}
                painting={render === 'pending'}
                view={view}
                peel={peel}
                sel={sel}
                hover={hover}
                onSelect={(id) => setSel((s) => (s === id ? null : id))}
                onHover={setHover}
              />
              {source && (
                <p className="try-src">
                  Sample: <b>{source.title}</b> · {source.detail} ·{' '}
                  <a href={source.credit.url} target="_blank" rel="noreferrer">
                    {source.credit.label} ↗
                  </a>
                </p>
              )}
              <DropZone onFiles={(f) => start(f)} />
              <SampleList samples={samples} onPick={pick} busyId={picking} compact />
            </div>
            <Report
              res={res}
              sel={sel}
              onSelect={(id) => setSel((s) => (s === id ? null : id))}
              onHover={setHover}
              render={renderInfo}
              renderState={render}
            />
          </div>
        )}
        {error && (
          <p className="try-err" role="alert">
            {error}
          </p>
        )}
        {!job && <Credits />}
      </div>
    </section>
  )
}
