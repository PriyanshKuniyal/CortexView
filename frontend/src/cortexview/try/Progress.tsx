import { useEffect, useState } from 'react'

/**
 * What the user sees while the scan is uploaded and analyzed.
 *
 * Upload progress is real (bytes sent). The server analyzes in one request and doesn't report progress,
 * so the analysis bar and stage list are an estimate from elapsed time, and are labelled as such.
 */

/** The server's steps in order (backend/cortexview/pipeline.py), with rough start times in seconds. */
const STAGES: { at: number; label: string }[] = [
  { at: 0, label: 'Reading DICOM headers' },
  { at: 1.5, label: 'Choosing the best series' },
  { at: 3, label: 'Mapping the doctor’s outlines' },
  { at: 5, label: 'Finding the skull and brain boundary' },
  { at: 9, label: 'Separating gray matter, white matter and fluid' },
  { at: 14, label: 'Measuring every layer, pixel by pixel' },
]
const TYPICAL_S = 20

/** Short, accurate facts; nothing here is about the patient. */
const FACTS = [
  'An MRI scanner doesn’t take a photo. It measures radio signals from hydrogen atoms, mostly in water and fat.',
  'Gray matter looks gray because it is packed with nerve cell bodies. White matter is pale because of myelin, a fatty insulation around nerve fibres.',
  'The brain floats in about 150 mL of cerebrospinal fluid, which the body replaces several times a day.',
  'A 3D MRI is a stack of thin 2D slices. Each uploaded file is usually one slice.',
  'Every DICOM file records how many millimetres one pixel covers. That is how pixel counts become cm².',
  'On T1 post-contrast scans, a dye called gadolinium makes some tumours light up where the blood–brain barrier is damaged.',
  'FLAIR scans darken normal fluid, so swelling and lesions stand out bright.',
  'The adult brain is about 2% of body weight but uses roughly 20% of the body’s energy.',
  'CortexView’s numbers are counted from the outlines. The AI only paints the picture, and is checked against the outlines.',
]

export interface ProgressProps {
  fileCount: number
  totalMB: number
  upload: number // 0..1
  analyzeStart: number | null // performance.now() when the upload finished
}

export function Progress({ fileCount, totalMB, upload, analyzeStart }: ProgressProps) {
  const [now, setNow] = useState(() => performance.now())
  const [fact, setFact] = useState(() => Math.floor(Math.random() * FACTS.length))

  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 250)
    return () => clearInterval(id)
  }, [])
  useEffect(() => {
    const id = setInterval(() => setFact((f) => (f + 1) % FACTS.length), 6500)
    return () => clearInterval(id)
  }, [])

  const t = analyzeStart == null ? 0 : (now - analyzeStart) / 1000
  // Upload fills the first 20%; the analysis estimate approaches (never reaches) 100%.
  const est = analyzeStart == null ? 0 : 1 - Math.exp(-t / (TYPICAL_S * 0.55))
  const frac = 0.2 * upload + 0.79 * est
  const stage = analyzeStart == null ? -1 : STAGES.reduce((k, s, i) => (t >= s.at ? i : k), 0)

  return (
    <div className="tprog" role="status" aria-live="polite">
      <div className="tprog-hd">
        <span className="mono">{analyzeStart == null ? 'Uploading' : 'Analyzing on the server'}</span>
        <b>{Math.floor(frac * 100)}%</b>
      </div>
      <div className="tprog-bar">
        <i style={{ transform: `scaleX(${frac})` }} />
        <span className="tprog-sweep" />
      </div>
      <div className="tprog-sub">
        {analyzeStart == null
          ? `Sending ${fileCount} file${fileCount === 1 ? '' : 's'} · ${totalMB.toFixed(0)} MB · ${Math.round(upload * 100)}%`
          : `${t.toFixed(0)} s · a full scan folder usually takes 10–30 s (stages below are estimated)`}
      </div>
      <ol className="tprog-stages">
        <li className={analyzeStart == null ? 'run' : 'ok'}>
          <span className="st" />
          Upload {fileCount} file{fileCount === 1 ? '' : 's'}
        </li>
        {STAGES.map((s, i) => (
          <li key={s.label} className={i < stage ? 'ok' : i === stage ? 'run' : ''}>
            <span className="st" />
            {s.label}
          </li>
        ))}
      </ol>
      <div className="tprog-fact">
        <span className="mono">Did you know</span>
        <p key={fact}>{FACTS[fact]}</p>
      </div>
    </div>
  )
}
