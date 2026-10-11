import { useEffect, useState } from 'react'

/**
 * Bundled sample scans (frontend/public/samples/, built by scripts/samples/build_samples.py).
 * They come from public datasets CortexView never trained or tuned on, so judges can try it on unseen scans.
 */
export interface Sample {
  id: string
  title: string
  detail: string
  kind: 'tumor' | 'normal'
  file: string
  thumb: string
  size_mb: number
  slices: number
  credit: { label: string; url: string }
}

const BASE = import.meta.env.BASE_URL

export function useSamples(): Sample[] {
  const [samples, setSamples] = useState<Sample[]>([])
  useEffect(() => {
    let live = true
    fetch(`${BASE}samples/manifest.json`)
      .then((r) => (r.ok ? r.json() : []))
      .then((s: Sample[]) => live && setSamples(s))
      .catch(() => live && setSamples([]))
    return () => {
      live = false
    }
  }, [])
  return samples
}

/** Fetch a sample's zip as a File, ready for the same upload path as a dropped file. */
export async function sampleFile(s: Sample): Promise<File> {
  const r = await fetch(BASE + s.file)
  if (!r.ok) throw new Error(`Couldn't load the sample (${r.status}).`)
  return new File([await r.blob()], `${s.id}.zip`, { type: 'application/zip' })
}

type Filter = 'all' | 'tumor' | 'normal'

export function SampleList({ samples, onPick, compact, busyId }: {
  samples: Sample[]
  onPick: (s: Sample) => void
  compact?: boolean
  busyId?: string | null
}) {
  const [filter, setFilter] = useState<Filter>('all')
  if (!samples.length) return null
  const shown = samples.filter((s) => filter === 'all' || s.kind === filter)
  const n = (k: Filter) => (k === 'all' ? samples.length : samples.filter((s) => s.kind === k).length)

  return (
    <section className={'smp' + (compact ? ' compact' : '')} aria-label="Sample scans">
      <div className="smp-hd">
        <div>
          <span className="mono">No scan handy? Try one of these</span>
          {!compact && (
            <p>
              {samples.length} real brain MRIs from public research datasets that CortexView never trained on. Different tumor
              locations, and healthy brains for comparison. Click one to run it.
            </p>
          )}
        </div>
        <div className="seg smp-filter" role="group" aria-label="Filter samples">
          {(['all', 'tumor', 'normal'] as const).map((k) => (
            <button key={k} type="button" aria-pressed={filter === k} onClick={() => setFilter(k)}>
              {k === 'all' ? 'All' : k === 'tumor' ? 'Tumor' : 'Healthy'} <small>{n(k)}</small>
            </button>
          ))}
        </div>
      </div>
      <ul className="smp-grid">
        {shown.map((s) => (
          <li key={s.id}>
            <button type="button" className={'smp-card' + (busyId === s.id ? ' busy' : '')} onClick={() => onPick(s)} disabled={!!busyId}>
              <img src={BASE + s.thumb} alt="" loading="lazy" />
              <span className="smp-tx">
                <b>{s.title}</b>
                <em>{s.detail}</em>
              </span>
              <span className={'smp-kind ' + s.kind}>{s.kind === 'tumor' ? 'Tumor' : 'Healthy'}</span>
            </button>
            <a className="smp-credit mono" href={s.credit.url} target="_blank" rel="noreferrer" title={`Source: ${s.credit.label}`}>
              {s.credit.label} ↗
            </a>
          </li>
        ))}
      </ul>
    </section>
  )
}
