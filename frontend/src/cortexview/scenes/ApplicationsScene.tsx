import { useEffect, useRef, useState } from 'react'
import { clamp, type Film } from '../film/useScrollFilm'

/** Scene 8 · Applications. Product moments, deliberately without the repeated anatomy stack. */
type Application = { number: string; audience: string; title: string; body: string; outcome: string; proof: string }

const APPLICATIONS: Application[] = [
  { number: '01', audience: 'Patient conversations', title: 'Make the scan make sense.', body: 'Turn an intimidating MRI into a guided visual conversation: what is visible, where it is, and what the measured region means in everyday terms.', outcome: 'A patient leaves able to point to what was discussed.', proof: 'MRI → labelled view → plain-language measurement' },
  { number: '02', audience: 'Tumour boards', title: 'Start from the same picture.', body: 'Give surgeons, oncologists and radiologists one clean visual reference before the conversation begins, with boundaries and measurements kept traceable.', outcome: 'Less translation. More time on the decision.', proof: 'Shared visual · reproducible measurements' },
  { number: '03', audience: 'Teaching & learning', title: 'Learn anatomy in context.', body: 'Let students explore layered brain structure on an MRI-derived image instead of memorising a disconnected textbook diagram.', outcome: 'A memorable bridge from slice to structure.', proof: 'Layered anatomy · pixel-derived labels' },
  { number: '04', audience: 'Follow-up planning', title: 'Make change visible.', body: 'Use the same measurement logic at every review to establish a clear baseline for discussing what has changed and what needs attention next.', outcome: 'A more grounded conversation over time.', proof: 'Comparable baseline · deterministic workflow' },
]

export function ApplicationsScene({ film }: { film: Film }) {
  const [scrollIdx, setScrollIdx] = useState(0)
  const [hover, setHover] = useState<number | null>(null)
  const last = useRef(0)
  useEffect(() => film.setHook(7, (u) => {
    const i = clamp(Math.floor(u * APPLICATIONS.length), 0, APPLICATIONS.length - 1)
    if (i !== last.current) { last.current = i; setScrollIdx(i) }
  }), [film])

  const active = hover ?? scrollIdx
  const application = APPLICATIONS[active]
  return (
    <section className="sc s7" data-name="Applications" ref={film.bindScene(7)}>
      <div className="in apx-layout">
        <header className="apx-head">
          <div className="mono" data-d="0.3">Scene 8 · Applications</div>
          <h2 data-d="0.45">Applications</h2>
          <p className="apx-intro" data-d="0.6">How CortexView makes brain MRI easier to understand, discuss, and teach.</p>
        </header>
        <div className="apx-content" data-d="0.7">
          <ol className="apx-list" onMouseLeave={() => setHover(null)}>
            {APPLICATIONS.map((item, i) => <li key={item.number} className={i === active ? 'on' : ''}>
              <button type="button" onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onClick={() => setHover(i)}>
                <span className="apx-n mono">{item.number}</span><span className="apx-tx"><span className="apx-who mono">{item.audience}</span><span className="apx-where">{item.title}</span></span><span className="apx-arrow" aria-hidden="true">↗</span>
              </button>
            </li>)}
          </ol>
          <aside className="apx-brief" key={application.number} aria-live="polite">
            <div className="apx-brief-top mono"><span>Application {application.number}</span><span>Active</span></div>
            <p className="apx-proof mono">{application.proof}</p><h3>{application.title}</h3><p>{application.body}</p>
            <div className="apx-outcome"><i />{application.outcome}</div><div className="apx-count mono">{application.number} / {String(APPLICATIONS.length).padStart(2, '0')} · Applications</div>
          </aside>
        </div>
      </div>
    </section>
  )
}
