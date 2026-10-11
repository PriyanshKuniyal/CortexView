import type { DemoData } from '../demo/types'
import type { Film } from '../film/useScrollFilm'

export function ApplicationsScene({ film, demo }: { film: Film; demo: DemoData }) {
  return (
    <section className="sc s7 applications" data-name="Applications" ref={film.bindScene(7)}>
      <div className="in">
        <div className="mono" data-d="0.35">
          Scene 8 · Applications
        </div>
        <div className="app-head" data-d="0.55">
          <div>
            <h2>From a single scan to a shared view of the brain.</h2>
            <p className="lede">
              The same pipeline is useful in the places where teams need a fast read on anatomy without losing the raw evidence.
            </p>
          </div>
        </div>

        <div className="app-grid" data-d="0.85">
          <article className="app-card">
            <span className="app-tag">Neuro-oncology</span>
            <h3>Follow a suspected lesion in context.</h3>
            <p>
              Clinicians can compare the affected slice against the surrounding tissue and keep the lesion anchored to the real anatomy instead of a floating thumbnail.
            </p>
            <ul>
              <li>Lesion boundary</li>
              <li>Nearby cortex and white matter</li>
              <li>Repeatable measurements</li>
            </ul>
          </article>

          <article className="app-card">
            <span className="app-tag">Surgical planning</span>
            <h3>Turn the scan into a readable map.</h3>
            <p>
              Trainees and specialists can peel layers, inspect the cortical ribbon, and reason about the structures around a target region before a discussion starts.
            </p>
            <ul>
              <li>Layer-by-layer anatomy</li>
              <li>Cross-sectional view</li>
              <li>Human-readable labels</li>
            </ul>
          </article>

          <article className="app-card">
            <span className="app-tag">Research triage</span>
            <h3>Scan batches without losing control.</h3>
            <p>
              Dense study sets are hard to review manually. CortexView helps sort, compare and flag scans where the anatomy and segmentation disagree or deserve a second look.
            </p>
            <ul>
              <li>Batch comparison</li>
              <li>QC across cases</li>
              <li>Layout for downstream review</li>
            </ul>
          </article>

          <article className="app-card">
            <span className="app-tag">Education</span>
            <h3>Show the anatomy, not just the numbers.</h3>
            <p>
              Teachers can pull a slice apart and rebuild it, demonstrating how tissue boundaries relate to what appears in the raw MRI and the measured mask.
            </p>
            <ul>
              <li>Visual teaching aid</li>
              <li>Transparent model logic</li>
              <li>Accessible to non-experts</li>
            </ul>
          </article>
        </div>

        <div className="app-showcase" data-d="1.1">
          <div className="mini-stack">
            <img src={demo.images.illus} alt="Illustrated layer view" />
            <img src={demo.images.mask} alt="Segmentation mask" />
          </div>
          <div className="mini-copy">
            <div className="mono">Application note</div>
            <h3>It is an interface for interpretation, not a replacement for expert judgment.</h3>
            <p>
              The AI paints the illustration. The segmentation remains the constraint. The result is a fast, visible explanation that still respects the measured anatomy.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
