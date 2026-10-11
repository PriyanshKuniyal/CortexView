import type { DemoData } from '../demo/types'
import type { Film } from '../film/useScrollFilm'

export function GuardrailsScene({ film, demo }: { film: Film; demo: DemoData }) {
  const { lesionAreaPx, pixelAreaMm2, layers } = demo.metrics
  return (
    <section className="sc s5" data-name="Guardrails" ref={film.bindScene(5)}>
      <div className="in">
        <div className="mono" data-d="0.4">
          Scene 6 · Guardrails
        </div>
        <h2 data-d="0.6">Pretty is easy. Trustworthy is the work.</h2>
        <div className="gr">
          <div className="gc" data-d="0.8">
            <div className="im">
              <img alt="" src={demo.images.lesionOnMri} />
              <span className="tag">Suspected lesion · not a diagnosis</span>
            </div>
            <h3>Not a diagnosis</h3>
            <p>An educational aid. Every screen says so.</p>
          </div>
          <div className="gc" data-d="1.0">
            <div className="im">
              <img alt="" src={demo.images.mask} />
              <span className="tag">
                {lesionAreaPx.toLocaleString()} px × {pixelAreaMm2.toFixed(3)} mm² = {layers.lesion.areaMm2.toFixed(0)} mm²
              </span>
            </div>
            <h3>Counted, not generated</h3>
            <p>Areas come from mask pixels × pixel size.</p>
          </div>
          <div className="gc" data-d="1.2">
            <div className="im">
              <img alt="" src={demo.images.illus} />
              <span className="tag">Clipped to segmentation</span>
            </div>
            <h3>Clipped to the masks</h3>
            <p>The picture can't invent anatomy.</p>
          </div>
          <div className="gc" data-d="1.4">
            <div className="im">
              <img alt="" src={demo.images.mriDim} />
              <span className="big">≤ 5:00</span>
              <span className="tag">In memory · deleted on fetch</span>
            </div>
            <h3>Nothing kept</h3>
            <p>No patient details in logs or filenames.</p>
          </div>
        </div>
      </div>
    </section>
  )
}
