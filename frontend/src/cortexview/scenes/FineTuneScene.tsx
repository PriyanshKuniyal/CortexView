import { useEffect, useRef } from 'react'
import type { DemoData } from '../demo/types'
import { clamp, ease, type Film } from '../film/useScrollFilm'

/**
 * Makes the rendering model visible in the story without suggesting that it
 * supplies any clinical measurement. The LoRA is trained on paired guide and
 * illustration images; the segmentation remains the source of truth.
 */
export function FineTuneScene({ film, demo }: { film: Film; demo: DemoData }) {
  const scene = useRef<HTMLElement>(null)

  useEffect(() => {
    return film.setHook(6, (u) => {
      const progress = ease(clamp(u * 1.25))
      const el = scene.current
      if (!el) return
      el.style.setProperty('--train', progress.toFixed(3))
      el.style.setProperty('--pulse', (0.35 + progress * 0.65).toFixed(3))
    })
  }, [film])

  return (
    <section className="sc s6 finetune" data-name="Fine-tuned AI" ref={(el) => {
      scene.current = el
      film.bindScene(6)(el)
    }}>
      <div className="in">
        <div className="mono" data-d="0.35">
          Scene 7 · Fine-tuned image AI
        </div>
        <div className="ft-head" data-d="0.55">
          <div>
            <h2>We trained the picture-maker to speak CortexView.</h2>
            <p className="lede">
              A small edit LoRA learns our illustration language from paired guides and curated target images—while the measured anatomy stays locked in the segmentation.
            </p>
          </div>
          <div className="ft-spec" aria-label="Fine-tuning configuration">
            <span>FLUX.2-klien 4b</span>
            <span>edit LoRA</span>
            <span>512 px · 2,000 steps</span>
          </div>
        </div>

        <div className="ft-flow" data-d="0.85">
          <article className="ft-card ft-guide">
            <div className="ft-card-hd">
              <span>01</span>
              <b>Segmentation guide</b>
            </div>
            <div className="ft-image">
              <img alt="Color-coded segmentation guide" src={demo.images.mask} />
              <span>Pixels + labels</span>
            </div>
            <p>The guide preserves the exact boundaries the renderer must follow.</p>
          </article>

          <div className="ft-link" aria-hidden="true">
            <i />
            <small>paired training</small>
          </div>

          <article className="ft-card ft-model">
            <div className="ft-card-hd">
              <span>02</span>
              <b>Fine-tune</b>
            </div>
            <div className="ft-core" aria-hidden="true">
              <div className="ft-grid" />
              <div className="ft-orbit a" />
              <div className="ft-orbit b" />
              <strong>LoRA</strong>
            </div>
            <p>We adapt the visual style, not the anatomy or the metrics.</p>
          </article>

          <div className="ft-link" aria-hidden="true">
            <i />
            <small>guided render</small>
          </div>

          <article className="ft-card ft-output">
            <div className="ft-card-hd">
              <span>03</span>
              <b>Illustrated layer</b>
            </div>
            <div className="ft-image">
              <img alt="CortexView-style illustration generated from the guide" src={demo.images.illus} />
              <span>Image generation</span>
            </div>
            <p>Every candidate is clipped back to the masks and checked before display.</p>
          </article>
        </div>

        <div className="ft-foot" data-d="1.1">
          <span className="ft-live"><i /> style learned from paired examples</span>
          <span>Segmentation → measurements → guide → fine-tuned renderer → fidelity gate</span>
        </div>
      </div>
    </section>
  )
}
