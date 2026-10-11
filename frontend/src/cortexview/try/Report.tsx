import { LAYERS, byId, css, type LayerId } from '../layers'
import type { AnalyzeResponse, Confidence } from '../api/types'
import { sequenceLabel, spacingLabel } from '../format'
import { fmt, shareInWords, volumeComparison, widthComparison } from './compare'

const CONF: Record<Confidence, string> = {
  high: 'High confidence',
  medium: 'Medium confidence',
  low: 'Low confidence',
  empty: 'Not on this slice',
}

const BACKEND: Record<string, string> = {
  'local-lora-v1': 'our fine-tuned FLUX.2 model',
  'workers-klein-4b': 'FLUX.2 klein 4B (Cloudflare Workers AI)',
  'workers-klein-9b': 'FLUX.2 klein 9B (Cloudflare Workers AI)',
  'workers-flux2-dev': 'FLUX.2 dev 32B (Cloudflare Workers AI)',
  'flat-fallback': 'flat colours (no AI image passed the check)',
  cache: 'a saved illustration of this exact slice',
}

const SEQ_WORDS: Record<string, string> = {
  T1_post: 'a T1 scan taken after a contrast dye',
  T1_pre: 'a T1 scan without contrast dye',
  T2: 'a T2 scan, where fluid looks bright',
  FLAIR: 'a FLAIR scan, where swelling looks bright',
  unknown: 'an MRI scan of unknown type',
}

export interface RenderInfo {
  backend: string
  fidelity?: { mean_iou: number; phantom_lesion_pct: number; passed: boolean }
}

interface Props {
  res: AnalyzeResponse
  sel: LayerId | null
  onSelect: (id: LayerId) => void
  onHover: (id: LayerId | null) => void
  render: RenderInfo | null
  renderState: 'idle' | 'pending' | 'done' | 'failed'
}

export function Report({ res, sel, onSelect, onHover, render, renderState }: Props) {
  const { slice, metrics } = res
  const les = metrics.layers.lesion
  const lesSource = res.layers.find((l) => l.id === 'lesion')?.source
  const tissue = LAYERS.filter((L) => L.id !== 'skull')
  const total = tissue.reduce((s, L) => s + (metrics.layers[L.id]?.pct_intracranial ?? 0), 0) || 1

  return (
    <div className="rep">
      <div className="rep-scan">
        <span className="mono">Your scan</span>
        <p>
          {SEQ_WORDS[slice.sequence] ?? 'an MRI scan'}, slice {slice.slice_index + 1} of {slice.n_slices}
          {slice.pixel_spacing_mm ? `, each pixel ${spacingLabel(slice.pixel_spacing_mm).replace(' mm/px', ' mm')} wide` : ''}.
          {les?.present ? ' This is the slice where the lesion is largest.' : ' This is a slice through the middle of the brain.'}
        </p>
        <div className="rep-tags mono">
          <span>{sequenceLabel(slice)}</span>
          <span>{slice.n_slices} slices</span>
          <span>{spacingLabel(slice.pixel_spacing_mm)}</span>
        </div>
      </div>

      {les?.present ? (
        <section className="rep-hero" onMouseEnter={() => onHover('lesion')} onMouseLeave={() => onHover(null)}>
          <div className="rep-hero-hd">
            <i style={{ background: css(byId('lesion').rgb) }} />
            <span>Suspected lesion</span>
            <em className="mono">{lesSource === 'doctor' ? 'from the doctor’s outline' : 'found automatically'}</em>
          </div>
          <div className="rep-big">
            {les.area_mm2 != null && (
              <div>
                <b>{fmt(les.area_mm2 / 100)}<small>cm²</small></b>
                <span>area on this slice</span>
              </div>
            )}
            {les.max_diameter_mm != null && (
              <div>
                <b>{les.max_diameter_mm.toFixed(0)}<small>mm</small></b>
                <span>widest point</span>
              </div>
            )}
            {les.volume_ml != null && les.volume_ml > 0 && (
              <div>
                <b>{fmt(les.volume_ml)}<small>mL</small></b>
                <span>across the whole scan</span>
              </div>
            )}
            {les.hemisphere && (
              <div>
                <b className="word">{les.hemisphere}</b>
                <span>side of the brain</span>
              </div>
            )}
          </div>
          <ul className="rep-scale">
            {les.max_diameter_mm != null && <li>At its widest, it is {widthComparison(les.max_diameter_mm)}.</li>}
            {les.volume_ml != null && les.volume_ml > 0 && <li>In total it takes up {volumeComparison(les.volume_ml)} of space.</li>}
            {les.pct_intracranial != null && (
              <li>
                On this slice it fills {les.pct_intracranial.toFixed(1)}% of the space inside the skull, {shareInWords(les.pct_intracranial)}.
              </li>
            )}
          </ul>
          <p className="rep-means">
            This marks tissue that looks different from its surroundings. It is <b>not a diagnosis</b>. Your doctor can explain what it means for you.
          </p>
        </section>
      ) : (
        <section className="rep-hero calm">
          <div className="rep-hero-hd">
            <span>No lesion marked on this slice</span>
          </div>
          <p className="rep-means">
            Nothing unusual was outlined here. That doesn’t rule anything out: automatic detection needs a FLAIR scan, and a doctor’s outline gives the best answer.
          </p>
        </section>
      )}

      <section>
        <div className="rep-h mono">Inside the skull, on this slice</div>
        <div className="rep-comp">
          {tissue.map((L) => {
            const p = metrics.layers[L.id]?.pct_intracranial ?? 0
            return p > 0 ? (
              <i
                key={L.id}
                style={{ width: `${(p / total) * 100}%`, background: css(L.rgb) }}
                title={`${L.name}: ${p.toFixed(1)}%`}
                onMouseEnter={() => onHover(L.id)}
                onMouseLeave={() => onHover(null)}
              />
            ) : null
          })}
        </div>
        <div className="rep-comp-key">
          {tissue.map((L) => (
            <span key={L.id}>
              <i style={{ background: css(L.rgb) }} />
              {L.name} <b>{(metrics.layers[L.id]?.pct_intracranial ?? 0).toFixed(0)}%</b>
            </span>
          ))}
        </div>
      </section>

      <section>
        <div className="rep-h mono">The five layers · tap one</div>
        <ul className="rep-layers">
          {LAYERS.map((L) => {
            const m = metrics.layers[L.id]
            const lr = res.layers.find((l) => l.id === L.id)
            const open = sel === L.id
            return (
              <li key={L.id} className={open ? 'open' : ''}>
                <button type="button" onClick={() => onSelect(L.id)} onMouseEnter={() => onHover(L.id)} onMouseLeave={() => onHover(null)} aria-expanded={open}>
                  <i style={{ background: css(L.rgb) }} />
                  <span className="rl-nm">{L.name}</span>
                  <span className="rl-vl">{m?.present ? (m.area_mm2 != null ? `${fmt(m.area_mm2 / 100)} cm²` : `${m.area_px} px`) : '—'}</span>
                  <span className={'rl-cf ' + (lr?.confidence ?? 'empty')}>{lr?.source === 'doctor' ? 'Doctor' : CONF[lr?.confidence ?? 'empty'].split(' ')[0]}</span>
                </button>
                {open && m && (
                  <div className="rep-detail">
                    <p>{L.description}</p>
                    <dl>
                      {m.present && m.pct_intracranial != null && (
                        <>
                          <dt>Share of brain space</dt>
                          <dd>{m.pct_intracranial.toFixed(1)}%</dd>
                        </>
                      )}
                      {m.present && (
                        <>
                          <dt>Separate pieces</dt>
                          <dd>{m.n_components}</dd>
                        </>
                      )}
                      {m.volume_ml != null && m.volume_ml > 0 && (
                        <>
                          <dt>Volume, whole scan</dt>
                          <dd>{fmt(m.volume_ml)} mL</dd>
                        </>
                      )}
                      {m.intensity?.ratio_to_brain != null && (
                        <>
                          <dt>Brightness vs. brain</dt>
                          <dd>×{m.intensity.ratio_to_brain.toFixed(2)}</dd>
                        </>
                      )}
                      <dt>Outline from</dt>
                      <dd>{lr?.source === 'doctor' ? 'doctor’s file' : 'automatic'}</dd>
                      <dt>Confidence</dt>
                      <dd>{CONF[lr?.confidence ?? 'empty']}</dd>
                    </dl>
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      </section>

      <section className="rep-foot">
        <p>
          <b>Picture:</b>{' '}
          {renderState === 'pending'
            ? 'being painted…'
            : renderState === 'failed'
              ? 'the illustration couldn’t be made, so the layers are flat colours.'
              : render
                ? `${BACKEND[render.backend] ?? render.backend}${render.fidelity ? `. Checked against the outlines: ${Math.round(render.fidelity.mean_iou * 100)}% overlap, ${render.fidelity.phantom_lesion_pct.toFixed(1)}% stray lesion colour.` : '.'}`
                : '—'}
        </p>
        <p>
          <b>Numbers:</b> counted from the outlines and the pixel size in the file. No AI writes or changes them.
        </p>
        {res.warnings.length > 0 && (
          <details>
            <summary>Notes about this scan ({res.warnings.length})</summary>
            <ul>
              {res.warnings.map((w) => (
                <li key={w.code}>{w.message}</li>
              ))}
            </ul>
          </details>
        )}
        <p className="rep-disc">{res.disclaimer}</p>
      </section>
    </div>
  )
}
