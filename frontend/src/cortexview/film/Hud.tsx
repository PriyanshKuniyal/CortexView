import { useCallback, useEffect, useRef } from 'react'
import type { Film } from './useScrollFilm'

const GLY = '<>/\\|=+*#01'

export function Hud({ film, names }: { film: Film; names: readonly string[] }) {
  const label = film.atEnd ? 'Watch from the start' : 'Skip to the demo'
  const txRef = useRef<HTMLSpanElement>(null)
  const raf = useRef(0)

  // The label text is owned here, not by React, because the scramble rewrites it every frame.
  const scramble = useCallback(() => {
    const el = txRef.current
    if (!el) return
    cancelAnimationFrame(raf.current)
    if (film.reduce) {
      el.textContent = label
      return
    }
    const t0 = performance.now(),
      dur = 520,
      rev = [...label].map((_, i) => (i / label.length) * dur * 0.8 + Math.random() * 90)
    const tick = (now: number) => {
      const t = now - t0
      let s = ''
      for (let i = 0; i < label.length; i++) {
        const ch = label[i]
        s += ch === ' ' || t >= rev[i] ? ch : GLY[(Math.random() * GLY.length) | 0]
      }
      el.textContent = s
      if (t < dur + 100) raf.current = requestAnimationFrame(tick)
      else el.textContent = label
    }
    raf.current = requestAnimationFrame(tick)
  }, [label, film.reduce])

  const first = useRef(true)
  useEffect(() => {
    if (first.current) {
      first.current = false
      if (txRef.current) txRef.current.textContent = label
      return
    }
    scramble()
  }, [label, scramble])
  useEffect(() => () => cancelAnimationFrame(raf.current), [])

  return (
    <>
      <div className="hud l">
        <button className="skip" type="button" title="Shortcut: S" onPointerEnter={scramble} onFocus={scramble} onClick={film.skip}>
          <span className="br a" />
          <span className="br b" />
          <span className="idx">[S]</span>
          <span className="tx" ref={txRef} />
          <span className="ar">{film.atEnd ? '↺' : '→'}</span>
        </button>
      </div>
      <div className="hud r">
        <div className="counter">
          <span className="brand">CortexView</span>
          <br />
          <b>{String(film.active + 1).padStart(2, '0')}</b> / {String(names.length).padStart(2, '0')}{' '}
          <span className="t">· {names[film.active]}</span>
        </div>
      </div>
      <div className="hint mono" ref={film.bindHint}>
        <span className="line" />
        Scroll to play
      </div>
      <nav className="scrub" aria-label="Scenes">
        {names.map((name, i) => (
          <button
            key={name}
            ref={film.bindScrub(i)}
            type="button"
            style={{ '--w': film.lengths[i] } as React.CSSProperties}
            aria-label={`Go to scene ${i + 1}: ${name}`}
            onClick={() => film.go(i)}
          />
        ))}
      </nav>
    </>
  )
}
