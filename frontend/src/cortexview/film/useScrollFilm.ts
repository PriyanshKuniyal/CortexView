/**
 * The scroll "film" engine from docs/design/CortexView.html: a smoothed scroll playhead drives
 * fixed scenes in and out, plus eased wheel/key scrolling, snapping out of half-finished
 * transitions, the scrubber and the skip button. Per-frame work is imperative (styles on refs),
 * as in the prototype; React state only changes when the active scene or the skip label does.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

/** Scene transition length, in viewport heights. */
export const T = 0.7

export const clamp = (v: number, a = 0, b = 1) => Math.min(b, Math.max(a, v))
export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

/** u = progress through the scene's hold, inn/out = transition progress, all 0–1. */
export type SceneHook = (u: number, inn: number, out: number) => void
export type FrameListener = (ps: number, now: number) => void

export interface Film {
  lengths: readonly number[]
  starts: readonly number[]
  last: number
  active: number
  atEnd: boolean
  reduce: boolean
  bindScene: (i: number) => (el: HTMLElement | null) => void
  bindScrub: (i: number) => (el: HTMLElement | null) => void
  bindHint: (el: HTMLElement | null) => void
  bindTrack: (el: HTMLElement | null) => void
  setHook: (i: number, hook: SceneHook) => () => void
  addFrameListener: (fn: FrameListener) => () => void
  go: (i: number, instant?: boolean) => void
  skip: () => void
}

const isTyping = (re: RegExp) => re.test(document.activeElement?.tagName ?? '')
const maxY = () => Math.max(0, document.documentElement.scrollHeight - innerHeight)

const newEngineState = () => ({
  scenes: [] as (HTMLElement | null)[],
  depth: [] as [HTMLElement, number][][],
  scrub: [] as (HTMLElement | null)[],
  hint: null as HTMLElement | null,
  track: null as HTMLElement | null,
  hooks: new Map<number, SceneHook>(),
  listeners: new Set<FrameListener>(),
  vh: 0,
  ps: 0,
  sTarget: 0,
  sCur: 0,
  sActive: false,
  curScene: -1,
  atEnd: false,
})
type EngineState = ReturnType<typeof newEngineState>

export function useScrollFilm(lengths: readonly number[]): Film {
  const starts = useMemo(() => {
    const a: number[] = []
    let acc = 0
    for (const l of lengths) {
      a.push(acc)
      acc += l
    }
    return a
  }, [lengths])
  const last = lengths.length - 1

  const [reduce] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [active, setActive] = useState(0)
  const [atEnd, setAtEnd] = useState(false)

  // Mutable engine state, read and written only by the frame loop, event handlers and ref callbacks (never rendered).
  const store = useRef<EngineState | null>(null)
  const get = useCallback(() => (store.current ??= newEngineState()), [])

  const smoothTo = useCallback(
    (y: number) => {
      const s = get()
      if (reduce) {
        scrollTo(0, y)
        return
      }
      if (!s.sActive) s.sCur = scrollY
      s.sTarget = clamp(y, 0, maxY())
      s.sActive = true
    },
    [reduce, get],
  )

  const go = useCallback(
    (i: number, instant?: boolean) => {
      const s = get()
      const y = starts[i] * s.vh
      if (instant) {
        s.sActive = false
        scrollTo(0, y)
        s.ps = starts[i] - 0.75
      } else smoothTo(y)
    },
    [starts, get, smoothTo],
  )

  const skip = useCallback(() => {
    if (get().ps > starts[last] - T) go(0, true)
    else go(last, true)
  }, [get, starts, last, go])

  // stable ref callbacks, one per scene / scrub button
  const sceneBinders = useMemo(
    () =>
      lengths.map((_, i) => (el: HTMLElement | null) => {
        const s = get()
        s.scenes[i] = el
        s.depth[i] = el ? [...el.querySelectorAll<HTMLElement>('[data-d]')].map((d) => [d, +(d.dataset.d ?? 0)]) : []
      }),
    [lengths, get],
  )
  const scrubBinders = useMemo(() => lengths.map((_, i) => (el: HTMLElement | null) => void (get().scrub[i] = el)), [lengths, get])
  const bindScene = useCallback((i: number) => sceneBinders[i], [sceneBinders])
  const bindScrub = useCallback((i: number) => scrubBinders[i], [scrubBinders])
  const bindHint = useCallback((el: HTMLElement | null) => void (get().hint = el), [get])
  const bindTrack = useCallback((el: HTMLElement | null) => void (get().track = el), [get])

  const setHook = useCallback(
    (i: number, hook: SceneHook) => {
      const { hooks } = get()
      hooks.set(i, hook)
      return () => {
        if (hooks.get(i) === hook) hooks.delete(i)
      }
    },
    [get],
  )
  const addFrameListener = useCallback(
    (fn: FrameListener) => {
      const { listeners } = get()
      listeners.add(fn)
      return () => void listeners.delete(fn)
    },
    [get],
  )

  useEffect(() => {
    const s = get()
    const END = starts[last]
    const size = () => {
      s.vh = innerHeight
      if (s.track) s.track.style.height = END * s.vh + s.vh + 'px'
    }
    size()
    s.sTarget = s.sCur = scrollY
    s.ps = scrollY / s.vh

    // smooth scroll (wheel + keys eased toward a target)
    const innerScroller = (start: EventTarget | null, dy: number) => {
      for (let el = start as HTMLElement | null; el && el !== document.body && el !== document.documentElement; el = el.parentElement) {
        if (el.scrollHeight > el.clientHeight + 1 && /(auto|scroll)/.test(getComputedStyle(el).overflowY)) {
          if (dy > 0 && el.scrollTop + el.clientHeight < el.scrollHeight - 1) return true
          if (dy < 0 && el.scrollTop > 0) return true
        }
      }
      return false
    }
    const onWheel = (e: WheelEvent) => {
      if (reduce || e.ctrlKey || innerScroller(e.target, e.deltaY)) return
      e.preventDefault()
      const d = e.deltaY * (e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? innerHeight : 1)
      smoothTo((s.sActive ? s.sTarget : scrollY) + d)
    }
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey || e.altKey || isTyping(/INPUT|TEXTAREA|SELECT/))) {
        const base = s.sActive ? s.sTarget : scrollY,
          page = innerHeight * 0.9
        const m = ({ ArrowDown: 120, ArrowUp: -120, PageDown: page, PageUp: -page, ' ': e.shiftKey ? -page : page } as Record<string, number>)[e.key]
        if (m != null) {
          e.preventDefault()
          smoothTo(base + m)
        } else if (e.key === 'Home') {
          e.preventDefault()
          smoothTo(0)
        } else if (e.key === 'End') {
          e.preventDefault()
          smoothTo(maxY())
        }
      }
      if ((e.key === 's' || e.key === 'S') && !e.metaKey && !e.ctrlKey && !isTyping(/INPUT|TEXTAREA/)) skip()
    }

    // snap gently out of half-finished transitions
    let snapT: ReturnType<typeof setTimeout> | undefined
    const onScroll = () => {
      clearTimeout(snapT)
      snapT = setTimeout(() => {
        if (s.sActive) return
        const p = scrollY / s.vh
        for (let i = 0; i < last; i++) {
          const a = starts[i] + lengths[i] - T,
            b = starts[i] + lengths[i]
          if (p > a + 0.02 && p < b - 0.02) {
            smoothTo((p - a < T / 2 ? a : b) * s.vh)
            break
          }
        }
      }, 180)
    }

    addEventListener('resize', size)
    addEventListener('wheel', onWheel, { passive: false })
    addEventListener('keydown', onKey)
    addEventListener('scroll', onScroll, { passive: true })

    let lastT = performance.now(),
      raf = 0
    const frame = (now: number) => {
      const dt = Math.min(64, now - lastT)
      lastT = now
      if (s.sActive) {
        s.sCur += (s.sTarget - s.sCur) * (1 - Math.exp(-dt / 110))
        if (Math.abs(s.sTarget - s.sCur) < 0.5) {
          s.sCur = s.sTarget
          s.sActive = false
        }
        scrollTo(0, s.sCur)
      }
      const pt = scrollY / s.vh
      s.ps += (pt - s.ps) * (reduce ? 1 : 1 - Math.exp(-dt / 70))
      if (Math.abs(pt - s.ps) < 0.0005) s.ps = pt
      const ps = s.ps

      let act = 0,
        best = -1
      s.scenes.forEach((sc, i) => {
        if (!sc) return
        const a = starts[i],
          L = lengths[i]
        const inn = i === 0 ? 1 : clamp((ps - (a - T)) / T)
        const out = i === last ? 0 : clamp((ps - (a + L - T)) / T)
        const u = clamp((ps - a) / Math.max(0.001, L - T))
        const vis = ease(inn) * (1 - ease(out))
        if (vis > best) {
          best = vis
          act = i
        }
        if (vis <= 0.002) {
          if (sc.style.visibility !== 'hidden') sc.style.visibility = 'hidden'
          return
        }
        sc.style.visibility = 'visible'
        const ei = ease(inn),
          eo = ease(out)
        sc.style.opacity = vis.toFixed(3)
        sc.style.transform = `scale(${(1 - (1 - ei) * 0.07 + eo * 0.09).toFixed(4)})`
        const bl = (1 - ei + eo) * 7
        sc.style.filter = bl > 0.05 && !reduce ? `blur(${bl.toFixed(2)}px)` : 'none'
        for (const [el, d] of s.depth[i] ?? []) {
          const y = ((1 - ei) * 1 - eo * 1) * d * 16
          el.style.transform = `translate3d(0,${y.toFixed(2)}vh,0)`
        }
        s.hooks.get(i)?.(u, inn, out)
      })
      s.scenes.forEach((sc, i) => sc?.classList.toggle('live', i === act))
      if (act !== s.curScene) {
        s.curScene = act
        setActive(act)
      }
      s.scrub.forEach((b, i) => b?.style.setProperty('--f', clamp((ps - starts[i]) / lengths[i]).toFixed(3)))
      const end = ps > starts[last] - T
      if (end !== s.atEnd) {
        s.atEnd = end
        setAtEnd(end)
      }
      if (s.hint) s.hint.style.opacity = ps < 0.15 ? '1' : '0'
      s.listeners.forEach((fn) => fn(ps, now))
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(snapT)
      removeEventListener('resize', size)
      removeEventListener('wheel', onWheel)
      removeEventListener('keydown', onKey)
      removeEventListener('scroll', onScroll)
    }
  }, [starts, lengths, last, reduce, get, smoothTo, skip])

  return { lengths, starts, last, active, atEnd, reduce, bindScene, bindScrub, bindHint, bindTrack, setHook, addFrameListener, go, skip }
}
