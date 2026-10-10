import { useEffect, useState } from 'react'

const GLYPHS = '#%&*+=-:./\\|<>[]{}01'
const randomGlyph = () => GLYPHS[Math.floor(Math.random() * GLYPHS.length)]
const FRAME_MS = 33
const blank =(s: string) => s.replace(/[^\n]/g, ' ')

export const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

interface Options {
  start: boolean
  delay?: number
  duration?: number
}

/**
 * Animates `target` in like a decoding terminal: each character appears as
 * random glyphs at a random moment, then settles into its final value.
 */
export function useScramble(target: string, { start, delay = 0, duration = 900 }: Options) {
  const [text, setText] = useState(() => blank(target))
  const [done, setDone] = useState(false)

  useEffect(() => {
    if (!start || !target) return
    const instant = prefersReducedMotion()
    const reveal = Array.from(target, () => Math.random() * duration)
    const t0 = performance.now()

    const tick = () => {
      const t = performance.now() - t0 - delay
      if (instant || t >= duration) {
        clearInterval(id)
        setText(target)
        setDone(true)
        return
      }
      let out = ''
      for (let i = 0; i < target.length; i++) {
        const ch = target[i]
        if (ch === ' ' || ch === '\n' || t >= reveal[i]) out += ch
        else if (t >= reveal[i] * 0.4) out += randomGlyph()
        else out += ' '
      }
      setText(out)
    }

    const id = setInterval(tick, FRAME_MS)
    return () => clearInterval(id)
  }, [start, target, delay, duration])

  return { text, done }
}
