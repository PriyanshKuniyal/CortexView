import { useEffect, useMemo, useRef, useState } from 'react'
import { prefersReducedMotion, useScramble } from '@/hooks/useScramble'
import { buildField, COLS, gridToText, ROWS, type Cell } from '@/lib/neurons'
import { cn } from '@/lib/utils'

const LINE = 1.2 // line-height in em
const CHAR_W = 0.6 // IBM Plex Mono advance width in em
const TICK_MS = 60
const TRAIL = 4
const PULSE = ['#', '*', '+', ':', '.'] // head, then fading trail
const FIRING: Record<string, string> = { '(': '[', ')': ']', o: '#', '@': '#' }

interface Pulse {
  edge: number
  forward: boolean
  pos: number
  speed: number
}

/**
 * The centrepiece: a static ASCII microcircuit with action potentials that
 * travel along axons, light up the receiving soma, and sometimes cascade.
 * Activity rises while the model is generating.
 */
interface NeuronClusterProps {
  /** begin decoding the tissue in */
  start: boolean
  /** raise the firing rate, e.g. while the model is generating */
  active: boolean
  className?: string
}

export function NeuronCluster({ start, active, className }: NeuronClusterProps) {
  const field = useMemo(() => buildField(), [])
  const structureText = useMemo(() => gridToText(field.structure), [field])
  const dustText = useMemo(() => gridToText(field.dust), [field])
  const tissue = useScramble(structureText, { start, delay: 200, duration: 1800 })

  const boxRef = useRef<HTMLDivElement>(null)
  const [fontSize, setFontSize] = useState(10)
  const [live, setLive] = useState<string | null>(null)

  const activeRef = useRef(active)
  useEffect(() => {
    activeRef.current = active
  }, [active])

  // Scale the art to fit its box at any viewport size
  useEffect(() => {
    const el = boxRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setFontSize(Math.max(3.5, Math.min(13, width / (COLS * CHAR_W), height / (ROWS * LINE))))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Spike simulation
  useEffect(() => {
    if (!tissue.done || prefersReducedMotion()) return
    const pulses: Pulse[] = []
    const firing = new Map<number, number>() // soma index -> glow until
    let lastSpawn = 0

    const cellAt = (p: Pulse, i: number): Cell | undefined => {
      const path = field.edges[p.edge].path
      return p.forward ? path[i] : path[path.length - 1 - i]
    }
    const launch = (edge: number, forward: boolean) =>
      pulses.push({ edge, forward, pos: 0, speed: 0.6 + Math.random() * 0.5 })

    const tick = () => {
      const now = performance.now()
      const busy = activeRef.current

      if (now - lastSpawn > (busy ? 110 : 420) && pulses.length < (busy ? 32 : 12)) {
        launch(Math.floor(Math.random() * field.edges.length), Math.random() < 0.5)
        lastSpawn = now
      }

      for (let i = pulses.length - 1; i >= 0; i--) {
        const p = pulses[i]
        p.pos += p.speed
        const edge = field.edges[p.edge]
        if (p.pos - TRAIL < edge.path.length) continue

        // Arrived: fire the receiving neuron and maybe propagate
        pulses.splice(i, 1)
        const target = p.forward ? edge.to : edge.from
        firing.set(target, now + 450)
        if (Math.random() < (busy ? 0.8 : 0.5)) {
          const out = field.edges
            .map((e, idx) => ({ e, idx }))
            .filter(({ e, idx }) => idx !== p.edge && (e.from === target || e.to === target))
          if (out.length) {
            const { e, idx } = out[Math.floor(Math.random() * out.length)]
            launch(idx, e.from === target)
          }
        }
      }

      // Monochrome: a signal is drawn by swapping glyphs, densest at its head
      const frame = field.structure.map((row) => [...row])
      const stamp = (c: Cell | undefined, ch: string) => {
        if (c) frame[c.y][c.x] = ch
      }

      for (const p of pulses) {
        const at = Math.floor(p.pos)
        for (let t = TRAIL; t >= 1; t--) if (at - t >= 0) stamp(cellAt(p, at - t), PULSE[t])
        stamp(cellAt(p, at), PULSE[0])
      }
      for (const [soma, until] of firing) {
        if (until < now) firing.delete(soma)
        else field.somas[soma].cells.forEach((c) => stamp(c, FIRING[frame[c.y][c.x]] ?? frame[c.y][c.x]))
      }

      setLive(gridToText(frame))
    }

    const id = setInterval(tick, TICK_MS)
    return () => clearInterval(id)
  }, [field, tissue.done])

  return (
    <figure className={cn('flex min-h-0 flex-col items-center', className)}>
      <div ref={boxRef} className="flex min-h-0 w-full flex-1 items-center justify-center overflow-hidden">
        <div aria-hidden className="relative select-none" style={{ fontSize, lineHeight: LINE }}>
          <pre className={cn('text-ink transition-opacity duration-[1500ms]', start ? 'opacity-100' : 'opacity-0')}>
            {dustText}
          </pre>
          <pre className="absolute inset-0 text-ink">{tissue.done && live ? live : tissue.text}</pre>
        </div>
      </div>
      <figcaption
        className={cn(
          'mt-3 flex w-full max-w-2xl items-center justify-between gap-4 px-2 font-mono text-[10px] tracking-[0.14em] text-ink uppercase transition-opacity duration-700',
          tissue.done ? 'opacity-100' : 'opacity-0',
        )}
      >
        <span>
          Fig. 01 — cortical microcircuit · {field.somas.length} neurons · {field.edges.length} synapses
        </span>
        <span className="hidden items-center gap-2 sm:flex">
          <span className="inline-block h-px w-8 bg-ink" /> 50 <span className="normal-case">µm</span>
        </span>
      </figcaption>
    </figure>
  )
}
