import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { useScramble } from '@/hooks/useScramble'
import { cn } from '@/lib/utils'

const FONT = '12px/14px var(--font-mono)'
const LINE_H = 14

function measureCharWidth() {
  const probe = document.createElement('span')
  probe.style.cssText = `position:absolute;visibility:hidden;white-space:pre;font:${FONT}`
  probe.textContent = '0'.repeat(100)
  document.body.appendChild(probe)
  const width = probe.getBoundingClientRect().width / 100
  probe.remove()
  return width || 7.2
}

/** A rounded ASCII box (.---. / '---') with a sparse dotted fill. */
function drawBox(cols: number, rows: number) {
  if (cols < 2 || rows < 2) return ''
  const lines: string[] = []
  for (let r = 0; r < rows; r++) {
    if (r === 0) lines.push('.' + '-'.repeat(cols - 2) + '.')
    else if (r === rows - 1) lines.push("'" + '-'.repeat(cols - 2) + "'")
    else {
      let inner = ''
      for (let c = 0; c < cols - 2; c++) inner += Math.random() < 0.04 ? '.' : ' '
      lines.push('|' + inner + '|')
    }
  }
  return lines.join('\n')
}

interface MaterializeProps {
  start: boolean
  delay?: number
  className?: string
  style?: React.CSSProperties
  children: ReactNode
}

/**
 * Draws an ASCII outline of its area, decodes it in, then cross-fades into
 * the real content.
 */
export function Materialize({ start, delay = 0, className, style, children }: MaterializeProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState('')

  useLayoutEffect(() => {
    if (!start || !ref.current) return
    const { width, height } = ref.current.getBoundingClientRect()
    setBox(drawBox(Math.floor(width / measureCharWidth()), Math.floor(height / LINE_H)))
  }, [start])

  const { text, done } = useScramble(box, { start: start && box !== '', delay, duration: 800 })

  return (
    <div ref={ref} style={style} className={cn('relative', className)}>
      <div
        className={cn(
          'flex h-full w-full flex-col transition-opacity duration-500',
          done ? 'opacity-100' : 'pointer-events-none opacity-0',
        )}
      >
        {children}
      </div>
      <pre
        aria-hidden
        style={{ font: FONT }}
        className={cn(
          'pointer-events-none absolute inset-0 overflow-hidden whitespace-pre text-ink transition-opacity duration-500',
          done ? 'opacity-0' : 'opacity-100',
        )}
      >
        {text}
      </pre>
    </div>
  )
}
