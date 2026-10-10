import { useEffect, useState } from 'react'
import { FADE_MS, LOAD_MS } from '@/hooks/useBoot'
import { cn } from '@/lib/utils'

const SPINNER = ['|', '/', '-', '\\']
const BAR_WIDTH = 28
const BOX_INNER = BAR_WIDTH + 14
const STEPS = [
  { at: 0.1, text: 'calibrating instruments' },
  { at: 0.3, text: 'staining neural tissue' },
  { at: 0.5, text: 'mapping synapses' },
  { at: 0.7, text: 'priming inference core' },
  { at: 0.9, text: 'assembling interface' },
]

export function LoadingScreen({ leaving }: { leaving: boolean }) {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const t0 = performance.now()
    const id = setInterval(() => setElapsed(performance.now() - t0), 50)
    return () => clearInterval(id)
  }, [])

  const progress = Math.min(1, elapsed / LOAD_MS)
  const filled = Math.round(progress * BAR_WIDTH)
  const spinner = SPINNER[Math.floor(elapsed / 100) % SPINNER.length]
  const pct = `${String(Math.round(progress * 100)).padStart(3, ' ')}%`
  const edge = '-'.repeat(BOX_INNER)
  // "|  " + content + "  |", padded so the right border lines up
  const pad = (used: number) => ' '.repeat(BOX_INNER - 4 - used)

  return (
    <div
      style={{ transitionDuration: `${FADE_MS}ms` }}
      className={cn(
        'fixed inset-0 z-[100] flex items-center justify-center bg-paper transition-opacity',
        leaving && 'pointer-events-none opacity-0',
      )}
    >
      <pre className="text-xs leading-relaxed text-ink">
        <span className="text-ink">{`.${edge}.`}</span>
        {'\n'}
        <span className="text-ink">|</span>
        {'  '}
        <span className="text-ink">{spinner}</span>
        <span className="text-ink"> initializing</span>
        {pad(14)}
        {'  '}
        <span className="text-ink">|</span>
        {'\n'}
        <span className="text-ink">|</span>
        {' '.repeat(BOX_INNER)}
        <span className="text-ink">|</span>
        {'\n'}
        <span className="text-ink">|</span>
        {'  ['}
        <span className="text-ink">{'#'.repeat(filled)}</span>
        <span className="text-ink">{'.'.repeat(BAR_WIDTH - filled)}</span>
        {']  '}
        <span className="text-ink tabular-nums">{pct}</span>
        {pad(BAR_WIDTH + 2 + 2 + pct.length)}
        {'  '}
        <span className="text-ink">|</span>
        {'\n'}
        <span className="text-ink">{`'${edge}'`}</span>
        {'\n\n'}
        {STEPS.map((s) => (
          <span key={s.text} className={cn('block transition-opacity duration-300', progress >= s.at ? 'opacity-100' : 'opacity-0')}>
            {`  > ${s.text.padEnd(30, '.')} `}
            <span className="text-ink">ok</span>
          </span>
        ))}
      </pre>
    </div>
  )
}
