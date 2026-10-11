import { useEffect, useState } from 'react'
import { MASCOT } from '@/lib/ascii'
import { cn } from '@/lib/utils'

const ACCENT = new Set(['@', '+'])

export function Mascot({ className }: { className?: string }) {
  const [blink, setBlink] = useState(false)

  useEffect(() => {
    let reopen: ReturnType<typeof setTimeout>
    const id = setInterval(() => {
      setBlink(true)
      reopen = setTimeout(() => setBlink(false), 160)
    }, 4200)
    return () => {
      clearInterval(id)
      clearTimeout(reopen)
    }
  }, [])

  return (
    <pre aria-hidden className={cn('select-none text-[9px] leading-[1.15] text-ink', className)}>
      {MASCOT.map((line, r) => (
        <span key={r} className="block">
          {[...(blink && r === 2 ? line.replace(/o/g, '-') : line)].map((ch, i) =>
            ACCENT.has(ch) ? (
              <span key={i} className="font-semibold text-ink">
                {ch}
              </span>
            ) : (
              ch
            ),
          )}
        </span>
      ))}
    </pre>
  )
}
