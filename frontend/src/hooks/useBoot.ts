import { useEffect, useState } from 'react'

export type BootPhase = 'loading' | 'leaving' | 'ready'

export const LOAD_MS = 3000
export const FADE_MS = 600

/** loading (splash visible) -> leaving (splash fading, UI assembling) -> ready */
export function useBoot() {
  const [phase, setPhase] = useState<BootPhase>('loading')

  useEffect(() => {
    const a = setTimeout(() => setPhase('leaving'), LOAD_MS)
    const b = setTimeout(() => setPhase('ready'), LOAD_MS + FADE_MS)
    return () => {
      clearTimeout(a)
      clearTimeout(b)
    }
  }, [])

  return phase
}
