import './cortexview.css'
import { useEffect, useState } from 'react'
import { DEMO_SOURCE } from './config'
import type { DemoData } from './demo/types'
import { buildSyntheticDemo } from './demo/synthetic'
import { loadRemindDemo } from './demo/remind'
import { useScrollFilm } from './film/useScrollFilm'
import { Hud } from './film/Hud'
import { Background } from './film/Background'
import { TitleScene } from './scenes/TitleScene'
import { SlicesScene } from './scenes/SlicesScene'
import { BeforeAfterScene } from './scenes/BeforeAfterScene'
import { PeelScene } from './scenes/PeelScene'
import { NumbersScene } from './scenes/NumbersScene'
import { GuardrailsScene } from './scenes/GuardrailsScene'
import { FineTuneScene } from './scenes/FineTuneScene'
import { ApplicationsScene } from './scenes/ApplicationsScene'
import { TryScene } from './scenes/TryScene'

/** Scene lengths in viewport heights, and their names for the counter and scrubber. */
const LEN = [1.2, 1.7, 1.8, 5, 1.4, 1.4, 2.2, 1.8, 1] as const
const NAMES = ['Title', 'The problem', 'Before / after', 'Peel', 'Numbers', 'Guardrails', 'Fine-tuned AI', 'Applications', 'Try it'] as const

function useDemoData(): DemoData | null {
  const [demo, setDemo] = useState<DemoData | null>(() => (DEMO_SOURCE === 'synthetic' ? buildSyntheticDemo() : null))
  useEffect(() => {
    if (DEMO_SOURCE !== 'remind') return
    let live = true
    loadRemindDemo()
      .then((d) => live && setDemo(d))
      .catch((e) => {
        console.error(e, '\nFalling back to the synthetic phantom.')
        if (live) setDemo(buildSyntheticDemo())
      })
    return () => {
      live = false
    }
  }, [])
  return demo
}

export function CortexView() {
  const film = useScrollFilm(LEN)
  const demo = useDemoData()

  return (
    <>
      <Background film={film} />
      <div id="track" ref={film.bindTrack} />
      <Hud film={film} names={NAMES} />
      <main>
        {demo && (
          <>
            <TitleScene film={film} demo={demo} />
            <SlicesScene film={film} demo={demo} />
            <BeforeAfterScene film={film} demo={demo} />
            <PeelScene film={film} demo={demo} />
            <NumbersScene film={film} demo={demo} />
            <GuardrailsScene film={film} demo={demo} />
            <FineTuneScene film={film} demo={demo} />
            <ApplicationsScene film={film} demo={demo} />
          </>
        )}
        <TryScene film={film} />
      </main>
    </>
  )
}
