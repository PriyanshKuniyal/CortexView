import { Moon, Plus, Sun } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'
import { ChatInput } from '@/components/ChatInput'
import { LoadingScreen } from '@/components/LoadingScreen'
import { Mascot } from '@/components/Mascot'
import { Materialize } from '@/components/Materialize'
import { NeuronCluster } from '@/components/NeuronCluster'
import { Sidebar } from '@/components/Sidebar'
import { Window } from '@/components/Window'
import { useBoot } from '@/hooks/useBoot'
import { useChat } from '@/hooks/useChat'
import { useScramble } from '@/hooks/useScramble'
import { useTheme } from '@/hooks/useTheme'
import { cn } from '@/lib/utils'

type WinId = 'prompt' | 'output'
type WinState = 'open' | 'minimized' | 'closed'

const WORDMARK = 'HACKING THE BANANAS'
const MIN_PROMPT_H = 64

export default function App() {
  const boot = useBoot()
  const booted = boot !== 'loading'
  const chat = useChat()
  const { theme, toggle: toggleTheme } = useTheme()
  const messages = chat.active.messages
  const hasOutput = messages.length > 0

  const [wins, setWins] = useState<Record<WinId, WinState>>({ prompt: 'open', output: 'open' })
  const [maximized, setMaximized] = useState<WinId | null>(null)
  const [promptHeight, setPromptHeight] = useState<number | null>(null) // null = compact

  const wordmark = useScramble(WORDMARK, { start: booted, delay: 100, duration: 1100 })
  const mainRef = useRef<HTMLDivElement>(null)
  const outputEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    outputEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setMaximized(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Each session starts with both windows restored
  const [sessionId, setSessionId] = useState(chat.active.id)
  if (sessionId !== chat.active.id) {
    setSessionId(chat.active.id)
    setWins({ prompt: 'open', output: 'open' })
    setMaximized(null)
  }

  const setWin = (id: WinId, state: WinState) => {
    setWins((w) => ({ ...w, [id]: state }))
    if (state !== 'open' && maximized === id) setMaximized(null)
  }

  const controls = (id: WinId) => ({
    maximized: maximized === id,
    controlsEnabled: hasOutput,
    onClose: () => setWin(id, 'closed'),
    onMinimize: () => setWin(id, 'minimized'),
    onToggleMaximize: () => setMaximized((m) => (m === id ? null : id)),
    className: 'flex-1',
  })

  const startResize = (e: React.PointerEvent) => {
    e.preventDefault()
    const rect = mainRef.current!.getBoundingClientRect()
    const onMove = (ev: PointerEvent) => {
      const h = rect.bottom - ev.clientY
      setPromptHeight(Math.min(rect.height * 0.6, Math.max(MIN_PROMPT_H, h)))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
    }
    document.body.style.cursor = 'row-resize'
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  const jumpToPrompt = (messageId: string) => {
    if (wins.output !== 'open') setWin('output', 'open')
    requestAnimationFrame(() =>
      document.getElementById(`msg-${messageId}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
    )
  }

  // Before the first prompt only the prompt window exists and it can't be hidden
  const promptOpen = !hasOutput || wins.prompt === 'open'
  const outputOpen = hasOutput && wins.output === 'open'
  const hidden = hasOutput ? (['output', 'prompt'] as WinId[]).filter((id) => wins[id] !== 'open') : []

  return (
    <>
      {boot !== 'ready' && <LoadingScreen leaving={boot === 'leaving'} />}

      <div className="flex h-full gap-4 p-4">
        <Materialize start={booted} className="hidden w-64 shrink-0 md:block">
          <div className="h-full overflow-hidden rounded-2xl border border-ink bg-paper">
            <Sidebar
              conversations={chat.conversations}
              activeId={chat.active.id}
              disabled={chat.isStreaming}
              onSelect={chat.selectConversation}
              onCreate={chat.createConversation}
              onDelete={chat.deleteConversation}
              onJumpToPrompt={jumpToPrompt}
            />
          </div>
        </Materialize>

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex items-start justify-between gap-4 px-1 pt-1 pb-3">
            <div className="flex min-w-0 items-center gap-3">
              <span className="relative flex size-2 shrink-0">
                <span className="absolute inset-0 animate-ping rounded-full bg-ink" />
                <span className="relative size-2 rounded-full bg-ink" />
              </span>
              <div className="min-w-0">
                <h1
                  aria-label={WORDMARK}
                  className="truncate font-mono text-[11px] font-medium tracking-[0.32em] whitespace-pre"
                >
                  {wordmark.text}
                </h1>
                <p className="font-mono text-[9px] tracking-[0.24em] text-ink uppercase">Neural interface · v0.1</p>
              </div>
            </div>

            <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-2">
              {hidden.map((id) => (
                <button
                  key={id}
                  onClick={() => setWin(id, 'open')}
                  className="rounded-full border border-ink bg-paper px-3 py-1 font-mono text-[10px] tracking-[0.14em] text-ink uppercase transition-colors hover:bg-ink hover:text-paper"
                >
                  ↑ {id} · {wins[id]}
                </button>
              ))}
              <span className="hidden items-center gap-2 rounded-full border border-ink bg-paper px-3 py-1 font-mono text-[10px] tracking-[0.14em] text-ink uppercase sm:flex">
                <span className={cn('size-1.5 rounded-full', chat.isStreaming ? 'animate-pulse bg-ink' : 'bg-ink')} />
                {chat.isStreaming ? 'Synthesizing' : 'Idle'}
              </span>
              <button
                onClick={chat.createConversation}
                disabled={chat.isStreaming}
                aria-label="New session"
                className="flex size-7 items-center justify-center rounded-full border border-ink bg-paper text-ink hover:bg-ink hover:text-paper disabled:border-dashed md:hidden"
              >
                <Plus size={14} />
              </button>
            </div>
            <button
              onClick={toggleTheme}
              role="switch"
              aria-checked={theme === 'dark'}
              aria-label="Dark theme"
              className="flex items-center gap-2 rounded-full border border-ink bg-paper py-1 pr-3 pl-1 font-mono text-[10px] tracking-[0.14em] text-ink uppercase transition-colors hover:bg-ink hover:text-paper"
            >
              <span className="relative flex h-3.5 w-6 items-center rounded-full border border-current">
                <span
                  className={cn(
                    'absolute size-2 rounded-full bg-current transition-transform duration-200',
                    theme === 'dark' ? 'translate-x-3' : 'translate-x-0.5',
                  )}
                />
              </span>
              {theme === 'dark' ? <Moon size={11} /> : <Sun size={11} />}
              {theme === 'dark' ? 'Dark' : 'Light'}
            </button>
            </div>
          </header>

          <main ref={mainRef} className="flex min-h-0 flex-1 flex-col">
            <section className="flex min-h-0 flex-1 flex-col">
              {outputOpen ? (
                <Materialize start className="mx-auto flex min-h-0 w-full max-w-4xl flex-1">
                  <Window {...controls('output')} title={chat.active.title}>
                    <div className="min-h-0 flex-1 overflow-y-auto">
                      <div className="mx-auto max-w-2xl space-y-6 px-6 py-8">
                        {messages.map((m, i) =>
                          m.role === 'user' ? (
                            <div
                              key={m.id}
                              id={`msg-${m.id}`}
                              className="flex scroll-mt-6 gap-3 border-t border-ink pt-6 font-mono text-[13px] text-ink first:border-0 first:pt-0"
                            >
                              <span className="text-ink">›</span>
                              <p className="whitespace-pre-wrap">{m.content}</p>
                            </div>
                          ) : (
                            <div key={m.id} className={cn('prose-chat', m.error && 'italic')}>
                              {m.content ? (
                                <Markdown>{m.content}</Markdown>
                              ) : (
                                chat.isStreaming &&
                                i === messages.length - 1 && (
                                  <span className="flex items-center gap-2 font-mono text-[11px] tracking-[0.14em] text-ink uppercase">
                                    <span className="size-1.5 animate-pulse rounded-full bg-ink" /> Synthesizing
                                  </span>
                                )
                              )}
                            </div>
                          ),
                        )}
                        <div ref={outputEndRef} />
                      </div>
                    </div>
                  </Window>
                </Materialize>
              ) : (
                <NeuronCluster start={booted} active={chat.isStreaming} className="flex-1 py-2" />
              )}
            </section>

            {outputOpen && promptOpen ? (
              <div
                onPointerDown={startResize}
                onDoubleClick={() => setPromptHeight(null)}
                role="separator"
                aria-orientation="horizontal"
                title="Drag to resize · double-click to reset"
                className="group mx-auto flex h-20 w-full max-w-3xl shrink-0 cursor-row-resize items-start justify-center pt-3"
              >
                <div className="h-1 w-10 rounded-full bg-ink transition-all group-hover:w-20 group-hover:bg-ink" />
              </div>
            ) : (
              // room for the mascot perched on the prompt window
              <div className="h-20 shrink-0" />
            )}

            {promptOpen && (
              <Materialize
                start={booted}
                delay={250}
                style={promptHeight && hasOutput ? { height: promptHeight } : undefined}
                className="mx-auto w-full max-w-3xl shrink-0"
              >
                <Mascot className="absolute right-7 bottom-full -mb-px" />
                <Window {...controls('prompt')}>
                  <ChatInput isStreaming={chat.isStreaming} onSend={chat.send} onStop={chat.stop} />
                </Window>
              </Materialize>
            )}

            <p className="pt-2.5 text-center font-mono text-[9px] tracking-[0.2em] text-ink uppercase">
              Research preview · outputs require clinical verification
            </p>
          </main>
        </div>
      </div>
    </>
  )
}
