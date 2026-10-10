import { Plus, X } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Conversation } from '@/types/chat'

interface SidebarProps {
  conversations: Conversation[]
  activeId: string
  disabled: boolean
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: (id: string) => void
  onJumpToPrompt: (messageId: string) => void
}

export function Sidebar({
  conversations,
  activeId,
  disabled,
  onSelect,
  onCreate,
  onDelete,
  onJumpToPrompt,
}: SidebarProps) {
  const sessions = conversations.filter((c) => c.messages.length > 0)

  return (
    <aside className="flex h-full flex-col">
      <div className="p-3">
        <button
          onClick={onCreate}
          disabled={disabled}
          className="group flex w-full items-center gap-2.5 rounded-xl border border-ink bg-paper px-3 py-2 text-[13px] font-medium transition-colors enabled:hover:bg-ink enabled:hover:text-paper disabled:cursor-not-allowed disabled:border-dashed"
        >
          <span className="flex size-5 items-center justify-center rounded-md bg-paper text-ink">
            <Plus size={13} strokeWidth={2.5} />
          </span>
          New session
        </button>
      </div>

      <div className="flex items-center justify-between px-4 pt-2 pb-2 font-mono text-[10px] tracking-[0.18em] text-ink uppercase">
        <span>Previous prompts</span>
        <span className="tabular-nums">{String(sessions.length).padStart(2, '0')}</span>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {sessions.length === 0 && (
          <div className="mx-2 mt-1 rounded-xl border border-dashed border-ink px-3 py-4 text-center">
            <pre aria-hidden className="text-[10px] leading-tight text-ink">{'o--·--·--o'}</pre>
            <p className="mt-2 text-xs text-ink">No sessions recorded yet.</p>
          </div>
        )}

        {sessions.map((c) => {
          const active = c.id === activeId
          const prompts = c.messages.filter((m) => m.role === 'user')

          return (
            <div key={c.id} className="mb-1">
              <div
                className={cn(
                  'group relative flex items-center gap-2 rounded-lg px-3 py-2 text-[13px] transition-colors',
                  active ? 'bg-ink text-paper' : 'text-ink hover:bg-ink hover:text-paper',
                )}
              >
                <button
                  onClick={() => onSelect(c.id)}
                  disabled={disabled}
                  className="min-w-0 flex-1 truncate text-left"
                >
                  {c.title}
                </button>
                <button
                  onClick={() => onDelete(c.id)}
                  disabled={disabled}
                  aria-label="Delete session"
                  className="text-current opacity-0 transition-opacity group-hover:opacity-100"
                >
                  <X size={13} />
                </button>
              </div>

              {active && (
                <ol className="mt-1 mb-2 ml-4 border-l border-ink pl-2">
                  {prompts.map((p, i) => (
                    <li key={p.id}>
                      <button
                        onClick={() => onJumpToPrompt(p.id)}
                        className="flex w-full items-baseline gap-2 rounded-md px-2 py-1 text-left text-xs text-ink transition-colors hover:bg-ink hover:text-paper"
                      >
                        <span className="font-mono text-[10px] text-current tabular-nums">
                          {String(i + 1).padStart(2, '0')}
                        </span>
                        <span className="truncate">{p.content}</span>
                      </button>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )
        })}
      </nav>

      <div className="flex items-center gap-2 border-t border-ink px-4 py-3 font-mono text-[10px] tracking-[0.14em] text-ink uppercase">
        <span className="size-1.5 rounded-full bg-ink" />
        Research preview
      </div>
    </aside>
  )
}
