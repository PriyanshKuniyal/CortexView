import { ArrowUp, Square } from 'lucide-react'
import { useState, type KeyboardEvent } from 'react'

interface ChatInputProps {
  isStreaming: boolean
  onSend: (text: string) => void
  onStop: () => void
}

export function ChatInput({ isStreaming, onSend, onStop }: ChatInputProps) {
  const [value, setValue] = useState('')

  const submit = () => {
    if (!value.trim() || isStreaming) return
    onSend(value)
    setValue('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex h-full items-end gap-2 px-3.5 pt-0.5 pb-2.5">
      <textarea
        autoFocus
        aria-label="Prompt"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        className="field-sizing-content max-h-40 min-h-8 flex-1 resize-none self-stretch bg-transparent py-1.5 text-[15px] leading-5 outline-none"
      />
      {isStreaming ? (
        <button
          onClick={onStop}
          aria-label="Stop"
          className="flex size-8 shrink-0 items-center justify-center rounded-full border border-ink text-ink transition-colors hover:bg-ink hover:text-paper"
        >
          <Square size={10} fill="currentColor" />
        </button>
      ) : (
        <button
          onClick={submit}
          disabled={!value.trim()}
          aria-label="Send"
          className="flex size-8 shrink-0 items-center justify-center rounded-full bg-ink text-paper border border-ink transition-colors enabled:hover:bg-paper enabled:hover:text-ink disabled:bg-paper disabled:text-ink"
        >
          <ArrowUp size={16} strokeWidth={2.25} />
        </button>
      )}
    </div>
  )
}
