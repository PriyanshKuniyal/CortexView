import type { ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface WindowProps {
  title?: string
  maximized: boolean
  controlsEnabled?: boolean
  onClose: () => void
  onMinimize: () => void
  onToggleMaximize: () => void
  children: ReactNode
  className?: string
}

export function Window({
  title,
  maximized,
  controlsEnabled = true,
  onClose,
  onMinimize,
  onToggleMaximize,
  children,
  className,
}: WindowProps) {
  return (
    <div
      className={cn(
        'flex min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border border-ink bg-paper',
        maximized && 'fixed inset-4 z-50 shadow-[0_0_0_9999px_var(--color-paper)]',
        className,
      )}
    >
      <div
        onDoubleClick={controlsEnabled ? onToggleMaximize : undefined}
        className={cn(
          'relative flex shrink-0 select-none items-center px-3.5',
          title ? 'h-9 border-b border-ink' : 'h-7',
        )}
      >
        <div className={cn('group relative z-10 flex gap-1.5')}>
          <Light label="Close" symbol="×" hover="enabled:hover:bg-ink" disabled={!controlsEnabled} onClick={onClose} />
          <Light label="Minimize" symbol="−" hover="enabled:hover:bg-ink" disabled={!controlsEnabled} onClick={onMinimize} />
          <Light
            label={maximized ? 'Exit full screen' : 'Full screen'}
            symbol={maximized ? '↙' : '↗'}
            hover="enabled:hover:bg-ink"
            disabled={!controlsEnabled}
            onClick={onToggleMaximize}
          />
        </div>
        {title && (
          <span className="pointer-events-none absolute inset-x-0 text-center font-mono text-[10px] tracking-[0.18em] text-ink uppercase">
            {title}
          </span>
        )}
      </div>
      <div className="flex min-h-0 flex-1 flex-col">{children}</div>
    </div>
  )
}

interface LightProps {
  label: string
  symbol: string
  hover: string
  disabled: boolean
  onClick: () => void
}

function Light({ label, symbol, hover, disabled, onClick }: LightProps) {
  return (
    <button
      onClick={onClick}
      onDoubleClick={(e) => e.stopPropagation()}
      disabled={disabled}
      aria-label={label}
      title={disabled ? 'Send a prompt to enable window controls' : label}
      className={cn(
        'flex size-2.5 items-center justify-center rounded-full border border-ink bg-ink disabled:bg-paper text-[7px] leading-none font-bold text-paper transition-colors disabled:cursor-not-allowed',
        hover,
      )}
    >
      <span className="opacity-0 group-hover:opacity-100">{!disabled && symbol}</span>
    </button>
  )
}
