import { useCallback, useRef, useState } from 'react'
import { streamChat } from '@/lib/api'
import { uid } from '@/lib/utils'
import type { Conversation, Message } from '@/types/chat'

const newConversation = (): Conversation => ({
  id: uid(),
  title: 'New chat',
  messages: [],
  updatedAt: Date.now(),
})

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>(() => [newConversation()])
  const [activeId, setActiveId] = useState(() => conversations[0].id)
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0]

  const updateConversation = useCallback(
    (id: string, fn: (c: Conversation) => Conversation) =>
      setConversations((prev) => prev.map((c) => (c.id === id ? fn(c) : c))),
    [],
  )

  const send = useCallback(
    async (text: string) => {
      const content = text.trim()
      if (!content || isStreaming) return

      const convId = active.id
      const userMsg: Message = { id: uid(), role: 'user', content, createdAt: Date.now() }
      const assistantMsg: Message = { id: uid(), role: 'assistant', content: '', createdAt: Date.now() }
      const history = [...active.messages, userMsg].map(({ role, content }) => ({ role, content }))

      updateConversation(convId, (c) => ({
        ...c,
        title: c.messages.length === 0 ? content.slice(0, 40) : c.title,
        messages: [...c.messages, userMsg, assistantMsg],
        updatedAt: Date.now(),
      }))

      const patchAssistant = (fn: (m: Message) => Message) =>
        updateConversation(convId, (c) => ({
          ...c,
          messages: c.messages.map((m) => (m.id === assistantMsg.id ? fn(m) : m)),
        }))

      const controller = new AbortController()
      abortRef.current = controller
      setIsStreaming(true)
      try {
        await streamChat(
          history,
          (chunk) => patchAssistant((m) => ({ ...m, content: m.content + chunk })),
          controller.signal,
        )
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          patchAssistant((m) => ({
            ...m,
            error: true,
            content: m.content || `⚠️ ${(err as Error).message}`,
          }))
        }
      } finally {
        setIsStreaming(false)
        abortRef.current = null
      }
    },
    [active, isStreaming, updateConversation],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])

  const createConversation = useCallback(() => {
    const conv = newConversation()
    setConversations((prev) => [conv, ...prev])
    setActiveId(conv.id)
  }, [])

  const deleteConversation = useCallback(
    (id: string) => {
      const next = conversations.filter((c) => c.id !== id)
      if (next.length === 0) next.push(newConversation())
      setConversations(next)
      if (id === activeId) setActiveId(next[0].id)
    },
    [conversations, activeId],
  )

  return {
    conversations,
    active,
    isStreaming,
    send,
    stop,
    selectConversation: setActiveId,
    createConversation,
    deleteConversation,
  }
}
