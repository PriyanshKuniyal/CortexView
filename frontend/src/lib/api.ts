import type { Message } from '@/types/chat'

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

/**
 * Streams the assistant reply for a conversation.
 *
 * Backend contract: POST /api/chat with `{ messages: [{ role, content }] }`,
 * responding with a plain-text stream of the reply.
 */
export async function streamChat(
  messages: Pick<Message, 'role' | 'content'>[],
  onChunk: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  if (USE_MOCK) return mockStream(messages, onChunk, signal)

  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new Error(`Request failed: ${res.status} ${res.statusText}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    onChunk(decoder.decode(value, { stream: true }))
  }
}

async function mockStream(
  messages: Pick<Message, 'role' | 'content'>[],
  onChunk: (text: string) => void,
  signal?: AbortSignal,
) {
  const last = messages.at(-1)?.content ?? ''
  const reply = `This is a **mock response** (set \`VITE_USE_MOCK=false\` to use the real backend).\n\nYou said:\n\n> ${last}`
  for (const word of reply.split(/(\s+)/)) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    await new Promise((r) => setTimeout(r, 25))
    onChunk(word)
  }
}
