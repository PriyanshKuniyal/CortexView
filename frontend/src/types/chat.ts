export type Role = 'user' | 'assistant' | 'system'

export interface Message {
  id: string
  role: Role
  content: string
  createdAt: number
  error?: boolean
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  updatedAt: number
}
