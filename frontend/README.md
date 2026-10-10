# Plie AI - Frontend

React + TypeScript + Vite + Tailwind CSS v4 chat UI for the AI system.

## Run

    npm install
    cp .env.example .env.local   # VITE_USE_MOCK=true works without a backend
    npm run dev                  # http://localhost:5173

## Backend contract

The dev server proxies `/api/*` to `VITE_API_TARGET` (default `http://localhost:8000`).
The UI calls `POST /api/chat` with `{ "messages": [{ "role": "user" | "assistant", "content": "..." }] }`
and expects the reply streamed back as plain text. See `src/lib/api.ts`.

## Structure

    src/
      components/   Sidebar, ChatMessage, ChatInput, EmptyState
      hooks/        useChat - conversation state, streaming, stop
      lib/          api client (real + mock), utils
      types/        shared types
