# Plie AI - Frontend

The main page is **CortexView** (`src/cortexview/`), a React port of `docs/design/CortexView.html`:
a 7-scene scroll film, with scene 7 ("Try it") wired to `POST /api/analyze` and `GET /api/render/{id}`
(contract C3/C4 in `docs/contracts.md`; types in `src/cortexview/api/types.ts`). Layers come from `shared/layers.json`.

- `VITE_USE_MOCK=true` answers "Try it" with an in-browser mock server (built-in phantom). File-name test hooks:
  a non-`.dcm`/`.zip` file returns `not_dicom`; a name containing `render-fail` makes the render fail.
- `DEMO_SOURCE` in `src/cortexview/config.ts` picks the film's demo slice: `'synthetic'` (phantom, default)
  or `'remind'` (a bundle in `public/demo/`, format documented in `src/cortexview/demo/remind.ts`).

The earlier chat UI (React + TypeScript + Vite + Tailwind CSS v4) is still available at `/#chat`.

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
