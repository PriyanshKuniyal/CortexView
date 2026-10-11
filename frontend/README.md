# Plie AI - Frontend

The main page is **CortexView** (`src/cortexview/`), a React port of `docs/design/CortexView.html`:
a 7-scene scroll film, with scene 7 ("Try it") wired to `POST /api/analyze` and `GET /api/render/{id}`
(contract C3/C4 in `docs/contracts.md`; types in `src/cortexview/api/types.ts`). Layers come from `shared/layers.json`.

- "Try it" always talks to the real backend (`backend/api/app.py`). Start it before `npm run dev`.
- `DEMO_SOURCE` in `src/cortexview/config.ts` picks the film's demo slice: `'synthetic'` (phantom, default)
  or `'remind'` (a bundle in `public/demo/`, format documented in `src/cortexview/demo/remind.ts`).

The earlier chat UI (React + TypeScript + Vite + Tailwind CSS v4) is still available at `/#chat`.

## Run

    # backend, from the repo root (first time: see backend/requirements.txt)
    backend/.venv/Scripts/python.exe -m uvicorn api.app:app --app-dir backend --port 8000

    # frontend
    npm install
    cp .env.example .env.local
    npm run dev                  # http://localhost:5173, /api proxied to the backend

## Backend contract

The dev server proxies `/api/*` to `VITE_API_TARGET` (default `http://localhost:8000`).
"Try it" calls `POST /api/analyze` and polls `GET /api/render/{job_id}` (contract C3, `docs/contracts.md`;
schema at `http://localhost:8000/api/docs`). The old chat UI at `/#chat` calls `POST /api/chat`, which the
backend doesn't implement.

## Vercel frontend + laptop backend

Vite's development proxy does not exist after a Vercel deployment. Expose the laptop's FastAPI server through a
stable HTTPS tunnel, configure `VITE_API_BASE_URL` in Vercel to that public URL, and redeploy. The browser then sends
uploads directly to the tunnel, avoiding Vercel proxy body-size limits. Add the exact Vercel origin to the laptop
backend's `CORS_ORIGINS`, for example `CORS_ORIGINS=https://your-project.vercel.app`.

## Structure

    src/
      components/   Sidebar, ChatMessage, ChatInput, EmptyState
      hooks/        useChat - conversation state, streaming, stop
      lib/          api client (real + mock), utils
      types/        shared types
