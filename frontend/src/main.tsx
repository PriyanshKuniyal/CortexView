import { StrictMode, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'

const root = createRoot(document.getElementById('root')!)
const mount = (node: ReactNode) => root.render(<StrictMode>{node}</StrictMode>)

// CortexView is the main page. The earlier chat UI stays reachable at #chat; each page loads
// only its own styles (the chat UI's Tailwind base styles would change the CortexView layout).
if (location.hash === '#chat') {
  Promise.all([import('./App.tsx'), import('./index.css')]).then(([{ default: App }]) => {
    document.title = 'Hacking the Bananas'
    mount(<App />)
  })
} else {
  import('./cortexview/CortexView').then(({ CortexView }) => mount(<CortexView />))
}
