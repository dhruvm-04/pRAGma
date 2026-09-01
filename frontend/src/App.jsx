import { useState, useRef } from 'react'

/*
 * RAG - one component, three jobs:
 *   1. Upload a PDF  -> POST /ingest  (backend chunks + embeds it)
 *   2. Ask a question -> POST /query  (backend retrieves + generates)
 *   3. Show the answer WITH the retrieved chunks and translation steps
 *      so you can see the "grounding" (what it actually read) in action.
 */

// Query translation methods
const TRANSLATION_METHODS = [
  { value: 'none', label: 'Direct', detail: 'Retrieve from the question as written.', stages: ['Question', 'Retrieve', 'Generate'] },
  { value: 'multi_query', label: 'Multi Query', detail: 'Create alternatives, then merge their evidence.', stages: ['Rewrite', 'Retrieve', 'Merge', 'Generate'] },
  { value: 'rag_fusion', label: 'RAG Fusion', detail: 'Retrieve per variation and rank shared evidence.', stages: ['Rewrite', 'Retrieve', 'Rank', 'Generate'] },
  { value: 'decomposition', label: 'Decomposition', detail: 'Break a complex question into smaller steps.', stages: ['Decompose', 'Retrieve', 'Synthesize'] },
  { value: 'step_back', label: 'Step Back', detail: 'Retrieve broad context before the specific question.', stages: ['Broaden', 'Retrieve', 'Generate'] },
  { value: 'hyde', label: 'HyDE', detail: 'Retrieve using a hypothetical answer passage.', stages: ['Hypothesize', 'Retrieve', 'Generate'] },
]

export default function App() {
  const [source, setSource] = useState(null)
  const [ingestMsg, setIngestMsg] = useState('')
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(null)
  const [selectedMethod, setSelectedMethod] = useState('none')
  const fileRef = useRef(null)

  //  1. ingest (background job + polling) 
  async function handleFile(e) {
    const file = e.target.files[0]
    if (!file) return
    setIngestMsg('')
    setProgress({ status: 'queued', message: 'uploading…', done: 0, total: 0 })
    const fd = new FormData()
    fd.append('file', file)
    const res = await fetch('/ingest', { method: 'POST', body: fd })
    const data = await res.json()
    if (data.job_id) pollJob(data.job_id)
    else setIngestMsg(data.error || 'ingest failed')
  }

  // Poll /ingest/status/{id} every 400ms until the job finishes.
  function pollJob(jobId) {
    const t0 = Date.now()
    const tick = async () => {
      const res = await fetch(`/ingest/status/${jobId}`)
      const j = await res.json()
      setProgress(j)
      if (j.status === 'done' || j.status === 'error') {
        setSource(j.source)
        setIngestMsg(j.status === 'done'
          ? `${j.done} chunks embedded in ${Date.now() - t0} ms`
          : j.message)
        setProgress(null)
        return
      }
      setTimeout(tick, 1000)
    }
    tick()
  }

  //  2. query 
  async function handleSend() {
    const q = question.trim()
    if (!q || busy) return
    const execution = TRANSLATION_METHODS.find(method => method.value === selectedMethod)
    setMessages(prev => [...prev, { role: 'user', text: q }])
    setQuestion('')
    setBusy(true)
    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, method: selectedMethod }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.answer,
        chunks: data.chunks || [],
        translationSteps: data.translation_steps || [],
        retrievalDetails: data.retrieval_details || [],
        execution,
        latency: data.latency,
        tokens: data.tokens,
      }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Backend unreachable — is uvicorn running?' }])
    }
    setBusy(false)
  }

  //  3. theme toggle 
  function toggleTheme() {
    const root = document.documentElement
    const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'
    root.setAttribute('data-theme', next)
    try { localStorage.setItem('rag-theme', next) } catch (e) {}
  }

  return (
    <>
      {/*  NAV  */}
      <nav className="nav">
        <a className="nav-logo" href="https://dhruvmaheshwari.vercel.app/" target="_blank" rel="noreferrer">DM<span className="dot">.</span></a>
        <ul className="nav-links">
          <li><a className="nav-link" href="#ingest"><span className="num">01</span>Ingest</a></li>
          <li><a className="nav-link" href="#chat"><span className="num">02</span>Ask</a></li>
          <li><a className="nav-link" href="#evidence"><span className="num">03</span>Evidence</a></li>
        </ul>
        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">☾</button>
      </nav>

      {/*  HEADER  */}
      <section id="top">
        <div className="container">
          <div className="section-label">
            <span className="num">00</span><span className="sep">/</span><span className="name">pRAGma</span>
            <span className="line"></span><span className="meta">retrieval-augmented generation</span>
          </div>
          <h1 className="home-name">Upload a PDF.<br /><span className="red">Ask it anything.</span></h1>
          <p className="home-tagline">
            PDF → chunks → embeddings <span className="sep">/</span> top-k retrieval
          </p>

          <div className="info-bento">
            <div className="info-cell"><div className="info-key">Pipeline</div><div className="info-val">Chunk · Embed · Retrieve · Generate</div></div>
            <div className="info-cell"><div className="info-key">Embeddings</div><div className="info-val">bge-small · local · free</div></div>
            <div className="info-cell info-modes">
              <div className="info-key">Execution modes</div>
              <div className="mode-list">
                {TRANSLATION_METHODS.map(method => <span key={method.value}>{method.label}</span>)}
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="container main-grid">
        {/*  INGEST  */}
        <section id="ingest" className="panel">
          <div className="section-label">
            <span className="num">01</span><span className="sep">/</span><span className="name">Ingest a PDF</span>
            <span className="line"></span><span className="meta">upload → chunk → embed</span>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            onChange={handleFile}
            className="file-input"
          />
          <button className="cta-btn primary full" onClick={() => fileRef.current && fileRef.current.click()}>
            Choose PDF
          </button>

          {progress && (
            <>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress.total ? Math.round(progress.done / progress.total * 100) : 0}%` }}></div>
              </div>
              <p className="progress-meta">
                {progress.message}
                {progress.total > 0 && ` · ${progress.done}/${progress.total} chunks (${Math.round(progress.done / progress.total * 100)}%)`}
              </p>
            </>
          )}

          <p className="meta-line">{ingestMsg || 'Upload a text-based PDF — the embedding model downloads on first use (~130MB).'}</p>
        </section>

        {/*  CHAT  */}
        <section id="chat" className="panel">
          <div className="section-label">
            <span className="num">02</span><span className="sep">/</span><span className="name">Ask</span>
            <span className="line"></span><span className="meta">question → top-k → llm</span>
          </div>

          <div className="chat-log" id="evidence">
            {messages.length === 0 && <p className="empty-note">No questions yet — upload a PDF, then ask about its contents.</p>}
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div className="msg-label">{m.role === 'user' ? 'you' : 'rag'}</div>
                <div className="msg-text">{m.text}</div>

                {m.role === 'assistant' && m.execution && (
                  <div className="execution-trace">
                    <div className="execution-overview">
                      <div>
                        <span className="b-label">execution mode</span>
                        <strong>{m.execution.label}</strong>
                        <p>{m.execution.detail}</p>
                      </div>
                      <div className="execution-stages" aria-label={`${m.execution.label} execution steps`}>
                        {m.execution.stages.map((stage, index) => (
                          <span key={stage} className="execution-stage"><i>{String(index + 1).padStart(2, '0')}</i>{stage}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* translation steps */}
                {m.role === 'assistant' && m.translationSteps && m.translationSteps.length > 0 && (
                  <div className="evidence translation-steps">
                    {m.translationSteps.map((step, si) => (
                      <div key={si} className="translation-step">
                        <div className="step-header">
                          <span className="b-label">{step.method}</span>
                          <span className="step-count">{step.queries.length} queries</span>
                        </div>
                        <div className="step-queries">
                          {step.queries.map((q, qi) => (
                            <div key={qi} className="step-query">
                              <span className="query-num">{qi + 1}.</span>
                              <span className="query-text">{q}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* retrieval details */}
                {m.role === 'assistant' && m.retrievalDetails && m.retrievalDetails.length > 0 && (
                  <div className="evidence retrieval-details">
                    <div className="ev-head">
                      <span className="b-label">// retrieval per query</span>
                    </div>
                    <div className="retrieval-grid">
                    {m.retrievalDetails.map((rd, ri) => (
                      <div key={ri} className="retrieval-query">
                        <div className="retrieval-query-head">
                          <span className="query-num">{String(ri + 1).padStart(2, '0')}</span>
                          <div className="retrieval-query-text">{rd.query}</div>
                        </div>
                        <div className="retrieval-hits">
                          {rd.hits.map((h, hi) => (
                            <div key={hi} className="retrieval-hit">
                              <span className="tag">chunk #{h.index}</span>
                              <span className="chunk-score">sim {h.score}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                    </div>
                  </div>
                )}

                {/* evidence chunks */}
                {m.role === 'assistant' && m.chunks && (
                  <div className="evidence">
                    <div className="ev-head">
                      <span className="b-label">final retrieved chunks</span>
                      <span className="ev-meta">
                        retr {m.latency?.retrieval_ms}ms · llm {m.latency?.llm_ms}ms · total {m.latency?.total_ms}ms
                        {' · '}{m.tokens?.input}/{m.tokens?.output} tok
                      </span>
                    </div>
                    {m.chunks.map((c, j) => (
                      <div key={j} className="chunk">
                        <div className="chunk-head">
                          <span className="tag">chunk #{c.index}</span>
                          <span className="chunk-score">sim {c.score}</span>
                        </div>
                        <p className="chunk-snippet">{c.snippet}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {busy && <p className="empty-note">thinking… (retrieving + generating)</p>}
          </div>

          <div className="ask-row">
            <input
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleSend() }}
              placeholder="Ask about the document…"
              disabled={busy}
            />
            <select
              id="method-select"
              value={selectedMethod}
              onChange={e => setSelectedMethod(e.target.value)}
              className="method-select"
              aria-label="Execution mode"
              disabled={busy}
            >
              {TRANSLATION_METHODS.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
            <button className="cta-btn primary" onClick={handleSend} disabled={busy}>Send</button>
          </div>
        </section>
      </div>

      <footer>
        <span className="r">//</span> grounding + reliability made visible — every answer shows exactly which chunks it read
      </footer>
    </>
  )
}
