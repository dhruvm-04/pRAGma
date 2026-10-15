"use strict";

const METHODS = {
  none: ["Direct", "Retrieve evidence using your original question."],
  multi_query: [
    "Multi Query",
    "Generate alternative questions and merge evidence by best similarity.",
  ],
  rag_fusion: [
    "RAG Fusion",
    "Retrieve for each question, then combine the rankings with reciprocal rank fusion.",
  ],
  decomposition: [
    "Decomposition",
    "Answer simpler sub-questions, then synthesize a response.",
  ],
  step_back: [
    "Step Back",
    "Search for broad context alongside your specific question.",
  ],
  hyde: [
    "HyDE",
    "Use a hypothetical answer to search for real document evidence.",
  ],
};
const $ = (id) => document.getElementById(id);
const state = { querying: false, uploading: false };

// Untrusted PDF, question, and model text always enters through textContent.
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function request(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error(
      "Cannot reach the backend. Check the connection and retry.",
    );
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      `The server returned an unreadable response (${response.status}).`,
    );
  }
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg).join("; ")
      : data.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : `Request failed (${response.status}).`,
    );
  }
  return data;
}

function setStatus(id, message, error = false) {
  $(id).textContent = message;
  $(id).classList.toggle("error", error);
}

function updateControls() {
  const busy = state.querying || state.uploading;
  for (const id of [
    "upload-button",
    "file-input",
    "question",
    "method-select",
    "send-button",
  ])
    $(id).disabled = busy;
  $("send-button").textContent = state.querying ? "Working…" : "Ask";
  $("question-form").setAttribute("aria-busy", String(state.querying));
}

async function refreshLibrary() {
  try {
    const health = await request("/health");
    $("library-status").textContent = health.chunks
      ? `${health.chunks} indexed chunks${health.source ? ` · Latest: ${health.source}` : ""}`
      : "No documents yet.";
  } catch (error) {
    $("library-status").textContent = error.message;
  }
}

async function uploadFile() {
  const file = $("file-input").files[0];
  if (!file || state.uploading || state.querying) return;
  if (!file.name.toLowerCase().endsWith(".pdf") || !file.size) {
    setStatus("upload-status", "Choose a nonempty PDF file.", true);
    $("file-input").value = "";
    return;
  }
  state.uploading = true;
  updateControls();
  $("upload-progress").hidden = false;
  $("upload-progress").removeAttribute("value");
  setStatus("upload-status", `Uploading ${file.name}…`);
  try {
    const form = new FormData();
    form.append("file", file);
    const { job_id: jobId } = await request("/ingest", {
      method: "POST",
      body: form,
    });
    if (!jobId)
      throw new Error("The server did not return an upload job. Please retry.");
    const started = Date.now();
    while (true) {
      const job = await request(`/ingest/status/${encodeURIComponent(jobId)}`);
      if (job.status === "error")
        throw new Error(job.message || "Document indexing failed.");
      if (job.total > 0)
        $("upload-progress").value = Math.min(
          100,
          (job.done / job.total) * 100,
        );
      setStatus(
        "upload-status",
        `${job.message}${job.total ? ` · ${job.done}/${job.total} chunks` : ""}`,
      );
      if (job.status === "done") {
        setStatus(
          "upload-status",
          `${job.source}: ${job.done} chunks indexed in ${((Date.now() - started) / 1000).toFixed(1)}s.`,
        );
        await refreshLibrary();
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  } catch (error) {
    setStatus("upload-status", error.message, true);
  } finally {
    state.uploading = false;
    $("upload-progress").hidden = true;
    $("file-input").value = "";
    updateControls();
  }
}

// A deliberately small text-only Markdown renderer: no raw HTML or remote assets.
function inlineText(parent, text) {
  for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)) {
    if (part.startsWith("**") && part.endsWith("**"))
      parent.append(element("strong", "", part.slice(2, -2)));
    else if (part.startsWith("`") && part.endsWith("`"))
      parent.append(element("code", "", part.slice(1, -1)));
    else parent.append(document.createTextNode(part));
  }
}

function renderAnswer(text) {
  const output = element("div", "answer-text");
  let paragraph = null,
    list = null,
    code = null;
  for (const line of text.split("\n")) {
    if (line.trim().startsWith("```")) {
      if (code) code = null;
      else {
        code = element("pre");
        output.append(code);
      }
      paragraph = list = null;
      continue;
    }
    if (code) {
      code.textContent += `${line}\n`;
      continue;
    }
    if (!line.trim()) {
      paragraph = list = null;
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.+)/);
    const bullet = line.match(/^\s*(?:([-*])|\d+[.)])\s+(.+)/);
    if (heading) {
      const node = element("h3");
      inlineText(node, heading[1]);
      output.append(node);
      paragraph = list = null;
    } else if (bullet) {
      const tag = bullet[1] ? "UL" : "OL";
      if (!list || list.tagName !== tag) {
        list = element(tag.toLowerCase());
        output.append(list);
      }
      const item = element("li");
      inlineText(item, bullet[2]);
      list.append(item);
      paragraph = null;
    } else {
      list = null;
      if (!paragraph) {
        paragraph = element("p");
        output.append(paragraph);
      } else paragraph.append(document.createTextNode("\n"));
      inlineText(paragraph, line);
    }
  }
  return output;
}

function detailsBlock(parent, title, open = false) {
  const details = element("details");
  details.open = open;
  details.append(element("summary", "", title));
  const body = element("div", "detail-body");
  details.append(body);
  parent.append(details);
  return body;
}

function renderResult(exchange, data, method) {
  exchange.append(renderAnswer(data.answer));
  const trace = element("div", "trace");
  trace.append(element("strong", "", METHODS[method][0]));
  trace.append(
    element(
      "p",
      "",
      data.fusion
        ? `RRF score = sum of 1 / (${data.fusion.k} + rank) across ${data.fusion.query_count} queries. Higher is better; this is not a similarity percentage.`
        : METHODS[method][1],
    ),
  );
  exchange.append(trace);
  const timing = data.latency,
    tokens = data.tokens;
  if (timing && tokens)
    exchange.append(
      element(
        "p",
        "metrics",
        `Rewrite ${timing.translation_ms ?? 0} ms · Retrieve ${timing.retrieval_ms} ms · Generate ${timing.llm_ms} ms · Total ${timing.total_ms} ms\nTokens: ${tokens.input} input / ${tokens.output} output (all model calls)`,
      ),
    );

  for (const step of data.translation_steps || []) {
    const body = detailsBlock(
      exchange,
      `Search queries (${step.queries.length})`,
    );
    const list = element("ol", "queries");
    for (const query of step.queries) list.append(element("li", "", query));
    body.append(list);
  }
  if (data.retrieval_details?.length) {
    const body = detailsBlock(exchange, "Rankings for each query");
    data.retrieval_details.forEach((retrieval, index) => {
      const card = element("div", "retrieval-query");
      card.append(element("h3", "", `Q${index + 1} · ${retrieval.query}`));
      for (const hit of retrieval.hits) {
        const row = element("div", "hit");
        row.append(
          element(
            "span",
            "",
            `#${hit.rank} · ${hit.source} · chunk ${hit.index}`,
          ),
        );
        row.append(
          element(
            "span",
            "",
            `Similarity ${hit.score.toFixed(3)}${hit.rrf_contribution !== undefined ? ` · RRF +${hit.rrf_contribution.toFixed(6)}` : ""}`,
          ),
        );
        card.append(row);
      }
      body.append(card);
    });
  }
  if (data.chunks?.length) {
    const body = detailsBlock(
      exchange,
      `Final evidence (${data.chunks.length} chunks)`,
      true,
    );
    for (const chunk of data.chunks) {
      const card = element("div", "chunk");
      const head = element("div", "chunk-head");
      head.append(
        element("strong", "", `Evidence ${chunk.rank} · chunk ${chunk.index}`),
      );
      head.append(
        element(
          "span",
          "score",
          chunk.score_type === "rrf"
            ? `RRF ${chunk.score.toFixed(6)}`
            : `Similarity ${chunk.score.toFixed(3)}`,
        ),
      );
      card.append(
        head,
        element("p", "chunk-source", chunk.source),
        element("p", "chunk-snippet", chunk.snippet),
      );
      if (chunk.contributions?.length)
        card.append(
          element(
            "p",
            "contributions",
            chunk.contributions
              .map(
                (c) =>
                  `Q${c.query_index + 1} rank ${c.rank}: +${c.contribution.toFixed(6)}`,
              )
              .join(" · "),
          ),
        );
      body.append(card);
    }
  }
}

async function ask(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || state.querying || state.uploading) return;
  const method = $("method-select").value;
  state.querying = true;
  updateControls();
  $("empty-state")?.remove();
  const exchange = element("article", "exchange");
  exchange.append(
    element("p", "eyebrow", "Your question"),
    element("p", "question-text", question),
  );
  $("evidence").append(exchange);
  setStatus(
    "query-status",
    `${METHODS[method][0]}: retrieving evidence and generating an answer…`,
  );
  try {
    const data = await request("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, method }),
    });
    if (typeof data.answer !== "string" || !data.answer.trim())
      throw new Error("The server returned an empty answer. Please retry.");
    renderResult(exchange, data, method);
    $("question").value = "";
    setStatus("query-status", "Answer ready. Review the evidence below.");
  } catch (error) {
    exchange.append(element("p", "error-text", error.message));
    setStatus(
      "query-status",
      "Question failed. Your question is preserved so you can retry.",
      true,
    );
  } finally {
    state.querying = false;
    updateControls();
    $("question").focus({ preventScroll: true });
    exchange.scrollIntoView({ block: "start" });
  }
}

function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === "dark";
  $("theme-toggle").textContent = dark ? "Light mode" : "Dark mode";
  $("theme-toggle").setAttribute(
    "aria-label",
    `Switch to ${dark ? "light" : "dark"} mode`,
  );
}
$("theme-toggle").addEventListener("click", () => {
  const next =
    document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try {
    localStorage.setItem("rag-theme", next);
  } catch {
    /* Storage may be disabled. */
  }
  updateThemeButton();
});
$("upload-button").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", uploadFile);
$("question-form").addEventListener("submit", ask);
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("question-form").requestSubmit();
  }
});
$("method-select").addEventListener("change", () => {
  $("method-description").textContent = METHODS[$("method-select").value][1];
});
$("method-description").textContent = METHODS[$("method-select").value][1];
updateThemeButton();
refreshLibrary();
