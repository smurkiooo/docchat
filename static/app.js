"use strict";

const $ = (selector) => document.querySelector(selector);

const els = {
  sidebar: $("#sidebar"),
  menuBtn: $("#menu-btn"),
  dropzone: $("#dropzone"),
  fileInput: $("#file-input"),
  docList: $("#doc-list"),
  emptyDocs: $("#empty-docs"),
  docTemplate: $("#doc-item"),
  status: $("#status"),
  statusText: $("#status-text"),
  title: $("#doc-title"),
  meta: $("#doc-meta"),
  openPdf: $("#open-pdf"),
  clearBtn: $("#clear-btn"),
  messages: $("#messages"),
  placeholder: $("#placeholder"),
  placeholderTitle: $("#placeholder-title"),
  placeholderText: $("#placeholder-text"),
  suggestions: $("#suggestions"),
  composer: $("#composer"),
  question: $("#question"),
  sendBtn: $("#send-btn"),
  toast: $("#toast"),
};

const state = {
  documents: [],
  activeId: localStorageGet("docchat.activeId"),
  controller: null,
  pollTimer: null,
  loadToken: 0,
};


function localStorageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function localStorageSet(key, value) {
  try { localStorage.setItem(key, value); } catch {  }
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function pluralPages(n) {
  const mod10 = n % 10, mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return `${n} страница`;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return `${n} страницы`;
  return `${n} страниц`;
}

let toastTimer;
function showToast(message) {
  els.toast.textContent = message;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 5000);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `Ошибка ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {  }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

const CITATION = /\[(стр\.|с\.|p\.|pp\.|page)\s*(\d+)([^\]]{0,20})\]/gi;

function renderMarkdown(text) {
  const withCitations = text.replace(CITATION, (_, label, page, rest) =>
    `<span class="cite" data-page="${page}">${escapeHtml(label + " " + page + rest)}</span>`);
  if (window.marked && window.DOMPurify) {
    return DOMPurify.sanitize(marked.parse(withCitations, { breaks: true }));
  }

  const escaped = escapeHtml(text).replace(/\n/g, "<br>");
  return escaped.replace(CITATION, (_, label, page, rest) =>
    `<span class="cite" data-page="${page}">${label} ${page}${rest}</span>`);
}


function activeDocument() {
  return state.documents.find((d) => d.id === state.activeId) || null;
}

function documentSubtitle(doc) {
  if (doc.status === "processing") return `Индексация… ${Math.round(doc.progress * 100)}%`;
  if (doc.status === "error") return doc.error || "Ошибка";
  return `${pluralPages(doc.pages)} · ${doc.chunks} фрагментов`;
}

function renderDocuments() {
  els.docList.replaceChildren();
  els.emptyDocs.hidden = state.documents.length > 0;
  for (const doc of state.documents) {
    const item = els.docTemplate.content.firstElementChild.cloneNode(true);
    item.classList.add(doc.status);
    item.classList.toggle("active", doc.id === state.activeId);
    item.querySelector(".doc-name").textContent = doc.filename;
    item.querySelector(".doc-name").title = doc.filename;
    item.querySelector(".doc-sub").textContent = documentSubtitle(doc);
    item.querySelector(".progress-bar").style.width = `${Math.round(doc.progress * 100)}%`;
    item.querySelector(".doc-select").addEventListener("click", () => selectDocument(doc.id));
    item.querySelector(".doc-delete").addEventListener("click", () => deleteDocument(doc));
    els.docList.append(item);
  }
}

async function refreshDocuments() {
  const previous = activeDocument();
  state.documents = await api("/api/documents");
  renderDocuments();

  const current = activeDocument();
  if (!current && state.documents.length) {
    await selectDocument(state.documents[0].id);
  } else if (!current) {
    renderHeader();
  } else if (previous?.status !== current.status) {
    renderHeader();
    if (current.status === "ready" && !els.messages.querySelector(".msg")) await loadMessages();
  } else {
    renderHeader();
  }

  const processing = state.documents.some((d) => d.status === "processing");
  clearTimeout(state.pollTimer);
  if (processing) state.pollTimer = setTimeout(() => refreshDocuments().catch(console.error), 1000);
}

function renderHeader() {
  const doc = activeDocument();
  els.title.textContent = doc ? doc.filename : "Выберите документ";
  els.meta.textContent = doc ? documentSubtitle(doc) : "";
  els.openPdf.hidden = !doc;
  els.clearBtn.hidden = !doc || doc.status !== "ready";
  if (doc) els.openPdf.href = `/api/documents/${doc.id}/file`;

  const ready = doc?.status === "ready";
  els.question.disabled = !ready;
  els.question.placeholder = ready
    ? "Задайте вопрос по документу…"
    : doc?.status === "processing" ? "Дождитесь окончания индексации…" : "Сначала загрузите PDF";
  updateSendButton();

  if (!doc) {
    setPlaceholder("Загрузите PDF, чтобы начать", "Модель будет отвечать только по содержимому документа и укажет страницы-источники.");
  } else if (doc.status === "processing") {
    setPlaceholder("Индексируем документ…", "Разбиваем текст на фрагменты и строим эмбеддинги. Это займёт от нескольких секунд до минуты.");
  } else if (doc.status === "error") {
    setPlaceholder("Не удалось обработать документ", doc.error || "");
  } else {
    setPlaceholder("Задайте вопрос по документу", "Ответ будет основан только на тексте PDF, со ссылками на страницы.", true);
  }
}

function setPlaceholder(title, text, withSuggestions = false) {
  els.placeholderTitle.textContent = title;
  els.placeholderText.textContent = text;
  els.suggestions.hidden = !withSuggestions;
}

async function selectDocument(id) {
  if (state.controller) state.controller.abort();
  state.activeId = id;
  localStorageSet("docchat.activeId", id);
  els.sidebar.classList.remove("open");
  renderDocuments();
  renderHeader();
  await loadMessages();
}

async function deleteDocument(doc) {
  if (!confirm(`Удалить «${doc.filename}» вместе с историей чата?`)) return;
  try {
    await api(`/api/documents/${doc.id}`, { method: "DELETE" });
    if (state.activeId === doc.id) state.activeId = null;
    if (!state.activeId) clearMessages();
    await refreshDocuments();
  } catch (err) {
    showToast(err.message);
  }
}

async function uploadFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showToast("Поддерживаются только PDF-файлы.");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  els.dropzone.classList.add("dragover");
  try {
    const doc = await api("/api/documents", { method: "POST", body: form });
    if (state.controller) state.controller.abort();
    state.activeId = doc.id;
    localStorageSet("docchat.activeId", doc.id);
    clearMessages();
    await refreshDocuments();
    await loadMessages();
  } catch (err) {
    showToast(err.message);
  } finally {
    els.dropzone.classList.remove("dragover");
    els.fileInput.value = "";
  }
}


function clearMessages() {
  els.messages.querySelectorAll(".msg").forEach((m) => m.remove());
  els.placeholder.hidden = false;
}

async function loadMessages() {
  const token = ++state.loadToken;
  clearMessages();
  const doc = activeDocument();
  if (!doc || doc.status !== "ready") return;
  const messages = await api(`/api/documents/${doc.id}/messages`);
  if (token !== state.loadToken) return;
  clearMessages();
  for (const message of messages) {
    const view = appendMessage(message.role);
    if (message.role === "user") {
      view.bubble.textContent = message.content;
    } else {
      view.bubble.innerHTML = renderMarkdown(message.content);
      renderSources(view, message.sources);
    }
  }
  scrollToBottom(true);
}

function appendMessage(role) {
  els.placeholder.hidden = true;
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  wrapper.append(bubble);
  els.messages.append(wrapper);
  return { wrapper, bubble };
}

function renderSources(view, sources, docId = state.activeId) {
  if (!sources?.length) return;
  const doc = activeDocument();
  const details = document.createElement("details");
  details.className = "sources";
  details.innerHTML = `<summary><svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg>Источники (${sources.length})</summary>`;
  const list = document.createElement("div");
  list.className = "source-list";
  for (const source of sources) {
    const pages = source.page_start === source.page_end
      ? `стр. ${source.page_start}` : `стр. ${source.page_start}–${source.page_end}`;
    const item = document.createElement("div");
    item.className = "source";
    item.dataset.pageStart = source.page_start;
    item.dataset.pageEnd = source.page_end;
    item.innerHTML = `
      <div class="source-head">
        <a href="/api/documents/${docId}/file#page=${source.page_start}" target="_blank" rel="noopener">${pages}</a>
        <span class="source-score" title="Косинусная близость к вопросу">${source.similarity != null ? `sim ${source.similarity.toFixed(2)}` : "BM25"}</span>
      </div>
      <div class="source-text"></div>`;
    item.querySelector(".source-text").textContent = source.text;
    item.addEventListener("dblclick", () => item.classList.toggle("expanded"));
    item.title = "Двойной клик — показать полностью";
    list.append(item);
  }
  details.append(list);
  view.wrapper.append(details);
}

function highlightSource(cite) {
  const page = Number(cite.dataset.page);
  const wrapper = cite.closest(".msg");
  const details = wrapper?.querySelector(".sources");
  if (!details) {
    const doc = activeDocument();
    if (doc) window.open(`/api/documents/${doc.id}/file#page=${page}`, "_blank", "noopener");
    return;
  }
  details.open = true;
  const sources = [...details.querySelectorAll(".source")];
  const match = sources.find((s) => page >= Number(s.dataset.pageStart) && page <= Number(s.dataset.pageEnd));
  if (!match) return;
  sources.forEach((s) => s.classList.remove("highlight"));
  match.classList.add("highlight", "expanded");
  match.scrollIntoView({ behavior: "smooth", block: "nearest" });
  setTimeout(() => match.classList.remove("highlight"), 2000);
}

function scrollToBottom(force = false) {
  const el = els.messages;
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  if (force || nearBottom) el.scrollTop = el.scrollHeight;
}


async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      yield { event, data: data ? JSON.parse(data) : null };
    }
  }
}

async function ask(question) {
  const doc = activeDocument();
  if (!doc || doc.status !== "ready" || state.controller) return;

  appendMessage("user").bubble.textContent = question;
  const view = appendMessage("assistant");
  view.bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  scrollToBottom(true);

  const controller = new AbortController();
  state.controller = controller;
  updateSendButton();

  let text = "";
  let sources = [];
  let completed = false;
  try {
    const response = await fetch(`/api/documents/${doc.id}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `Ошибка ${response.status}`);
    }
    for await (const { event, data } of readSSE(response)) {
      if (event === "sources") sources = data;
      else if (event === "token") {
        text += data.t;
        view.bubble.innerHTML = renderMarkdown(text);
        scrollToBottom();
      } else if (event === "done") completed = true;
      else if (event === "error") throw new Error(data.detail);
    }
    if (!completed) throw new Error("Поток ответа прерван. Попробуйте ещё раз.");
  } catch (err) {
    if (err.name === "AbortError") {
      if (!text) view.bubble.textContent = "Генерация остановлена.";
    } else {
      view.wrapper.classList.add("error");
      view.bubble.textContent = text ? `${text}\n\n⚠ ${err.message}` : err.message;
    }
  } finally {
    if (state.controller === controller) state.controller = null;
    renderSources(view, sources, doc.id);
    updateSendButton();
    scrollToBottom();
    els.question.focus();
  }
}

function updateSendButton() {
  const streaming = Boolean(state.controller);
  els.sendBtn.classList.toggle("streaming", streaming);
  els.sendBtn.disabled = !streaming && (els.question.disabled || !els.question.value.trim());
  els.sendBtn.setAttribute("aria-label", streaming ? "Остановить" : "Отправить");
}

function autoResize() {
  els.question.style.height = "auto";
  els.question.style.height = `${Math.min(els.question.scrollHeight, 200)}px`;
}

function submitQuestion() {
  const question = els.question.value.trim();
  if (!question) return;
  els.question.value = "";
  autoResize();
  ask(question);
}


async function checkHealth() {
  try {
    const health = await api("/api/health");
    els.status.classList.toggle("ok", health.llm_available);
    els.status.classList.toggle("bad", !health.llm_available);
    els.statusText.textContent = health.llm_available
      ? `${health.llm_model} · готова`
      : `${health.llm_model} · недоступна`;
    els.status.title = `LLM: ${health.llm_provider} / ${health.llm_model}\nЭмбеддинги: ${health.embedding_model}`;
  } catch {
    els.status.classList.add("bad");
    els.statusText.textContent = "Сервер недоступен";
  }
}


els.fileInput.addEventListener("change", () => uploadFile(els.fileInput.files[0]));

for (const type of ["dragenter", "dragover"]) {
  document.addEventListener(type, (e) => { e.preventDefault(); els.dropzone.classList.add("dragover"); });
}
for (const type of ["dragleave", "drop"]) {
  document.addEventListener(type, (e) => {
    e.preventDefault();
    if (type === "dragleave" && e.relatedTarget) return;
    els.dropzone.classList.remove("dragover");
  });
}
document.addEventListener("drop", (e) => uploadFile(e.dataTransfer?.files[0]));

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.controller) state.controller.abort();
  else submitQuestion();
});

els.question.addEventListener("input", () => { autoResize(); updateSendButton(); });
els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!state.controller) submitQuestion();
  }
});

els.suggestions.addEventListener("click", (e) => {
  const button = e.target.closest(".suggestion");
  if (button) ask(button.textContent);
});

els.messages.addEventListener("click", (e) => {
  const cite = e.target.closest(".cite");
  if (cite) highlightSource(cite);
});

els.clearBtn.addEventListener("click", async () => {
  const doc = activeDocument();
  if (!doc || !confirm("Очистить историю чата по этому документу?")) return;
  try {
    await api(`/api/documents/${doc.id}/messages`, { method: "DELETE" });
    clearMessages();
  } catch (err) {
    showToast(err.message);
  }
});

els.menuBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  els.sidebar.classList.toggle("open");
});
document.querySelector(".chat").addEventListener("click", () => els.sidebar.classList.remove("open"));


checkHealth();
setInterval(checkHealth, 30000);
refreshDocuments().catch((err) => showToast(err.message));
