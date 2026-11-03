"use strict";

const $ = (selector) => document.querySelector(selector);

const els = {
  sidebar: $("#sidebar"),
  layout: $("#layout"),
  chat: $("#chat"),
  backdrop: $("#sidebar-backdrop"),
  sidebarClose: $("#sidebar-close"),
  emptyUpload: $("#empty-upload"),
  composerDock: $("#composer-dock"),
  uploadHint: $("#upload-hint"),
  indexProgress: $("#index-progress"),
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
  libraryLoaded: false,
  uploading: false,
  focusQuestion: false,
  sidebarOpen: false,
  sidebarCollapsed: localStorageGet("docchat.sidebarCollapsed") === "true",
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
    `<button type="button" class="cite" data-page="${page}">${escapeHtml(label + " " + page + rest)}</button>`);
  if (window.marked && window.DOMPurify) {
    return DOMPurify.sanitize(marked.parse(withCitations, { breaks: true }));
  }

  const escaped = escapeHtml(text).replace(/\n/g, "<br>");
  return escaped.replace(CITATION, (_, label, page, rest) =>
    `<button type="button" class="cite" data-page="${page}">${label} ${page}${rest}</button>`);
}


function activeDocument() {
  return state.documents.find((d) => d.id === state.activeId) || null;
}

function documentSubtitle(doc) {
  if (doc.status === "processing") return `Обработка… ${Math.round(doc.progress * 100)}%`;
  if (doc.status === "error") return "Ошибка обработки";
  return pluralPages(doc.pages);
}

function renderDocuments() {
  els.docList.replaceChildren();
  els.emptyDocs.hidden = state.documents.length > 0;
  for (const doc of state.documents) {
    const item = els.docTemplate.content.firstElementChild.cloneNode(true);
    item.classList.add(doc.status);
    item.classList.toggle("active", doc.id === state.activeId);
    if (doc.id === state.activeId) item.querySelector(".doc-select").setAttribute("aria-current", "true");
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
  state.libraryLoaded = true;
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
  els.title.textContent = doc ? doc.filename : "DocChat";
  els.meta.textContent = doc ? documentSubtitle(doc) : "";
  els.openPdf.hidden = !doc;
  els.clearBtn.hidden = !doc || doc.status !== "ready";
  if (doc) els.openPdf.href = `/api/documents/${doc.id}/file`;

  const ready = doc?.status === "ready";
  const welcome = !state.documents.length;
  els.layout.classList.toggle("welcome", welcome);
  els.menuBtn.hidden = welcome;
  els.composerDock.hidden = !ready;
  els.indexProgress.hidden = doc?.status !== "processing";
  els.indexProgress.value = doc?.progress || 0;
  els.uploadHint.hidden = !welcome;
  if (!state.uploading) els.emptyUpload.textContent = "Выбрать PDF";
  syncSidebar();
  els.question.disabled = !ready;
  els.question.placeholder = ready
    ? "Задайте вопрос по документу…"
    : doc?.status === "processing" ? "Дождитесь окончания индексации…" : "Сначала загрузите PDF";
  updateSendButton();

  if (!doc) {
    setPlaceholder("Чат с PDF", "Загрузите документ, чтобы задать вопрос и найти нужные страницы.");
  } else if (doc.status === "processing") {
    setPlaceholder("Обрабатываем PDF", `Готово ${Math.round(doc.progress * 100)}%. Когда обработка закончится, здесь появится чат.`);
  } else if (doc.status === "error") {
    setPlaceholder("Не удалось обработать документ", doc.error || "");
  } else {
    setPlaceholder("Что вы хотите узнать?", "Задайте вопрос или выберите один из примеров ниже.", true);
    if (state.focusQuestion) {
      state.focusQuestion = false;
      if (!mobileLayout.matches) els.question.focus({ preventScroll: true });
    }
  }
}

function setPlaceholder(title, text, withSuggestions = false) {
  els.placeholderTitle.textContent = title;
  els.placeholderText.textContent = text;
  els.suggestions.hidden = !withSuggestions;
  els.emptyUpload.hidden = Boolean(activeDocument());
}

async function selectDocument(id) {
  if (state.controller) state.controller.abort();
  state.activeId = id;
  localStorageSet("docchat.activeId", id);
  closeSidebar();
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
  if (!file || state.uploading) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showToast("Поддерживаются только PDF-файлы.");
    return;
  }
  state.uploading = true;
  els.emptyUpload.disabled = true;
  els.dropzone.disabled = true;
  els.emptyUpload.textContent = "Загружаем…";
  const form = new FormData();
  form.append("file", file);
  els.dropzone.classList.add("dragover");
  try {
    const doc = await api("/api/documents", { method: "POST", body: form });
    if (state.controller) state.controller.abort();
    state.activeId = doc.id;
    state.focusQuestion = true;
    closeSidebar();
    localStorageSet("docchat.activeId", doc.id);
    clearMessages();
    await refreshDocuments();
    await loadMessages();
  } catch (err) {
    showToast(err.message);
  } finally {
    els.layout.classList.remove("dragover");
    els.dropzone.classList.remove("dragover");
    state.uploading = false;
    els.emptyUpload.disabled = false;
    els.dropzone.disabled = false;
    els.emptyUpload.textContent = "Выбрать PDF";
    els.fileInput.value = "";
  }
}


function clearMessages() {
  els.messages.querySelectorAll(".msg").forEach((m) => m.remove());
  els.placeholder.hidden = false;
  els.chat.classList.add("empty");
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
  els.chat.classList.remove("empty");
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
        <button class="source-toggle" type="button" aria-expanded="false">Развернуть</button>
      </div>
      <div class="source-text"></div>`;
    item.querySelector(".source-text").textContent = source.text;
    item.querySelector(".source-toggle").addEventListener("click", () => {
      const expanded = item.classList.toggle("expanded");
      const button = item.querySelector(".source-toggle");
      button.setAttribute("aria-expanded", String(expanded));
      button.textContent = expanded ? "Свернуть" : "Развернуть";
    });
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
  match.querySelector(".source-toggle").setAttribute("aria-expanded", "true");
  match.querySelector(".source-toggle").textContent = "Свернуть";
  match.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
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

  ++state.loadToken;
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
      ? "Модель подключена"
      : "Модель недоступна";
    els.status.title = `LLM: ${health.llm_provider} / ${health.llm_model}\nЭмбеддинги: ${health.embedding_model}`;
  } catch {
    els.status.classList.add("bad");
    els.statusText.textContent = "Сервер недоступен";
  }
}


els.fileInput.addEventListener("change", () => uploadFile(els.fileInput.files[0]));

for (const type of ["dragenter", "dragover"]) {
  document.addEventListener(type, (e) => {
    if (!e.dataTransfer?.types.includes("Files")) return;
    e.preventDefault();
    els.layout.classList.add("dragover");
  });
}
for (const type of ["dragleave", "drop"]) {
  document.addEventListener(type, (e) => {
    e.preventDefault();
    if (type === "dragleave" && e.relatedTarget) return;
    els.layout.classList.remove("dragover");
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
  if (button) ask(button.dataset.question || button.textContent);
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

const mobileLayout = matchMedia("(max-width: 760px)");

function syncSidebar() {
  const mobile = mobileLayout.matches;
  const welcome = els.layout.classList.contains("welcome");
  const visible = !welcome && (mobile ? state.sidebarOpen : !state.sidebarCollapsed);
  els.layout.classList.toggle("sidebar-collapsed", !welcome && !mobile && !visible);
  els.sidebar.classList.toggle("open", mobile && visible);
  els.sidebar.inert = !visible;
  els.sidebar.setAttribute("aria-hidden", String(!visible));
  els.backdrop.hidden = !mobile || !visible;
  els.chat.inert = mobile && visible;
  els.menuBtn.setAttribute("aria-expanded", String(visible));
  els.menuBtn.setAttribute("aria-label", visible ? "Скрыть список документов" : "Показать список документов");
  if (mobile && visible) {
    els.sidebar.setAttribute("role", "dialog");
    els.sidebar.setAttribute("aria-modal", "true");
  } else {
    els.sidebar.removeAttribute("role");
    els.sidebar.removeAttribute("aria-modal");
  }
}

function closeSidebar() {
  const wasOpen = state.sidebarOpen;
  state.sidebarOpen = false;
  syncSidebar();
  if (wasOpen && mobileLayout.matches) els.menuBtn.focus();
}

els.menuBtn.addEventListener("click", () => {
  if (mobileLayout.matches) state.sidebarOpen = !state.sidebarOpen;
  else {
    state.sidebarCollapsed = !state.sidebarCollapsed;
    localStorageSet("docchat.sidebarCollapsed", String(state.sidebarCollapsed));
  }
  syncSidebar();
  if (mobileLayout.matches && state.sidebarOpen) els.sidebarClose.focus({ preventScroll: true });
});
els.sidebarClose.addEventListener("click", closeSidebar);
els.backdrop.addEventListener("click", closeSidebar);
mobileLayout.addEventListener("change", closeSidebar);
document.addEventListener("keydown", (event) => {
  if (!mobileLayout.matches || !state.sidebarOpen) return;
  if (event.key === "Escape") closeSidebar();
  if (event.key === "Tab") {
    const focusable = [...els.sidebar.querySelectorAll("a[href], button:not(:disabled)")];
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  }
});
els.dropzone.addEventListener("click", () => els.fileInput.click());
els.emptyUpload.addEventListener("click", () => {
  if (state.libraryLoaded) els.fileInput.click();
  else loadLibrary();
});

async function loadLibrary() {
  els.emptyUpload.hidden = true;
  try {
    await refreshDocuments();
  } catch {
    if (state.libraryLoaded) {
      showToast("Не удалось загрузить историю. Обновите страницу.");
      return;
    }
    setPlaceholder("Не удалось открыть библиотеку", "Проверьте, запущен ли сервер, и повторите попытку.");
    els.emptyUpload.textContent = "Повторить";
    els.emptyUpload.hidden = false;
  }
}
syncSidebar();

checkHealth();
setInterval(checkHealth, 30000);
loadLibrary();
