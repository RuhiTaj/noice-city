"use strict";

const API = "/api";
const SESSION_KEY = "voicetrainer_session";
const ACTIVE_VARIANT_KEY = "voicetrainer_active_variant";
const VARIANT_SESSIONS_KEY = "voicetrainer_variant_sessions";
const MAX_GENERATION_SEED = 4294967295;
const MODEL_DOWNLOAD_DISCLOSURES = Object.freeze({
  qwen: {
    name: "Qwen3-TTS 1.7B Base",
    repository: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    revision: "a6eb4f68e4b056f1215157bb696209bc82a6db48",
    license: "Apache-2.0 (declared by the pinned model card)",
    size: "4.54 GB",
    intro: "Qwen is the required local voice model for recording, training, and generation.",
  },
  whisper: {
    name: "Whisper large-v3-turbo",
    repository: "openai/whisper-large-v3-turbo",
    revision: "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    license: "MIT (declared by the pinned model card)",
    size: "1.623 GB",
    intro: "Whisper is optional for recording and generation. It adds automatic transcripts and WER checks, and is required before training because checkpoint evaluation is mandatory.",
  },
});

function bootstrapSessionToken() {
  const fragment = window.location.hash.slice(1);
  if (/^[a-f0-9]{64}$/i.test(fragment)) {
    sessionStorage.setItem(SESSION_KEY, fragment);
    history.replaceState(null, "", `${location.pathname}${location.search}`);
    return fragment;
  }
  return sessionStorage.getItem(SESSION_KEY) || "";
}

const sessionToken = bootstrapSessionToken();

const state = {
  view: "record",
  variants: [],
  variantId: localStorage.getItem(ACTIVE_VARIANT_KEY) || "",
  variantSessions: loadVariantSessions(),
  variantEpoch: 0,
  promptKind: "hybrid",
  customScriptUnlocked: false,
  customScriptComplete: false,
  customScriptPendingCount: null,
  savingCustomScript: false,
  skippedPrompts: [],
  prompt: null,
  recordings: [],
  datasets: [],
  checkpoints: [],
  status: {},
  statusRequestId: 0,
  reviewFilter: "all",
  selectedRecordings: new Set(),
  playedBackRecordings: new Set(),
  transcriptEdits: {},
  session: { takes: 0, seconds: 0, guided: 0, interview: 0 },
  capture: null,
  take: null,
  recordingStarting: false,
  savingTake: false,
  trainingRunId: null,
  trainingVariantId: null,
  trainingPoll: null,
  evaluationPoll: null,
  setupPoll: null,
  cancellingModelSetup: null,
  pendingModelDownload: null,
  outputUrl: null,
  blind: {
    checkpointId: null,
    comparison: null,
    index: 0,
    audio: null,
    audioUrl: null,
    audioOption: null,
    audioRequestId: 0,
  },
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type === "error" ? "error" : ""}`;
  toast.textContent = message;
  $("#toast-region").append(toast);
  window.setTimeout(() => toast.remove(), 4500);
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (sessionToken) headers.set("X-VoiceTrainer-Session", sessionToken);
  if (options.body && !(options.body instanceof FormData) && typeof options.body !== "string") {
    headers.set("Content-Type", "application/json");
    options.body = JSON.stringify(options.body);
  }
  headers.set("Accept", options.accept || "application/json");
  const response = await fetch(`${API}${path}`, {
    credentials: "same-origin",
    cache: "no-store",
    ...options,
    headers,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || body.error || body.message || message;
    } catch (_) { /* Preserve the HTTP error. */ }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.startsWith("audio/") || options.accept === "audio/wav") return response.blob();
  if (response.status === 204) return null;
  return response.json();
}

function normalizeList(value, keys = []) {
  if (Array.isArray(value)) return value;
  for (const key of keys) if (Array.isArray(value?.[key])) return value[key];
  return [];
}

function secondsLabel(seconds = 0) {
  const safe = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(safe / 60)}:${String(safe % 60).padStart(2, "0")}`;
}

function safeId(value) {
  return String(value ?? "").replace(/[^a-zA-Z0-9_.-]/g, "");
}

function loadVariantSessions() {
  try {
    const value = JSON.parse(localStorage.getItem(VARIANT_SESSIONS_KEY) || "{}");
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  } catch (_) {
    return {};
  }
}

function persistVariantSessions() {
  localStorage.setItem(VARIANT_SESSIONS_KEY, JSON.stringify(state.variantSessions));
}

function variantId(variant) {
  return String(variant?.id || variant?.variant_id || variant?.slug || "");
}

function normalizeVariant(variant) {
  return {
    ...variant,
    id: variantId(variant),
    name: variant?.name || variant?.display_name || variant?.label || variantId(variant),
    description: variant?.description || "",
    status: String(variant?.state || variant?.status || (variant?.archived ? "archived" : "active")).toLowerCase(),
  };
}

function activeVariant() {
  return state.variants.find((variant) => variant.id === state.variantId) || null;
}

function variantName() {
  return activeVariant()?.name || "no variant selected";
}

function variantPath(path) {
  if (!state.variantId) return path;
  return `${path}${path.includes("?") ? "&" : "?"}variant_id=${encodeURIComponent(state.variantId)}`;
}

function resourceVariantId(item) {
  if (!item || typeof item === "string") return "";
  const metrics = item.metrics_json || item.metrics || {};
  return String(
    item.variant_id
    || item.voice_variant_id
    || item.variant?.id
    || item.voice_variant?.variant_id
    || metrics.variant_id
    || metrics.voice_variant_id
    || "",
  );
}

function belongsToVariant(item, expectedVariant) {
  const resourceVariant = resourceVariantId(item);
  return Boolean(expectedVariant && resourceVariant && resourceVariant === String(expectedVariant));
}

function belongsToActiveVariant(item) {
  return belongsToVariant(item, state.variantId);
}

function currentVariantSession() {
  return state.variantId ? state.variantSessions[state.variantId] || null : null;
}

function sessionStats(session = {}) {
  const counts = session.counts || session.stats || {};
  return {
    takes: Number(session.takes ?? session.take_count ?? session.recording_count ?? counts.takes ?? 0) || 0,
    seconds: Number(session.seconds ?? session.duration_seconds ?? session.recorded_seconds ?? counts.seconds ?? 0) || 0,
    guided: Number(session.guided ?? session.guided_count ?? counts.guided ?? 0) || 0,
    interview: Number(session.interview ?? session.interview_count ?? counts.interview ?? 0) || 0,
  };
}

function sessionFromResponse(value, mode = state.promptKind) {
  const session = value?.session || value || {};
  const id = String(session.id || session.session_id || "");
  if (!id) throw new Error("The local service did not return a recording session ID.");
  return { id, mode: session.mode || mode, ...sessionStats(session) };
}

function recordingSessionMode() {
  return state.promptKind === "custom" ? "guided" : state.promptKind;
}

function saveCurrentSession() {
  const session = currentVariantSession();
  if (!session || !state.variantId) return;
  state.variantSessions[state.variantId] = { ...session, ...state.session };
  persistVariantSessions();
  renderVariantContext();
}

function clearGeneratedOutput() {
  if (state.outputUrl?.startsWith("blob:")) URL.revokeObjectURL(state.outputUrl);
  state.outputUrl = null;
  $("#generated-audio").removeAttribute("src");
  $("#generated-audio").load();
  $("#download-output").removeAttribute("href");
  $("#output-placeholder").hidden = false;
  $("#generated-output").hidden = true;
}

function renderVariantContext() {
  const name = variantName();
  for (const id of ["recording-variant-name", "review-variant-name", "train-variant-name", "generate-variant-name", "checkpoint-variant-name"]) {
    $(`#${id}`).textContent = name;
  }
  const session = currentVariantSession();
  $("#recording-session-label").textContent = session
    ? `Active session · ${session.id.slice(-6)}`
    : state.variantId ? "Creating a private session…" : "Waiting for a variant";
  $("#new-recording-session").disabled = !state.variantId || Boolean(state.capture || state.take || state.recordingStarting || state.savingTake);
}

function renderVariantControls() {
  const active = state.variants.filter((variant) => variant.status !== "archived");
  for (const selector of ["#variant-select", "#mobile-variant-select"]) {
    const select = $(selector);
    select.replaceChildren(new Option(active.length ? "Choose a variant" : "No variants yet", ""));
    for (const variant of active) select.append(new Option(variant.name, variant.id));
    select.value = active.some((variant) => variant.id === state.variantId) ? state.variantId : "";
    select.disabled = !active.length;
  }
  renderVariantContext();
  renderVariantManager();
}

function renderVariantManager() {
  const list = $("#variant-manager-list");
  list.replaceChildren();
  if (!state.variants.length) {
    const empty = document.createElement("p");
    empty.className = "variant-manager-empty";
    empty.textContent = "No voice variants exist yet. Create the first independent voice lineage below.";
    list.append(empty);
    return;
  }
  for (const variant of state.variants) {
    const row = document.createElement("article");
    row.className = `variant-manager-row ${variant.id === state.variantId ? "active" : ""} ${variant.status === "archived" ? "archived" : ""}`;
    const copy = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = variant.name;
    const description = document.createElement("p");
    description.textContent = variant.description || (variant.status === "archived" ? "Archived voice lineage" : "Independent local voice lineage");
    copy.append(name, description);
    const actions = document.createElement("div");
    if (variant.status !== "archived") {
      const use = document.createElement("button");
      use.type = "button";
      use.className = "button secondary compact";
      use.textContent = variant.id === state.variantId ? "In use" : "Use";
      use.disabled = variant.id === state.variantId;
      use.addEventListener("click", async () => {
        if (await switchVariant(variant.id)) $("#variant-dialog").close();
      });
      const rename = document.createElement("button");
      rename.type = "button";
      rename.className = "button secondary compact";
      rename.textContent = "Rename";
      rename.addEventListener("click", () => renameVariant(variant));
      const archive = document.createElement("button");
      archive.type = "button";
      archive.className = "button danger compact";
      archive.textContent = "Archive";
      archive.disabled = Boolean(variant.is_default);
      if (variant.is_default) archive.title = "The default voice variant cannot be archived.";
      archive.addEventListener("click", () => archiveVariant(variant));
      actions.append(use, rename, archive);
    } else {
      const badge = document.createElement("span");
      badge.className = "variant-archived-badge";
      badge.textContent = "Archived";
      actions.append(badge);
    }
    row.append(copy, actions);
    list.append(row);
  }
}

function variantSwitchBlockReason() {
  if (state.capture || state.recordingStarting) return "Stop the active recording before switching voice variants.";
  if (state.take) return "Save or discard the unsaved take before switching voice variants.";
  if (state.savingTake) return "Wait for the take to finish saving before switching voice variants.";
  return "";
}

async function loadVariants() {
  try {
    const data = await api("/variants");
    state.variants = normalizeList(data, ["variants", "items"]).map(normalizeVariant).filter((variant) => variant.id);
    const selectable = state.variants.filter((variant) => variant.status !== "archived");
    const saved = selectable.find((variant) => variant.id === state.variantId);
    renderVariantControls();
    if (saved) await switchVariant(saved.id, { force: true });
    else if (selectable.length) await switchVariant(selectable[0].id, { force: true });
    else {
      state.variantId = "";
      localStorage.removeItem(ACTIVE_VARIANT_KEY);
      renderVariantControls();
      $("#variant-dialog").showModal();
    }
  } catch (error) {
    state.variantId = "";
    renderVariantControls();
    showToast(`Could not load voice variants: ${error.message}`, "error");
  }
}

async function ensureRecordingSession({ forceNew = false } = {}) {
  if (!state.variantId) return null;
  const id = state.variantId;
  const epoch = state.variantEpoch;
  if (forceNew) {
    delete state.variantSessions[id];
    persistVariantSessions();
  }
  const stored = state.variantSessions[id];
  if (stored?.id) {
    state.session = sessionStats(stored);
    updateSession();
    renderVariantContext();
    return stored;
  }
  state.session = { takes: 0, seconds: 0, guided: 0, interview: 0 };
  updateSession();
  renderVariantContext();
  try {
    const created = sessionFromResponse(await api("/recording-sessions", {
      method: "POST",
      body: { variant_id: id, mode: recordingSessionMode() },
    }), recordingSessionMode());
    if (state.variantId !== id || state.variantEpoch !== epoch) return null;
    state.variantSessions[id] = created;
    persistVariantSessions();
    state.session = sessionStats(created);
    updateSession();
    renderVariantContext();
    return created;
  } catch (error) {
    if (state.variantId === id) {
      $("#recording-session-label").textContent = "Session unavailable";
      showToast(`Could not create a recording session: ${error.message}`, "error");
    }
    return null;
  }
}

function resetVariantScopedState() {
  state.statusRequestId += 1;
  state.promptKind = "hybrid";
  state.customScriptUnlocked = false;
  state.customScriptComplete = false;
  state.customScriptPendingCount = null;
  state.savingCustomScript = false;
  state.prompt = null;
  state.skippedPrompts = [];
  state.recordings = [];
  state.datasets = [];
  state.checkpoints = [];
  state.status = {};
  state.selectedRecordings.clear();
  state.playedBackRecordings.clear();
  state.transcriptEdits = {};
  state.session = { takes: 0, seconds: 0, guided: 0, interview: 0 };
  state.trainingRunId = null;
  state.trainingVariantId = null;
  window.clearTimeout(state.trainingPoll);
  state.trainingPoll = null;
  window.clearTimeout(state.evaluationPoll);
  state.evaluationPoll = null;
  stopBlindAudio();
  state.blind.checkpointId = null;
  state.blind.comparison = null;
  $("#blind-review").hidden = true;
  clearGeneratedOutput();
  $("#prompt-text").textContent = state.variantId ? "Creating a private recording session…" : "Choose a voice variant to begin.";
  $("#prompt-position").textContent = "Next prompt";
  $("#training-progress").hidden = true;
  $("#training-options").hidden = false;
  $("#training-title").textContent = "Ready for a new run";
  $("#training-state").textContent = "Idle";
  $("#training-state").className = "training-state idle";
  resetCustomScriptUI();
  renderRecordings();
  renderStatus();
  updateSession();
}

async function switchVariant(nextId, { force = false } = {}) {
  const next = state.variants.find((variant) => variant.id === String(nextId) && variant.status !== "archived");
  if (!next) {
    renderVariantControls();
    return false;
  }
  if (!force && next.id === state.variantId) {
    renderVariantControls();
    return true;
  }
  const blockReason = variantSwitchBlockReason();
  if (blockReason) {
    renderVariantControls();
    showToast(blockReason, "error");
    return false;
  }
  saveCurrentSession();
  state.variantEpoch += 1;
  state.variantId = next.id;
  localStorage.setItem(ACTIVE_VARIANT_KEY, next.id);
  resetVariantScopedState();
  renderVariantControls();
  const session = await ensureRecordingSession();
  if (!session || state.variantId !== next.id) return false;
  await Promise.allSettled([
    loadPrompt(),
    refreshStatus(),
    state.view === "review" ? loadRecordings() : Promise.resolve(),
  ]);
  return true;
}

async function createVariant(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("#create-variant");
  const name = $("#variant-name").value.trim();
  const description = $("#variant-description").value.trim();
  if (!name) return;
  button.disabled = true;
  button.textContent = "Creating…";
  try {
    const result = normalizeVariant(await api("/variants", { method: "POST", body: { name, description } }));
    const data = await api("/variants");
    state.variants = normalizeList(data, ["variants", "items"]).map(normalizeVariant).filter((variant) => variant.id);
    form.reset();
    renderVariantControls();
    const created = state.variants.find((variant) => variant.id === result.id)
      || [...state.variants].reverse().find((variant) => variant.name === name && variant.status !== "archived");
    if (!created) throw new Error("The new variant could not be found after creation.");
    if (await switchVariant(created.id)) {
      $("#variant-dialog").close();
      showToast(`${created.name} is ready for a fresh recording session.`);
    }
  } catch (error) {
    showToast(`Could not create voice variant: ${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Create variant";
  }
}

async function renameVariant(variant) {
  const name = window.prompt("Rename this voice variant", variant.name)?.trim();
  if (!name || name === variant.name) return;
  try {
    await api(`/variants/${encodeURIComponent(variant.id)}`, { method: "PATCH", body: { name } });
    variant.name = name;
    renderVariantControls();
    showToast("Voice variant renamed.");
  } catch (error) {
    showToast(`Could not rename voice variant: ${error.message}`, "error");
  }
}

async function archiveVariant(variant) {
  const blockReason = variant.id === state.variantId ? variantSwitchBlockReason() : "";
  if (blockReason) return showToast(blockReason, "error");
  if (!window.confirm(`Archive “${variant.name}”? Its local recordings, datasets, and checkpoints will remain stored.`)) return;
  try {
    await api(`/variants/${encodeURIComponent(variant.id)}`, { method: "PATCH", body: { state: "archived" } });
    variant.status = "archived";
    const remaining = state.variants.filter((item) => item.status !== "archived" && item.id !== variant.id);
    renderVariantControls();
    if (variant.id === state.variantId) {
      if (remaining.length) await switchVariant(remaining[0].id, { force: true });
      else {
        state.variantId = "";
        localStorage.removeItem(ACTIVE_VARIANT_KEY);
        resetVariantScopedState();
        renderVariantControls();
      }
    }
    showToast("Voice variant archived.");
  } catch (error) {
    showToast(`Could not archive voice variant: ${error.message}`, "error");
  }
}

async function newRecordingSession() {
  const blockReason = variantSwitchBlockReason();
  if (blockReason) return showToast(blockReason, "error");
  if (!state.variantId) return showToast("Choose a voice variant first.", "error");
  if (currentVariantSession()?.takes && !window.confirm("Start a new recording session for this variant? Saved takes stay in Review.")) return;
  const session = await ensureRecordingSession({ forceNew: true });
  if (session) {
    state.skippedPrompts = [];
    await loadPrompt();
    showToast(`New recording session started for ${variantName()}.`);
  }
}

function setView(name) {
  if (!$("[data-view-panel='" + name + "']")) return;
  if (state.view === "train" && name !== "train") stopBlindAudio();
  state.view = name;
  $$("[data-view-panel]").forEach((panel) => {
    const active = panel.dataset.viewPanel === name;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  history.replaceState(null, "", `#${name}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (state.variantId && name === "review") loadRecordings();
  if (state.variantId && (name === "train" || name === "generate")) refreshStatus();
}

function normalizePrompt(data) {
  const prompt = data?.prompt || data || {};
  return {
    id: prompt.id || prompt.prompt_id || crypto.randomUUID(),
    text: prompt.text || prompt.prompt || prompt.expected_text || "No prompt is available yet.",
    kind: prompt.kind || prompt.type || state.promptKind,
    instruction: prompt.instruction || prompt.delivery_note || (prompt.style ? `Delivery: ${prompt.style}. Keep it natural and leave a short pause at the end.` : "Use your everyday speaking voice and leave a short pause at the end."),
    position: prompt.position || prompt.number || prompt.ordinal || null,
    total: prompt.total || null,
    split: prompt.split || null,
  };
}

function updatePromptKindButtons() {
  $$(`[data-prompt-kind]`).forEach((button) => button.classList.toggle("active", button.dataset.promptKind === state.promptKind));
}

function updateCustomScriptSubmit() {
  const input = $("#custom-script-text");
  const length = input.value.length;
  $("#custom-script-count").textContent = length.toLocaleString();
  $("#save-custom-script").disabled = !state.customScriptUnlocked || state.savingCustomScript || !input.value.trim();
}

function updateCustomScriptGate(acceptedSeconds = 0, explicitUnlocked, eligibleSeconds = acceptedSeconds) {
  const unlockedByStatus = typeof explicitUnlocked === "boolean" ? explicitUnlocked : Number(acceptedSeconds) >= 120;
  const unlocked = Boolean(state.variantId && unlockedByStatus);
  state.customScriptUnlocked = unlocked;
  const button = $("#custom-script-mode");
  button.classList.toggle("locked", !unlocked);
  button.setAttribute("aria-disabled", String(!unlocked));
  button.title = unlocked
    ? "Record supplemental passages from your own English script"
    : `Unlocks after 2 approved Recommended mix minutes (${secondsLabel(eligibleSeconds)} eligible)`;
  $("#custom-script-lock").textContent = unlocked ? "✓ ready" : "● 2 min";
  updateCustomScriptSubmit();
}

function resetCustomScriptUI() {
  $("#custom-script-text").value = "";
  $("#custom-script-panel").hidden = true;
  $("#custom-script-complete").hidden = true;
  $("#prompt-card").hidden = false;
  $("#recorder").hidden = false;
  updatePromptKindButtons();
  updateCustomScriptGate(0);
}

function showCustomScriptEditor({ complete = false } = {}) {
  state.prompt = null;
  state.customScriptComplete = complete;
  if (complete) state.customScriptPendingCount = 0;
  $("#custom-script-panel").hidden = false;
  $("#custom-script-complete").hidden = !complete;
  $("#custom-script-title").textContent = complete ? "Record another script" : "Record your own script";
  $("#prompt-card").hidden = true;
  $("#recorder").hidden = true;
  $("#prompt-position").textContent = complete ? "Script complete" : "Supplemental script";
  updateCustomScriptSubmit();
}

function renderLoadedPrompt(prompt) {
  state.prompt = prompt;
  $("#custom-script-panel").hidden = true;
  $("#prompt-card").hidden = false;
  $("#recorder").hidden = false;
  $("#prompt-text").textContent = prompt.text;
  const custom = state.promptKind === "custom";
  $("#prompt-label").textContent = custom ? "Read your script naturally" : prompt.kind === "guided" ? "Read this naturally" : "Answer in your own words";
  $("#delivery-note").textContent = custom
    ? "Supplemental training take · use your natural narration voice and leave a short pause at the end."
    : prompt.instruction;
  if (custom && Number.isFinite(state.customScriptPendingCount)) {
    const remaining = Math.max(1, state.customScriptPendingCount);
    $("#prompt-position").textContent = `${remaining} short take${remaining === 1 ? "" : "s"} remaining`;
  } else {
    $("#prompt-position").textContent = prompt.position
      ? `Prompt ${prompt.position}${prompt.total ? ` of ${prompt.total}` : ""}`
      : custom ? "Your script" : "Next prompt";
  }
}

function customPromptFromResponse(data) {
  const value = data?.prompt || data?.item || data;
  if (!value || typeof value !== "object") return null;
  if (!(value.id || value.prompt_id) || !(value.text || value.prompt || value.expected_text)) return null;
  return normalizePrompt({
    ...value,
    kind: "custom",
    position: value.position || value.custom_position_number,
    total: value.total || value.custom_script_item_count,
    instruction: value.instruction || "Use your natural narration voice and leave a short pause at the end.",
  });
}

async function loadNextCustomPrompt({ completeWhenEmpty = true } = {}) {
  if (state.promptKind !== "custom" || !state.variantId || !state.customScriptUnlocked) return;
  const variant = state.variantId;
  const epoch = state.variantEpoch;
  const session = await ensureRecordingSession();
  if (!session || state.variantId !== variant || state.variantEpoch !== epoch || state.promptKind !== "custom") return;
  $("#custom-script-panel").hidden = true;
  $("#prompt-card").hidden = false;
  $("#recorder").hidden = true;
  $("#prompt-text").textContent = "Loading the next passage from your script…";
  $("#prompt-label").textContent = "Your script";
  $("#prompt-position").textContent = "Preparing take";
  try {
    const data = await api(`/custom-scripts/next?variant_id=${encodeURIComponent(variant)}`);
    if (state.variantId !== variant || state.variantEpoch !== epoch || state.promptKind !== "custom") return;
    const returnedCount = Number(data?.custom_script_pending_count ?? data?.pending_count ?? data?.item_count);
    if (Number.isFinite(returnedCount)) state.customScriptPendingCount = returnedCount;
    const prompt = customPromptFromResponse(data);
    if (!prompt) {
      showCustomScriptEditor({ complete: completeWhenEmpty });
      return;
    }
    renderLoadedPrompt(prompt);
  } catch (error) {
    if (state.variantId !== variant || state.variantEpoch !== epoch || state.promptKind !== "custom") return;
    if (error.status === 404) {
      showCustomScriptEditor({ complete: completeWhenEmpty });
      return;
    }
    showCustomScriptEditor({ complete: false });
    showToast(`Could not load your script: ${error.message}`, "error");
  }
}

async function saveCustomScript() {
  const text = $("#custom-script-text").value.trim();
  if (!text || !state.variantId || !state.customScriptUnlocked || state.savingCustomScript) return;
  const variant = state.variantId;
  const epoch = state.variantEpoch;
  state.savingCustomScript = true;
  updateCustomScriptSubmit();
  $("#save-custom-script").textContent = "Saving locally…";
  try {
    const result = await api("/custom-scripts", { method: "POST", body: { variant_id: variant, text } });
    if (state.variantId !== variant || state.variantEpoch !== epoch || state.promptKind !== "custom") return;
    const count = Number(result?.item_count ?? result?.pending_count);
    state.customScriptPendingCount = Number.isFinite(count) ? count : null;
    $("#custom-script-text").value = "";
    state.customScriptComplete = false;
    updateCustomScriptSubmit();
    showToast(Number.isFinite(count)
      ? `Script saved as about ${count} short take${count === 1 ? "" : "s"}.`
      : "Script saved locally. Your first short take is ready.");
    await loadNextCustomPrompt({ completeWhenEmpty: true });
  } catch (error) {
    if (state.variantId !== variant || state.variantEpoch !== epoch) return;
    showToast(`Could not save your script: ${error.message}`, "error");
  } finally {
    state.savingCustomScript = false;
    $("#save-custom-script").textContent = "Save & start recording";
    updateCustomScriptSubmit();
  }
}

async function selectPromptKind(kind) {
  if (state.capture) return;
  if (kind === "custom" && !state.customScriptUnlocked) {
    showToast("My script unlocks after this variant has 2 approved minutes. Keep building the Recommended mix first.", "error");
    updatePromptKindButtons();
    return;
  }
  discardTake();
  state.promptKind = kind;
  state.prompt = null;
  updatePromptKindButtons();
  if (kind === "custom") await loadNextCustomPrompt({ completeWhenEmpty: false });
  else {
    $("#custom-script-panel").hidden = true;
    $("#prompt-card").hidden = false;
    $("#recorder").hidden = false;
    loadPrompt();
  }
}

async function skipPrompt() {
  if (state.capture) return;
  if (state.promptKind !== "custom") {
    if (state.prompt?.id) state.skippedPrompts.push(state.prompt.id);
    discardTake();
    loadPrompt();
    return;
  }
  if (!state.prompt?.id || !state.variantId) return showCustomScriptEditor({ complete: false });
  const variant = state.variantId;
  const epoch = state.variantEpoch;
  const promptId = state.prompt.id;
  discardTake();
  try {
    await api(`/custom-scripts/prompts/${encodeURIComponent(promptId)}/skip`, {
      method: "POST",
      body: { variant_id: variant },
    });
    if (state.variantId !== variant || state.variantEpoch !== epoch || state.promptKind !== "custom") return;
    if (Number.isFinite(state.customScriptPendingCount)) state.customScriptPendingCount = Math.max(0, state.customScriptPendingCount - 1);
    await loadNextCustomPrompt({ completeWhenEmpty: true });
  } catch (error) {
    if (state.variantId !== variant || state.variantEpoch !== epoch) return;
    showToast(`Could not skip this script passage: ${error.message}`, "error");
  }
}

async function loadPrompt({ retrySession = true } = {}) {
  if (state.promptKind === "custom") return loadNextCustomPrompt({ completeWhenEmpty: true });
  const text = $("#prompt-text");
  if (!state.variantId) {
    state.prompt = null;
    text.textContent = "Choose a voice variant to begin.";
    return;
  }
  const variant = state.variantId;
  const epoch = state.variantEpoch;
  const session = await ensureRecordingSession();
  if (!session || state.variantId !== variant || state.variantEpoch !== epoch) return;
  text.textContent = "Loading your next prompt…";
  try {
    const query = new URLSearchParams({ session_id: session.id, kind: state.promptKind });
    for (const id of state.skippedPrompts) query.append("exclude", id);
    const prompt = normalizePrompt(await api(`/prompts/next?${query}`));
    if (state.variantId !== variant || state.variantEpoch !== epoch) return;
    renderLoadedPrompt(prompt);
  } catch (error) {
    if (state.variantId !== variant || state.variantEpoch !== epoch) return;
    if (retrySession && [404, 409, 410].includes(error.status) && state.variantId === variant) {
      const replacement = await ensureRecordingSession({ forceNew: true });
      if (replacement) return loadPrompt({ retrySession: false });
    }
    state.prompt = null;
    text.textContent = "The local prompt library is unavailable.";
    showToast(`Could not load a prompt: ${error.message}`, "error");
  }
}

function updateSession() {
  $("#session-takes").textContent = state.session.takes;
  $("#session-time").textContent = secondsLabel(state.session.seconds);
  const total = state.session.guided + state.session.interview;
  const guided = total ? Math.round((state.session.guided / total) * 100) : 80;
  $("#guided-mix").textContent = `${guided}%`;
  $("#interview-mix").textContent = `${100 - guided}%`;
  $("#guided-mix-bar").style.width = `${guided}%`;
}

function drawIdleMeter() {
  const canvas = $("#input-meter");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#deddd6";
  ctx.beginPath();
  for (let x = 0; x < width; x += 8) {
    const wave = 4 + 5 * Math.sin(x * 0.027) ** 2 + 2 * Math.sin(x * 0.091) ** 2;
    ctx.moveTo(x, height / 2 - wave);
    ctx.lineTo(x, height / 2 + wave);
  }
  ctx.stroke();
}

function drawLevel(level) {
  const canvas = $("#input-meter");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  const bars = 90;
  const active = Math.max(2, Math.round(level * bars * 3.1));
  for (let index = 0; index < bars; index += 1) {
    const x = (index / bars) * width;
    const near = Math.abs(index - bars / 2) / (bars / 2);
    const activity = index < active || bars - index < active;
    const heightScale = activity ? Math.max(.15, level * 1.8) : .07;
    const barHeight = 9 + (1 - near) * 52 * heightScale;
    ctx.fillStyle = activity ? "#d7593e" : "#deddd6";
    ctx.fillRect(x, (height - barHeight) / 2, 3, barHeight);
  }
}

function mergeChannels(buffer) {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0).slice();
  const output = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const samples = buffer.getChannelData(channel);
    for (let index = 0; index < samples.length; index += 1) output[index] += samples[index] / buffer.numberOfChannels;
  }
  return output;
}

function encodeWav(chunks, sourceSampleRate, sampleRate = 48000) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const captured = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { captured.set(chunk, offset); offset += chunk.length; }
  let samples = captured;
  if (sourceSampleRate !== sampleRate && captured.length) {
    const outputLength = Math.max(1, Math.round(captured.length * sampleRate / sourceSampleRate));
    samples = new Float32Array(outputLength);
    const ratio = sourceSampleRate / sampleRate;
    for (let index = 0; index < outputLength; index += 1) {
      const position = index * ratio;
      const left = Math.floor(position);
      const right = Math.min(captured.length - 1, left + 1);
      const mix = position - left;
      samples[index] = captured[left] * (1 - mix) + captured[right] * mix;
    }
  }
  const bytesPerSample = 3;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);
  const write = (position, value) => [...value].forEach((character, index) => view.setUint8(position + index, character.charCodeAt(0)));
  write(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  write(8, "WAVE"); write(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true); view.setUint16(34, 24, true); write(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    let value = Math.round(sample < 0 ? sample * 0x800000 : sample * 0x7fffff);
    if (value < 0) value += 0x1000000;
    const byteOffset = 44 + index * bytesPerSample;
    view.setUint8(byteOffset, value & 0xff);
    view.setUint8(byteOffset + 1, (value >> 8) & 0xff);
    view.setUint8(byteOffset + 2, (value >> 16) & 0xff);
  }
  return { blob: new Blob([view], { type: "audio/wav" }), seconds: samples.length / sampleRate, samples, sampleRate, bitDepth: 24 };
}

async function createCapture(onSamples) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: 48000, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
  });
  const context = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
  const source = context.createMediaStreamSource(stream);
  const analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = .72;
  source.connect(analyser);
  let processor;
  if (context.audioWorklet) {
    try {
      await context.audioWorklet.addModule("/recorder-worklet.js");
      processor = new AudioWorkletNode(context, "mono-recorder");
      processor.port.onmessage = (event) => onSamples(new Float32Array(event.data));
    } catch (_) { /* Older WebKit builds fall back to raw WebAudio capture below. */ }
  }
  if (!processor) {
    processor = context.createScriptProcessor(2048, 1, 1);
    processor.onaudioprocess = (event) => onSamples(mergeChannels(event.inputBuffer));
  }
  const silent = context.createGain();
  silent.gain.value = 0;
  source.connect(processor);
  processor.connect(silent);
  silent.connect(context.destination);
  return { stream, context, source, analyser, processor, silent, sampleRate: context.sampleRate };
}

function closeCapture(capture) {
  if (!capture) return;
  cancelAnimationFrame(capture.animationFrame);
  capture.stream.getTracks().forEach((track) => track.stop());
  capture.processor?.disconnect();
  capture.source?.disconnect();
  capture.silent?.disconnect();
  capture.context?.close();
}

function animateMeter(capture) {
  const samples = new Float32Array(capture.analyser.fftSize);
  const tick = () => {
    capture.analyser.getFloatTimeDomainData(samples);
    const rms = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
    drawLevel(rms);
    capture.animationFrame = requestAnimationFrame(tick);
  };
  tick();
}

async function startRecording() {
  if (state.recordingStarting || state.capture || state.savingTake) return;
  if (!state.variantId) return showToast("Choose a voice variant before recording.", "error");
  if (!state.prompt) return showToast("Wait for a prompt before recording.", "error");
  discardTake();
  state.recordingStarting = true;
  renderVariantContext();
  $("#record-button").disabled = true;
  $("#record-state").textContent = "Opening microphone…";
  try {
    const session = await ensureRecordingSession();
    if (!session) throw new Error("A recording session is required before capture can begin.");
    const chunks = [];
    const capture = await createCapture((samples) => chunks.push(samples));
    capture.chunks = chunks;
    capture.startedAt = performance.now();
    state.capture = capture;
    renderVariantContext();
    animateMeter(capture);
    $("#record-button").classList.add("recording");
    $("#record-button").setAttribute("aria-label", "Stop recording");
    $("#record-state").textContent = "Recording… press again to stop";
    $("#record-hint").textContent = "Raw mono WAV · local only";
    $("#skip-prompt").disabled = true;
  } catch (error) {
    showToast(`Microphone unavailable: ${error.message}`, "error");
  } finally {
    state.recordingStarting = false;
    $("#record-button").disabled = false;
    renderVariantContext();
  }
}

function stopRecording() {
  const capture = state.capture;
  if (!capture) return;
  closeCapture(capture);
  state.capture = null;
  const encoded = encodeWav(capture.chunks, capture.sampleRate);
  const url = URL.createObjectURL(encoded.blob);
  state.take = { ...encoded, url, sourceSampleRate: capture.sampleRate };
  renderVariantContext();
  $("#record-button").classList.remove("recording");
  $("#record-button").setAttribute("aria-label", "Start recording");
  $("#record-state").textContent = encoded.seconds < 2 ? "That take is very short" : "Take captured locally";
  $("#record-hint").textContent = "Listen before saving";
  $("#skip-prompt").disabled = false;
  $("#take-preview").hidden = false;
  $("#take-audio").src = url;
  $("#take-duration").textContent = secondsLabel(encoded.seconds);
  $("#take-qa").textContent = encoded.seconds < 2 ? "Too short" : "Ready to review";
  $("#take-qa").className = `qa-pill ${encoded.seconds < 2 ? "quarantined" : "pending"}`;
  drawIdleMeter();
}

function discardTake() {
  if (state.take?.url) URL.revokeObjectURL(state.take.url);
  state.take = null;
  renderVariantContext();
  $("#take-audio").removeAttribute("src");
  $("#take-audio").load();
  $("#take-preview").hidden = true;
  $("#record-state").textContent = "Ready when you are";
  $("#record-hint").textContent = "Press the red button to begin";
}

async function saveTake() {
  if (state.savingTake || !state.take || !state.prompt) return;
  const session = currentVariantSession();
  if (!state.variantId || !session?.id) return showToast("This take is not attached to a voice variant session.", "error");
  const button = $("#save-take");
  state.savingTake = true;
  button.disabled = true;
  button.textContent = "Checking locally…";
  try {
    const form = new FormData();
    form.append("audio", state.take.blob, `${safeId(state.prompt.id)}.wav`);
    form.append("prompt_id", state.prompt.id);
    form.append("prompt_kind", state.prompt.kind);
    form.append("expected_text", state.prompt.text);
    form.append("expected_transcript", state.prompt.text);
    form.append("sample_rate", String(state.take.sampleRate));
    form.append("bit_depth", String(state.take.bitDepth));
    form.append("duration_seconds", String(state.take.seconds));
    form.append("session_id", session.id);
    if (state.prompt.split) form.append("split", state.prompt.split);
    await api("/recordings", { method: "POST", body: form });
    state.session.takes += 1;
    state.session.seconds += state.take.seconds;
    if (state.promptKind === "custom") {
      if (Number.isFinite(state.customScriptPendingCount)) {
        state.customScriptPendingCount = Math.max(0, state.customScriptPendingCount - 1);
      }
    } else {
      state.session[state.prompt.kind === "interview" ? "interview" : "guided"] += 1;
    }
    updateSession();
    saveCurrentSession();
    showToast("Take saved. Local quality checks are running.");
    discardTake();
    await Promise.allSettled([loadPrompt(), refreshStatus()]);
  } catch (error) {
    showToast(`Could not save take: ${error.message}`, "error");
  } finally {
    state.savingTake = false;
    button.disabled = false;
    button.textContent = "Save take";
    renderVariantContext();
  }
}

function isRecordShortcutTargetEditable(target) {
  if (!target || target === document || target === document.body || target === document.documentElement) return false;
  if (!(target instanceof Element)) return false;
  if (target.closest("#record-button, #save-take, #discard-take")) return false;
  return Boolean(target.closest("input, select, textarea, button, audio, [contenteditable]:not([contenteditable='false'])"));
}

function handleRecordShortcut(event) {
  if (
    state.view !== "record"
    || event.defaultPrevented
    || event.repeat
    || event.isComposing
    || event.metaKey
    || event.ctrlKey
    || event.altKey
    || event.shiftKey
    || isRecordShortcutTargetEditable(event.target)
  ) return;

  let action = null;
  if (event.code === "Space" || event.key === " ") {
    if (state.capture) action = stopRecording;
    else if (
      !state.take
      && !state.recordingStarting
      && !state.savingTake
      && state.prompt
      && !$("#record-button").disabled
    ) action = startRecording;
  } else if (event.key === "Enter") {
    if (state.take && !state.capture && !state.savingTake && !$("#save-take").disabled) action = saveTake;
  } else if (event.key === "Escape") {
    if (state.take && !state.capture && !state.savingTake) action = discardTake;
  }

  if (!action) return;
  event.preventDefault();
  action();
}

async function checkMicrophone() {
  if (state.capture) return;
  $("#monitor-button").disabled = true;
  try {
    const capture = await createCapture(() => {});
    animateMeter(capture);
    $("#record-state").textContent = "Checking input for 3 seconds…";
    window.setTimeout(() => {
      closeCapture(capture);
      drawIdleMeter();
      $("#record-state").textContent = "Microphone is ready";
      $("#monitor-button").disabled = false;
    }, 3000);
  } catch (error) {
    $("#monitor-button").disabled = false;
    showToast(`Microphone unavailable: ${error.message}`, "error");
  }
}

function recordingStatus(recording) {
  return String(recording.status || recording.review_status || (recording.approved ? "approved" : "pending")).toLowerCase();
}

function recordingDuration(recording) {
  return Number(recording.duration_seconds || recording.duration || 0);
}

function recordingAudio(recording) {
  const value = recording.audio_url || recording.url;
  if (value?.startsWith(`${API}/`)) return value.slice(API.length);
  return value || variantPath(`/recordings/${encodeURIComponent(recording.id)}/audio`);
}

async function loadRecordings() {
  if (!state.variantId) {
    state.recordings = [];
    renderRecordings();
    return;
  }
  const variant = state.variantId;
  try {
    const data = await api(variantPath("/recordings"));
    if (state.variantId !== variant) return;
    state.recordings = normalizeList(data, ["recordings", "items"]).filter(belongsToActiveVariant);
    renderRecordings();
  } catch (error) {
    if (state.variantId !== variant) return;
    if (!state.recordings.length) renderRecordings();
    showToast(`Could not refresh recordings: ${error.message}`, "error");
  }
}

function qaFlags(recording) {
  const qa = recording.qa || recording.qa_json || recording.quality || {};
  const audio = qa.audio || qa;
  const failures = new Set(audio.failures || qa.failures || []);
  const warnings = new Set(audio.warnings || qa.warnings || []);
  const flags = [];
  const namedChecks = [
    ["Low signal-to-noise", failures.has("snr_too_low"), true],
    ["Too much silence", failures.has("too_much_silence"), true],
    ["Clipping", failures.has("peak_too_hot"), true],
    ["Too quiet", failures.has("level_too_quiet"), true],
    ["Quiet level", warnings.has("level_quiet"), true],
    ["Extra silence", warnings.has("extra_silence"), true],
  ];
  const values = [
    ["Clipping", qa.clipping, true], ["Noise", qa.noise || qa.snr_failed, true], ["Silence", qa.silence_failed, true],
    ["Needs review", recording.qa_needs_review === true, true], ["Transcript match", qa.transcript_match === false || recording.transcript_status === "mismatch", true],
    ["Level good", qa.level_ok, false], ["Clean signal", qa.passed === true || recording.qa_passed === true, false],
  ];
  for (const [label, value, warning] of namedChecks) if (value) flags.push({ label, warning });
  for (const [label, value, warning] of values) if (value) flags.push({ label, warning });
  if (!flags.length) flags.push({ label: qa.pending ? "Checks running" : "QA available after analysis", warning: Boolean(qa.pending) });
  return flags;
}

function visibleRecordings() {
  return state.recordings.filter((recording) => {
    if (state.reviewFilter === "all") return true;
    const status = recordingStatus(recording);
    if (state.reviewFilter === "quarantined") return ["quarantined", "needs_attention", "rejected"].includes(status);
    return status === state.reviewFilter;
  });
}

function recordingRequiresPlayback(recording) {
  const transcriptStatus = String(recording.transcript_status || "").toLowerCase();
  return Boolean(recording.qa_needs_review)
    || ["pending", "review_required"].includes(transcriptStatus);
}

function recordingPlaybackCompleted(recording) {
  return Boolean(recording.playback_confirmed)
    || state.playedBackRecordings.has(recording.id);
}

function updateSelectionToolbar() {
  const count = state.selectedRecordings.size;
  $("#selected-recording-count").textContent = `${count} selected`;
  for (const id of ["approve-selected-recordings", "quarantine-selected-recordings", "reject-selected-recordings"]) {
    $(`#${id}`).disabled = count === 0;
  }
  $("#clear-recording-selection").disabled = count === 0;
  $("#select-visible-recordings").disabled = visibleRecordings().length === 0;
}

function renderRecordings() {
  const list = $("#recording-list");
  const statuses = state.recordings.map(recordingStatus);
  const approved = statuses.filter((status) => status === "approved").length;
  const attention = statuses.filter((status) => status === "quarantined" || status === "rejected" || status === "needs_attention").length;
  const approvedSeconds = state.recordings.filter((item) => recordingStatus(item) === "approved").reduce((sum, item) => sum + recordingDuration(item), 0);
  $("#review-total").textContent = state.recordings.length;
  $("#review-approved").textContent = approved;
  $("#review-needs-attention").textContent = attention;
  $("#review-approved-time").textContent = secondsLabel(approvedSeconds);
  $("#review-count").hidden = !state.recordings.length;
  $("#review-count").textContent = state.recordings.length;
  const validIds = new Set(state.recordings.map((recording) => recording.id));
  state.selectedRecordings = new Set([...state.selectedRecordings].filter((id) => validIds.has(id)));
  state.playedBackRecordings = new Set([...state.playedBackRecordings].filter((id) => validIds.has(id)));
  const filtered = visibleRecordings();
  updateSelectionToolbar();
  list.replaceChildren();
  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = `<span class="empty-icon" aria-hidden="true">◌</span><h2>${state.recordings.length ? "Nothing in this filter" : "No takes to review yet"}</h2><p>${state.recordings.length ? "Choose another filter to see your recordings." : "Record your first prompt and it will appear here with local quality checks."}</p>`;
    const button = document.createElement("button");
    button.className = "button primary";
    button.type = "button";
    button.textContent = "Start recording";
    button.addEventListener("click", () => setView("record"));
    empty.append(button);
    list.append(empty);
    return;
  }
  for (const recording of filtered) list.append(makeRecordingCard(recording));
}

function makeRecordingCard(recording) {
  const card = document.createElement("article");
  card.className = `recording-card ${state.selectedRecordings.has(recording.id) ? "selected" : ""}`;
  const status = recordingStatus(recording);
  const selectLabel = document.createElement("label");
  selectLabel.className = "recording-select";
  selectLabel.title = "Select this take";
  const select = document.createElement("input");
  select.type = "checkbox";
  select.checked = state.selectedRecordings.has(recording.id);
  select.setAttribute("aria-label", `Select take: ${recording.prompt_text || recording.id}`);
  select.addEventListener("change", () => {
    if (select.checked) state.selectedRecordings.add(recording.id);
    else state.selectedRecordings.delete(recording.id);
    card.classList.toggle("selected", select.checked);
    updateSelectionToolbar();
  });
  selectLabel.append(select);
  const audio = new Audio();
  let audioObjectUrl = null;
  let playedBack = recordingPlaybackCompleted(recording);
  const requiresPlayback = recordingRequiresPlayback(recording);
  const play = document.createElement("button");
  play.type = "button"; play.className = "play-take"; play.textContent = "▶"; play.setAttribute("aria-label", "Play take");
  play.addEventListener("click", async () => {
    try {
      if (!audio.src) {
        const blob = await api(recordingAudio(recording), { accept: "audio/wav" });
        audioObjectUrl = URL.createObjectURL(blob);
        audio.src = audioObjectUrl;
      }
      if (audio.paused) { await audio.play(); play.textContent = "❚❚"; } else { audio.pause(); play.textContent = "▶"; }
    } catch (error) {
      showToast(`Could not play take: ${error.message}`, "error");
    }
  });
  audio.addEventListener("ended", () => {
    play.textContent = "▶";
    playedBack = true;
    state.playedBackRecordings.add(recording.id);
    if (status !== "approved") approve.disabled = false;
    if (audioObjectUrl) URL.revokeObjectURL(audioObjectUrl);
    audioObjectUrl = null;
    audio.removeAttribute("src");
  });
  const copy = document.createElement("div");
  copy.className = "recording-copy";
  const quote = document.createElement("blockquote");
  quote.textContent = recording.expected_text || recording.prompt_text || recording.text || "Untitled recording";
  const meta = document.createElement("small");
  meta.textContent = `${recording.kind || recording.prompt_kind || "guided"} · ${secondsLabel(recordingDuration(recording))} · ${recording.created_at ? new Date(recording.created_at).toLocaleString() : "this session"}`;
  const transcript = document.createElement("input");
  transcript.className = "transcript-input";
  transcript.value = state.transcriptEdits[recording.id] ?? recording.confirmed_transcript ?? recording.asr_transcript ?? recording.transcript ?? "";
  transcript.placeholder = "Confirm the local transcript";
  transcript.setAttribute("aria-label", "Confirmed transcript");
  transcript.addEventListener("input", () => { state.transcriptEdits[recording.id] = transcript.value; });
  const flags = document.createElement("div");
  flags.className = "qa-flags";
  qaFlags(recording).forEach((flag) => {
    const chip = document.createElement("span"); chip.className = `qa-flag ${flag.warning ? "warn" : ""}`; chip.textContent = flag.label; flags.append(chip);
  });
  copy.append(quote, meta, transcript, flags);
  const actions = document.createElement("div");
  actions.className = "recording-actions";
  const badge = document.createElement("span");
  badge.className = `qa-pill ${status === "approved" ? "approved" : status === "pending" ? "pending" : "quarantined"}`;
  badge.textContent = status.replaceAll("_", " ");
  const quarantine = document.createElement("button");
  quarantine.type = "button"; quarantine.className = "button secondary"; quarantine.textContent = "Quarantine";
  quarantine.disabled = status === "quarantined";
  quarantine.addEventListener("click", () => reviewRecording(recording.id, "quarantined", transcript.value, playedBack));
  const approve = document.createElement("button");
  approve.type = "button"; approve.className = "button primary"; approve.textContent = status === "approved" ? "Approved" : "Approve";
  approve.disabled = status === "approved" || (requiresPlayback && !playedBack);
  if (requiresPlayback && !playedBack) approve.title = "Play the complete take before approval";
  approve.addEventListener("click", () => reviewRecording(recording.id, "approved", transcript.value, playedBack));
  actions.append(badge, quarantine, approve);
  card.append(selectLabel, play, copy, actions);
  return card;
}

async function reviewRecording(id, status, transcript, playbackConfirmed = false) {
  if (!transcript.trim() && status === "approved") return showToast("Confirm the transcript before approving.", "error");
  if (!state.variantId) return showToast("Choose a voice variant first.", "error");
  const variant = state.variantId;
  try {
    await api(`/recordings/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: {
        variant_id: variant,
        status,
        transcript: transcript.trim(),
        transcript_confirmed: Boolean(transcript.trim()),
        playback_confirmed: playbackConfirmed,
      },
    });
    if (state.variantId !== variant) return;
    state.selectedRecordings.delete(id);
    delete state.transcriptEdits[id];
    showToast(status === "approved" ? "Take approved." : "Take quarantined.");
    await Promise.allSettled([loadRecordings(), refreshStatus()]);
  } catch (error) {
    if (state.variantId !== variant) return;
    showToast(`Could not update take: ${error.message}`, "error");
  }
}

function selectVisibleRecordings() {
  for (const recording of visibleRecordings()) state.selectedRecordings.add(recording.id);
  renderRecordings();
}

function clearRecordingSelection() {
  state.selectedRecordings.clear();
  renderRecordings();
}

async function bulkReviewRecordings(status) {
  const selectedIds = [...state.selectedRecordings];
  if (!selectedIds.length) return;
  if (!state.variantId) return showToast("Choose a voice variant first.", "error");
  const variant = state.variantId;
  const verb = status === "approved" ? "approve" : status === "quarantined" ? "quarantine" : "reject";
  const selectedRows = state.recordings.filter((recording) => state.selectedRecordings.has(recording.id));
  const unplayedRows = status === "approved"
    ? selectedRows.filter((recording) => recordingRequiresPlayback(recording) && !recordingPlaybackCompleted(recording))
    : [];
  const unplayedIds = new Set(unplayedRows.map((recording) => recording.id));
  const requestRows = status === "approved"
    ? selectedRows.filter((recording) => !unplayedIds.has(recording.id))
    : selectedRows;
  const recordingIds = requestRows.map((recording) => recording.id);
  if (!recordingIds.length) {
    showToast(
      `Nothing approved. Play ${unplayedRows.length === 1 ? "the selected uncertain take" : `all ${unplayedRows.length} selected uncertain takes`} to the end first. Quarantine and Reject remain available.`,
      "error",
    );
    return;
  }
  const failedQaCount = requestRows.filter((recording) => recording.qa_passed === false).length;
  const warning = status === "approved"
    ? `This confirms you accept ${recordingIds.length} recordings and their displayed transcripts.${failedQaCount ? ` ${failedQaCount} are QA-flagged and will be explicitly overridden.` : ""}${unplayedRows.length ? ` ${unplayedRows.length} uncertain or manually transcribed ${unplayedRows.length === 1 ? "take remains" : "takes remain"} selected until played to the end.` : ""}`
    : status === "rejected"
      ? "Rejected takes remain stored locally but will be excluded from training."
      : "Quarantined takes remain stored locally and excluded from training.";
  if (!window.confirm(`${verb[0].toUpperCase()}${verb.slice(1)} ${recordingIds.length} selected takes?\n\n${warning}`)) return;
  const transcripts = Object.fromEntries(
    recordingIds
      .filter((id) => state.transcriptEdits[id]?.trim())
      .map((id) => [id, state.transcriptEdits[id].trim()])
  );
  try {
    const result = await api("/recordings/bulk", {
      method: "POST",
      body: {
        variant_id: variant,
        recording_ids: recordingIds,
        status,
        playback_confirmed_ids: status === "approved"
          ? requestRows.filter(recordingPlaybackCompleted).map((recording) => recording.id)
          : [],
        confirm_playback: false,
        confirm_transcripts: status === "approved",
        override_failed_qa: status === "approved",
        transcripts,
      },
    });
    if (state.variantId !== variant) return;
    state.selectedRecordings = new Set([
      ...unplayedRows.map((recording) => recording.id),
      ...(result.skipped || []).map((item) => item.id),
    ]);
    for (const id of result.updated_ids || []) delete state.transcriptEdits[id];
    const overridden = Number(result.overridden_count || 0);
    const remaining = state.selectedRecordings.size;
    showToast(`${verb[0].toUpperCase()}${verb.slice(1)}d ${result.updated_count || 0} takes${overridden ? ` (${overridden} QA overrides recorded)` : ""}${remaining ? `; ${remaining} remain selected${unplayedRows.length ? " because playback is still required" : ""}` : ""}.`);
    await Promise.allSettled([loadRecordings(), refreshStatus()]);
  } catch (error) {
    if (state.variantId !== variant) return;
    showToast(`Could not ${verb} selected takes: ${error.message}`, "error");
    renderRecordings();
  }
}

function statusNumber(...values) {
  for (const value of values) if (Number.isFinite(Number(value))) return Number(value);
  return 0;
}

function renderModelSetup(setup = {}) {
  const status = state.status || {};
  const modelReady = Boolean(setup.models?.qwen?.ready ?? status.model_ready ?? status.model?.ready);
  const ready = typeof setup.ready === "boolean" ? setup.ready : modelReady;
  const active = Boolean(setup.active || setup.status === "running" || setup.status === "cancelling");
  const qwenActive = active && (setup.component || "qwen") === "qwen";
  const cancelling = setup.status === "cancelling" || state.cancellingModelSetup === "qwen";
  const failed = setup.status === "failed";
  const cancelled = setup.status === "cancelled";
  const overlay = $("#model-setup-overlay");
  overlay.hidden = ready;
  $(".app-shell").inert = !ready;
  document.body.classList.toggle("setup-required", !ready);
  renderSpeechChecks(setup);
  if (ready) {
    $("#cancel-model-setup").hidden = true;
    if (active && !state.setupPoll) state.setupPoll = window.setTimeout(refreshModelSetup, 1000);
    if (!active) {
      window.clearTimeout(state.setupPoll);
      state.setupPoll = null;
    }
    return;
  }
  const progress = Math.max(0, Math.min(1, Number(setup.progress) || 0));
  $("#model-setup-progress").hidden = !qwenActive;
  $("#model-setup-message").textContent = setup.message || "Preparing model download…";
  $("#model-setup-percent").textContent = `${Math.round(progress * 100)}%`;
  $("#model-setup-bar").style.width = `${progress * 100}%`;
  $("#model-setup-phase").textContent = qwenActive
    ? cancelling
      ? "Stopping the installer and removing this download's partial files…"
      : "Keep Noice City open. Download time depends on your connection."
    : "The models are downloaded only once and remain private on this Mac.";
  $("#model-setup-error").hidden = !failed;
  $("#model-setup-error").textContent = failed ? setup.error || "Model setup failed. Check your connection and try again." : "";
  const button = $("#start-model-setup");
  button.disabled = qwenActive;
  button.textContent = qwenActive
    ? cancelling ? "Canceling Qwen3-TTS download…" : "Downloading Qwen3-TTS…"
    : failed || cancelled
      ? "Restart Qwen3-TTS download"
      : "Download Qwen3-TTS · about 4.5 GB";
  const cancelButton = $("#cancel-model-setup");
  cancelButton.hidden = !qwenActive;
  cancelButton.disabled = cancelling;
  cancelButton.textContent = cancelling ? "Canceling download…" : "Cancel Qwen download";
  if (active && !state.setupPoll) {
    state.setupPoll = window.setTimeout(refreshModelSetup, 1000);
  } else if (!active) {
    window.clearTimeout(state.setupPoll);
    state.setupPoll = null;
  }
}

async function refreshModelSetup() {
  window.clearTimeout(state.setupPoll);
  state.setupPoll = null;
  try {
    const setup = await api("/model-setup");
    state.status.model_setup = setup;
    renderModelSetup(setup);
    if (setup.ready) await refreshStatus();
  } catch (error) {
    renderModelSetup({ status: "failed", error: error.message });
  }
}

function openModelDownloadConsent(component) {
  const disclosure = MODEL_DOWNLOAD_DISCLOSURES[component];
  if (!disclosure) return;
  state.pendingModelDownload = component;
  $("#model-consent-title").textContent = `Download ${disclosure.name}?`;
  $("#model-consent-intro").textContent = disclosure.intro;
  $("#model-consent-repository").textContent = disclosure.repository;
  $("#model-consent-revision").textContent = disclosure.revision;
  $("#model-consent-license").textContent = disclosure.license;
  $("#model-consent-size").textContent = disclosure.size;
  $("#confirm-model-consent").textContent = `Confirm ${disclosure.size} download`;
  $("#model-consent-dialog").showModal();
}

function closeModelDownloadConsent() {
  state.pendingModelDownload = null;
  $("#model-consent-dialog").close();
}

async function confirmModelDownload() {
  const component = state.pendingModelDownload;
  if (!MODEL_DOWNLOAD_DISCLOSURES[component]) return;
  state.pendingModelDownload = null;
  $("#model-consent-dialog").close();
  if (component === "qwen") await startModelSetup();
  else await installSpeechChecks();
}

async function startModelSetup() {
  const button = $("#start-model-setup");
  button.disabled = true;
  try {
    const setup = await api("/model-setup", { method: "POST", body: { component: "qwen" } });
    state.status.model_setup = setup;
    renderModelSetup(setup);
  } catch (error) {
    renderModelSetup({ status: "failed", error: error.message });
  }
}

function whisperFeaturesReady(setup = state.status?.model_setup || {}) {
  const status = state.status || {};
  return Boolean(setup.models?.whisper?.ready ?? setup.speech_checks_ready ?? status.asr?.ready);
}

function renderSpeechChecks(setup = {}) {
  const ready = whisperFeaturesReady(setup);
  const active = Boolean(
    (setup.active || setup.status === "running" || setup.status === "cancelling")
    && setup.component === "whisper"
  );
  const cancelling = active && (setup.status === "cancelling" || state.cancellingModelSetup === "whisper");
  const reviewLabel = $("#speech-checks-state");
  const trainingLabel = $("#training-speech-checks-state");
  const buttons = [$("#install-speech-checks"), $("#install-training-speech-checks")].filter(Boolean);
  if (reviewLabel) reviewLabel.textContent = ready
    ? "Whisper installed: automatic transcripts and WER checks are ready"
    : active
      ? cancelling ? "Canceling the Whisper download safely…" : "Installing optional Whisper features…"
      : "Core audio checks are active without Whisper; transcripts stay manual";
  if (trainingLabel) trainingLabel.textContent = ready
    ? "Whisper is installed for automatic WER and checkpoint evaluation."
    : active
      ? cancelling ? "Canceling the Whisper download safely…" : "Installing Whisper for automatic WER and checkpoint evaluation…"
      : "Whisper is optional for recording and generation, but required before training because every checkpoint runs automatic transcript/WER evaluation.";
  for (const button of buttons) {
    button.hidden = ready || active;
    button.disabled = active;
    button.textContent = active
      ? "Installing Whisper…"
      : button.id === "install-training-speech-checks"
        ? "Install Whisper for checkpoint evaluation · 1.6 GB"
        : "Install Whisper for automatic transcripts · 1.6 GB";
  }
  const cancelButtons = [$("#cancel-speech-checks"), $("#cancel-training-speech-checks")].filter(Boolean);
  for (const button of cancelButtons) {
    button.hidden = !active;
    button.disabled = cancelling;
    button.textContent = cancelling ? "Canceling Whisper download…" : "Cancel Whisper download";
  }
  updateTrainButton();
  updateRuntimeEstimate();
}

async function installSpeechChecks() {
  const buttons = [$("#install-speech-checks"), $("#install-training-speech-checks")].filter(Boolean);
  for (const button of buttons) button.disabled = true;
  try {
    const setup = await api("/model-setup", { method: "POST", body: { component: "whisper" } });
    state.status.model_setup = setup;
    renderModelSetup(setup);
  } catch (error) {
    showToast(`Could not install Whisper features: ${error.message}`, "error");
    for (const button of buttons) button.disabled = false;
  }
}

async function cancelModelSetup(component) {
  if (!MODEL_DOWNLOAD_DISCLOSURES[component] || state.cancellingModelSetup) return;
  state.cancellingModelSetup = component;
  const current = state.status?.model_setup || { component, active: true };
  renderModelSetup({ ...current, component, active: true, status: "cancelling" });
  try {
    const setup = await api("/model-setup/cancel", {
      method: "POST",
      body: { component },
    });
    state.status.model_setup = setup;
    showToast(`${MODEL_DOWNLOAD_DISCLOSURES[component].name} download canceled. Partial files were removed.`);
    renderModelSetup(setup);
  } catch (error) {
    showToast(`Could not cancel the model download: ${error.message}`, "error");
    await refreshModelSetup();
  } finally {
    state.cancellingModelSetup = null;
    renderModelSetup(state.status?.model_setup || {});
  }
}

function renderStatus() {
  const status = state.status || {};
  const model = status.model || {};
  const modelReady = Boolean(status.model_ready ?? model.ready ?? status.ready);
  const system = $("#system-status");
  $(".status-light", system).className = `status-light ${modelReady ? "" : "error"}`;
  $("strong", system).textContent = modelReady ? "Model ready" : "Model unavailable";
  $("small", system).textContent = "127.0.0.1 only";
  renderModelSetup(status.model_setup || {});

  const datasetStatus = status.dataset || status.collection || {};
  const acceptedSeconds = statusNumber(datasetStatus.accepted_seconds, status.accepted_seconds, datasetStatus.approved_seconds);
  updateCustomScriptGate(
    acceptedSeconds,
    typeof status.custom_script_unlocked === "boolean" ? status.custom_script_unlocked : undefined,
    statusNumber(status.custom_script_gate_seconds, acceptedSeconds),
  );
  const acceptedMinutes = acceptedSeconds / 60;
  const acceptedDisplay = acceptedMinutes < 10 ? acceptedMinutes.toFixed(1) : Math.round(acceptedMinutes);
  $("#accepted-minutes").textContent = acceptedDisplay;
  $("#collection-target-label").textContent = `${acceptedDisplay} of 30 minutes`;
  $("#collection-ring").style.setProperty("--progress", `${Math.min(360, acceptedMinutes / 30 * 360)}deg`);
  const readiness = Math.min(100, Math.round(acceptedMinutes / 30 * 100));
  $("#readiness-score").textContent = `${readiness}%`;
  $("#readiness-bar").style.width = `${readiness}%`;
  $("#readiness-title").textContent = acceptedMinutes >= 30 ? "Ready for full training" : acceptedMinutes >= 2 ? "Pilot training is available" : "Collecting recordings";
  $("#check-pilot").classList.toggle("complete", acceptedMinutes >= 2);
  $("#check-pilot > span").textContent = acceptedMinutes >= 2 ? "✓" : "○";
  $("#check-full").classList.toggle("complete", acceptedMinutes >= 30);
  $("#check-full > span").textContent = acceptedMinutes >= 30 ? "✓" : "○";
  const kinds = datasetStatus.kinds || {};
  const hasMix = Boolean((kinds.guided || datasetStatus.guided_count) && (kinds.interview || datasetStatus.interview_count));
  $("#check-split").classList.toggle("complete", hasMix);
  $("#check-split > span").textContent = hasMix ? "✓" : "○";

  state.datasets = normalizeList(status.datasets || datasetStatus.versions, ["datasets", "versions"])
    .filter((item) => belongsToVariant(item, state.variantId));
  state.checkpoints = normalizeList(status.checkpoints || status.adapters, ["checkpoints", "items", "adapters"])
    .filter((item) => belongsToVariant(item, state.variantId));
  renderDatasets();
  renderCheckpoints();
}

function datasetId(dataset) { return typeof dataset === "string" ? dataset : dataset.id || dataset.version || dataset.dataset_version; }
function checkpointId(checkpoint) { return typeof checkpoint === "string" ? checkpoint : checkpoint.id || checkpoint.version || checkpoint.checkpoint; }

function generationCheckpointLabel(id) {
  const checkpoint = state.checkpoints.find((item) => checkpointId(item) === id);
  if (!checkpoint || typeof checkpoint === "string") return id;
  const metrics = checkpoint.metrics_json || checkpoint.metrics || {};
  const label = checkpoint.label || checkpoint.name || metrics.label || id;
  return checkpoint.epoch ? `${label}, epoch ${checkpoint.epoch}` : label;
}

function datasetCreatedDate(dataset) {
  if (typeof dataset !== "string") {
    const raw = dataset.created_at || dataset.createdAt || dataset.frozen_at;
    if (raw) {
      const parsed = new Date(raw);
      if (!Number.isNaN(parsed.getTime())) return parsed;
    }
  }
  const source = typeof dataset === "string" ? dataset : dataset.name || "";
  const match = source.match(/(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})/);
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match;
  const parsed = new Date(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second));
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function datasetTimestamp(dataset) {
  const created = datasetCreatedDate(dataset);
  if (!created) return "";
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(created);
}

function checkpointCreatedDate(checkpoint) {
  if (!checkpoint || typeof checkpoint === "string") return null;
  const raw = checkpoint.created_at || checkpoint.createdAt || checkpoint.saved_at;
  if (raw) {
    const parsed = new Date(raw);
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }
  const match = checkpointId(checkpoint).match(/checkpoint_(\d{8})T(\d{6})/);
  if (!match) return null;
  const date = match[1];
  const time = match[2];
  const parsed = new Date(Date.UTC(
    Number(date.slice(0, 4)), Number(date.slice(4, 6)) - 1, Number(date.slice(6, 8)),
    Number(time.slice(0, 2)), Number(time.slice(2, 4)), Number(time.slice(4, 6)),
  ));
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function checkpointTimestamp(checkpoint) {
  const created = checkpointCreatedDate(checkpoint);
  if (!created) return "";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(created);
}

function checkpointVersion(checkpoint) {
  const id = checkpointId(checkpoint);
  const suffix = id.split("_").at(-1) || id;
  return suffix.slice(0, 8);
}

function datasetDisplayLabel(dataset) {
  const name = typeof dataset === "string" ? dataset : dataset.label || dataset.name || datasetId(dataset);
  const timestamp = datasetTimestamp(dataset);
  return timestamp ? `${name} · frozen ${timestamp}` : name;
}

function updateDatasetVersionMeta() {
  const meta = $("#dataset-version-meta");
  const selected = state.datasets.find((dataset) => datasetId(dataset) === $("#dataset-version").value);
  if (!selected) {
    meta.hidden = true;
    meta.textContent = "";
    return;
  }
  const timestamp = datasetTimestamp(selected);
  const details = [];
  if (timestamp) details.push(`Frozen ${timestamp}`);
  if (typeof selected !== "string" && Number.isFinite(Number(selected.recording_count))) {
    details.push(`${Number(selected.recording_count)} takes`);
  }
  if (typeof selected !== "string" && Number.isFinite(Number(selected.duration_seconds))) {
    details.push(`${secondsLabel(selected.duration_seconds)} audio`);
  }
  meta.textContent = details.join(" · ");
  meta.hidden = !details.length;
}

function evaluationGate() {
  let active = Boolean(state.status?.active_evaluation_checkpoint);
  let awaitingReview = false;
  for (const checkpoint of state.checkpoints) {
    if (typeof checkpoint === "string") continue;
    const metrics = checkpoint.metrics_json || checkpoint.metrics || {};
    const evaluationStatus = String(metrics.evaluation_status || "").toLowerCase();
    const hasComparison = Boolean(metrics.comparison_manifest);
    if (evaluationStatus === "awaiting_blind_review" || (evaluationStatus === "pending" && hasComparison)) {
      awaitingReview = true;
    } else if (["queued", "running"].includes(evaluationStatus) || (evaluationStatus === "pending" && !hasComparison)) {
      active = true;
    }
  }
  return { active, awaitingReview };
}

function syncEvaluationPolling() {
  const { active } = evaluationGate();
  if (!active) {
    window.clearTimeout(state.evaluationPoll);
    state.evaluationPoll = null;
    return;
  }
  if (state.evaluationPoll) return;
  state.evaluationPoll = window.setTimeout(async () => {
    state.evaluationPoll = null;
    await refreshStatus();
  }, 2000);
}

function renderDatasets() {
  const select = $("#dataset-version");
  const current = select.value;
  select.replaceChildren(new Option("No frozen dataset", ""));
  for (const dataset of state.datasets) {
    const id = datasetId(dataset);
    select.append(new Option(datasetDisplayLabel(dataset), id));
  }
  if (current && [...select.options].some((option) => option.value === current)) select.value = current;
  else if (state.datasets.length) select.value = datasetId(state.datasets[0]);
  updateDatasetVersionMeta();
  updateTrainButton();
  updateRuntimeEstimate();
}

function updateRuntimeEstimate() {
  const gate = evaluationGate();
  if (gate.active) {
    $("#runtime-estimate").textContent = "Fresh-process checkpoint evaluation is running. Training is locked until it finishes.";
    return;
  }
  if (gate.awaitingReview) {
    $("#runtime-estimate").textContent = "Finish the blinded checkpoint review before starting another training run.";
    return;
  }
  if (!whisperFeaturesReady()) {
    $("#runtime-estimate").textContent = "Install Whisper before training so the saved checkpoint can run automatic transcript/WER evaluation.";
    return;
  }
  const selected = state.datasets.find((dataset) => datasetId(dataset) === $("#dataset-version").value);
  if (!selected) {
    $("#runtime-estimate").textContent = "Choose a frozen dataset for a local runtime estimate.";
    return;
  }
  const epochs = $("#training-stage").value === "full" ? 3 : 1;
  const recordings = Number(selected.recording_count || selected.items || 1);
  const midpoint = Math.max(8, Math.round(recordings * epochs * 0.12));
  const low = Math.max(5, Math.round(midpoint * .65));
  const high = Math.max(low + 5, Math.round(midpoint * 1.7));
  $("#runtime-estimate").textContent = `Estimated training time on this Mac: ${low}–${high} minutes, followed by fresh-process evaluation.`;
}

function renderCheckpoints() {
  const list = $("#checkpoint-list");
  const select = $("#generation-checkpoint");
  const previous = select.value;
  const variant = state.variantId;
  const owner = variantName();
  const checkpoints = state.checkpoints.filter((checkpoint) => belongsToVariant(checkpoint, variant));
  state.checkpoints = checkpoints;
  list.replaceChildren();
  select.replaceChildren(new Option("No trained checkpoint", ""));
  if (!checkpoints.length) list.innerHTML = `<div class="small-empty">No checkpoints for ${owner} yet.</div>`;
  for (const checkpoint of checkpoints) {
    const id = checkpointId(checkpoint);
    const metrics = checkpoint.metrics_json || checkpoint.metrics || {};
    const label = typeof checkpoint === "string" ? checkpoint : checkpoint.label || checkpoint.name || metrics.label || id;
    const evaluationStatus = metrics.evaluation_status || "pending";
    const item = document.createElement("div");
    item.className = `checkpoint-item ${checkpoint.promoted || checkpoint.status === "promoted" ? "promoted" : ""}`;
    const top = document.createElement("div");
    const name = document.createElement("strong"); name.textContent = label;
    const stateLabel = document.createElement("small"); stateLabel.textContent = checkpoint.promoted || checkpoint.status === "promoted" ? "Promoted" : checkpoint.status || "Saved";
    top.append(name, stateLabel);
    const identity = document.createElement("div");
    identity.className = "checkpoint-identity";
    const ownerBadge = document.createElement("span");
    ownerBadge.className = "checkpoint-owner";
    ownerBadge.textContent = owner;
    const version = document.createElement("span");
    version.className = "checkpoint-version";
    const timestamp = checkpointTimestamp(checkpoint);
    version.textContent = `Version ${checkpointVersion(checkpoint)}${timestamp ? ` · saved ${timestamp}` : ""}`;
    identity.append(ownerBadge, version);
    const evaluation = metrics.evaluation || {};
    const meta = document.createElement("p");
    meta.textContent = `${metrics.stage || checkpoint.stage || "checkpoint"} · evaluation ${evaluationStatus.replaceAll("_", " ")}${evaluation.wer != null ? ` · WER ${(Number(evaluation.wer) * 100).toFixed(1)}%` : ""}`;
    item.append(top, identity, meta);
    if (evaluationStatus === "awaiting_blind_review" || evaluationStatus === "pending") {
      if (metrics.comparison_manifest) {
        const review = document.createElement("button");
        review.type = "button"; review.className = "button secondary checkpoint-review"; review.textContent = "Review blinded A/B";
        review.addEventListener("click", () => openBlindReview(id));
        item.append(review);
      }
    } else if (evaluationStatus === "failed") {
      const retry = document.createElement("button");
      retry.type = "button"; retry.className = "button secondary checkpoint-review"; retry.textContent = "Retry local evaluation";
      retry.addEventListener("click", () => retryEvaluation(id));
      item.append(retry);
    }
    list.append(item);
    if (checkpoint.status !== "rejected") {
      select.append(new Option(`${owner} — ${label} · ${checkpointVersion(checkpoint)} · ${checkpoint.status === "promoted" ? "promoted" : "candidate"}`, id));
    }
  }
  if (previous && [...select.options].some((option) => option.value === previous)) {
    select.value = previous;
  } else {
    const promoted = checkpoints.find((checkpoint) => checkpoint.status === "promoted");
    const promotedId = promoted ? checkpointId(promoted) : "";
    if (promotedId && [...select.options].some((option) => option.value === promotedId)) select.value = promotedId;
    else if (select.options.length > 1) select.selectedIndex = 1;
  }
  updateGenerateButton();
}

async function retryEvaluation(checkpointId) {
  try {
    await api(variantPath(`/checkpoints/${encodeURIComponent(checkpointId)}/evaluations`), { method: "POST", body: {} });
    showToast("Fresh-process evaluation queued.");
    await refreshStatus();
  } catch (error) {
    showToast(`Could not queue evaluation: ${error.message}`, "error");
  }
}

async function openBlindReview(checkpointId) {
  try {
    const comparison = await api(variantPath(`/checkpoints/${encodeURIComponent(checkpointId)}/comparison`));
    stopBlindAudio();
    state.blind = {
      checkpointId,
      comparison,
      index: 0,
      audio: null,
      audioUrl: null,
      audioOption: null,
      audioRequestId: state.blind.audioRequestId + 1,
    };
    const firstMissing = comparison.pairs.findIndex((pair) => !comparison.votes?.[pair.prompt_id]);
    state.blind.index = firstMissing >= 0 ? firstMissing : 0;
    $("#blind-review").hidden = false;
    renderBlindPair();
  } catch (error) {
    showToast(`Could not open blinded review: ${error.message}`, "error");
  }
}

function renderBlindPair() {
  const { comparison, index } = state.blind;
  if (!comparison) return;
  stopBlindAudio();
  const pair = comparison.pairs[index];
  $("#blind-position").textContent = `${index + 1} of ${comparison.pairs.length}`;
  $("#blind-text").textContent = pair.text;
  $$(`[data-blind-choice]`).forEach((button) => {
    button.classList.toggle("primary", comparison.votes?.[pair.prompt_id] === button.dataset.blindChoice);
  });
}

function updateBlindPlaybackControls() {
  for (const option of ["A", "B"]) {
    const button = $(`#play-option-${option.toLowerCase()}`);
    const isCurrent = state.blind.audioOption === option && Boolean(state.blind.audio);
    const isPlaying = isCurrent && !state.blind.audio.paused && !state.blind.audio.ended;
    const canResume = isCurrent && state.blind.audio.paused && state.blind.audio.currentTime > 0;
    button.disabled = false;
    button.textContent = isPlaying ? `Pause clip ${option}` : canResume ? `Resume clip ${option}` : `Play clip ${option}`;
    button.setAttribute("aria-pressed", String(isPlaying));
    button.classList.toggle("playing", isPlaying);
  }
}

function stopBlindAudio(message = "Nothing playing. You can stop as soon as you know.") {
  state.blind.audioRequestId = (state.blind.audioRequestId || 0) + 1;
  if (state.blind.audio) {
    state.blind.audio.pause();
    state.blind.audio.removeAttribute("src");
    state.blind.audio.load();
  }
  if (state.blind.audioUrl) URL.revokeObjectURL(state.blind.audioUrl);
  state.blind.audio = null;
  state.blind.audioUrl = null;
  state.blind.audioOption = null;
  updateBlindPlaybackControls();
  const status = $("#blind-playback-status");
  if (status) status.textContent = message;
}

async function playBlindOption(option) {
  const pair = state.blind.comparison?.pairs[state.blind.index];
  if (!pair) return;
  if (state.blind.audioOption === option && state.blind.audio) {
    if (state.blind.audio.paused) {
      try {
        await state.blind.audio.play();
        $("#blind-playback-status").textContent = `Playing clip ${option}. Click again to pause.`;
      } catch (error) {
        showToast(`Could not resume option ${option}: ${error.message}`, "error");
      }
    } else {
      state.blind.audio.pause();
      $("#blind-playback-status").textContent = `Clip ${option} paused. Click again to resume.`;
    }
    updateBlindPlaybackControls();
    return;
  }

  stopBlindAudio();
  const requestId = state.blind.audioRequestId;
  const path = option === "A" ? pair.option_a_audio : pair.option_b_audio;
  const button = $(`#play-option-${option.toLowerCase()}`);
  button.disabled = true;
  button.textContent = `Loading clip ${option}…`;
  $("#blind-playback-status").textContent = `Loading clip ${option} locally…`;
  try {
    const blob = await api(path.startsWith(API) ? path.slice(API.length) : path, { accept: "audio/wav" });
    if (requestId !== state.blind.audioRequestId) return;
    const audioUrl = URL.createObjectURL(blob);
    const audio = new Audio(audioUrl);
    state.blind.audioUrl = audioUrl;
    state.blind.audio = audio;
    state.blind.audioOption = option;
    audio.addEventListener("play", updateBlindPlaybackControls);
    audio.addEventListener("pause", updateBlindPlaybackControls);
    audio.addEventListener("ended", () => {
      if (state.blind.audio === audio) stopBlindAudio(`Clip ${option} finished. Replay it or choose the voice that sounds more like you.`);
    }, { once: true });
    await audio.play();
    if (requestId !== state.blind.audioRequestId) {
      audio.pause();
      URL.revokeObjectURL(audioUrl);
      return;
    }
    $("#blind-playback-status").textContent = `Playing clip ${option}. Click again to pause.`;
    updateBlindPlaybackControls();
  } catch (error) {
    if (requestId !== state.blind.audioRequestId) return;
    stopBlindAudio();
    showToast(`Could not play option ${option}: ${error.message}`, "error");
  }
}

async function submitBlindVote(choice) {
  const pair = state.blind.comparison?.pairs[state.blind.index];
  if (!pair) return;
  stopBlindAudio("Audio stopped. Saving your voice preference…");
  $$(`[data-blind-choice]`).forEach((button) => { button.disabled = true; });
  try {
    const result = await api(variantPath(`/checkpoints/${encodeURIComponent(state.blind.checkpointId)}/blind-votes`), {
      method: "POST",
      body: { prompt_id: pair.prompt_id, choice },
    });
    const persistedVotes = result.checkpoint?.metrics_json?.blind_votes
      || result.checkpoint?.metrics?.blind_votes;
    if (persistedVotes && typeof persistedVotes === "object") {
      state.blind.comparison.votes = { ...persistedVotes };
    } else {
      state.blind.comparison.votes[pair.prompt_id] = choice;
    }
    if (result.decision.status === "promoted") {
      showToast("Checkpoint passed every gate and was promoted.");
      $("#blind-review").hidden = true;
      await refreshStatus();
      return;
    }
    if (result.decision.status === "rejected") {
      showToast("Candidate did not pass every gate; the prior promoted model was preserved.", "error");
      $("#blind-review").hidden = true;
      await refreshStatus();
      return;
    }
    const nextMissing = state.blind.comparison.pairs.findIndex((item) => !state.blind.comparison.votes[item.prompt_id]);
    state.blind.index = nextMissing >= 0 ? nextMissing : Math.min(state.blind.index + 1, 19);
    renderBlindPair();
  } catch (error) {
    showToast(`Could not save blinded vote: ${error.message}`, "error");
  } finally {
    $$(`[data-blind-choice]`).forEach((button) => { button.disabled = false; });
  }
}

async function refreshStatus() {
  if (!state.variantId) return;
  const variant = state.variantId;
  const epoch = state.variantEpoch;
  const requestId = ++state.statusRequestId;
  try {
    const [statusResult, datasetsResult, checkpointsResult] = await Promise.allSettled([
      api(variantPath("/status")), api(variantPath("/datasets")), api(variantPath("/checkpoints")),
    ]);
    if (statusResult.status === "rejected") throw statusResult.reason;
    if (state.variantId !== variant || state.variantEpoch !== epoch || state.statusRequestId !== requestId) return;
    state.status = { ...statusResult.value };
    if (datasetsResult.status === "fulfilled") {
      state.status.datasets = normalizeList(datasetsResult.value, ["datasets", "versions", "items"])
        .filter((item) => belongsToVariant(item, variant));
    } else {
      state.status.datasets = normalizeList(state.status.datasets, ["datasets", "versions", "items"])
        .filter((item) => belongsToVariant(item, variant));
    }
    if (checkpointsResult.status === "fulfilled") {
      state.status.checkpoints = normalizeList(checkpointsResult.value, ["checkpoints", "items", "adapters"])
        .filter((item) => belongsToVariant(item, variant));
    } else {
      state.status.checkpoints = normalizeList(state.status.checkpoints, ["checkpoints", "items", "adapters"])
        .filter((item) => belongsToVariant(item, variant));
    }
    renderStatus();
    syncEvaluationPolling();
  } catch (error) {
    if (state.variantId !== variant || state.variantEpoch !== epoch || state.statusRequestId !== requestId) return;
    const system = $("#system-status");
    $(".status-light", system).className = "status-light error";
    $("strong", system).textContent = "Local service unavailable";
    $("small", system).textContent = "Check the app server";
    if (state.view !== "record") showToast(`Could not read local status: ${error.message}`, "error");
  }
}

function updateTrainButton() {
  const button = $("#start-training");
  const gate = evaluationGate();
  const speechChecksReady = whisperFeaturesReady();
  const hasDataset = Boolean($("#dataset-version").value);
  button.disabled = !state.variantId || !hasDataset || Boolean(state.trainingRunId) || gate.active || gate.awaitingReview || !speechChecksReady;
  button.textContent = gate.active
    ? "Evaluation running…"
    : gate.awaitingReview
      ? "Finish blinded review"
      : !hasDataset
        ? "Choose a frozen dataset"
        : !speechChecksReady
          ? "Install Whisper above to train"
          : "Start local training";
}

async function freezeDataset() {
  if (!state.variantId) return showToast("Choose a voice variant before freezing a dataset.", "error");
  const variant = state.variantId;
  const button = $("#freeze-dataset");
  button.disabled = true; button.textContent = "Freezing…";
  try {
    const frozen = await api("/datasets/freeze", { method: "POST", body: { variant_id: state.variantId } });
    if (state.variantId !== variant) return;
    const timestamp = datasetTimestamp(frozen);
    showToast(timestamp ? `Dataset frozen — ${timestamp}.` : `Dataset ${datasetId(frozen) || "version"} frozen.`);
    await refreshStatus();
    const frozenId = datasetId(frozen);
    if ([...$("#dataset-version").options].some((option) => option.value === frozenId)) {
      $("#dataset-version").value = frozenId;
      updateDatasetVersionMeta();
      updateTrainButton();
      updateRuntimeEstimate();
    }
  } catch (error) {
    if (state.variantId !== variant) return;
    showToast(`Could not freeze dataset: ${error.message}`, "error");
  } finally {
    button.disabled = false; button.textContent = "Freeze current dataset";
  }
}

async function startTraining() {
  const dataset = $("#dataset-version").value;
  if (!state.variantId) return showToast("Choose a voice variant before training.", "error");
  if (!dataset) return;
  if (!whisperFeaturesReady()) {
    showToast("Install Whisper first. Noice City requires it for automatic transcript/WER and checkpoint evaluation after training.", "error");
    return;
  }
  const variant = state.variantId;
  const gate = evaluationGate();
  if (gate.active || gate.awaitingReview) {
    showToast(gate.active ? "Wait for checkpoint evaluation to finish." : "Finish the blinded checkpoint review first.", "error");
    return;
  }
  const stage = $("#training-stage").value;
  try {
    const run = await api("/training-runs", {
      method: "POST",
      body: {
        variant_id: variant,
        dataset_version: dataset,
        stage,
        label: $("#checkpoint-label").value.trim() || null,
        epochs: stage === "full" ? 3 : 1,
      },
    });
    if (state.variantId !== variant) return;
    state.trainingRunId = run.id || run.run_id;
    if (!state.trainingRunId) throw new Error("The local worker did not return a run ID.");
    state.trainingVariantId = state.variantId;
    showTrainingProgress({ status: "queued", progress: 0, message: "Queued on the local MLX worker" });
    pollTraining();
  } catch (error) {
    if (state.variantId !== variant) return;
    showToast(`Training did not start: ${error.message}`, "error");
  }
}

function showTrainingProgress(run) {
  const status = String(run.status || run.state || "running").toLowerCase();
  const done = ["completed", "complete", "failed", "cancelled", "canceled"].includes(status);
  const failed = ["failed", "cancelled", "canceled"].includes(status);
  const rawProgress = Number(run.progress_percent ?? run.progress ?? 0) || 0;
  const progress = Math.max(0, Math.min(100, rawProgress <= 1 ? rawProgress * 100 : rawProgress));
  $("#training-progress").hidden = false;
  $("#training-options").hidden = !done;
  $("#training-state").textContent = status;
  $("#training-state").className = `training-state ${done ? (failed ? "failed" : "complete") : "running"}`;
  $("#training-title").textContent = done ? (failed ? "Run stopped" : "Checkpoint saved") : "Training on this Mac";
  $("#training-step").textContent = run.message || run.current_step || (done ? status : "Training speaker adaptation…");
  $("#training-percent").textContent = `${Math.round(progress)}%`;
  $("#training-bar").style.width = `${progress}%`;
  $("#training-epoch").textContent = run.epoch ? `Epoch ${run.epoch}${run.total_epochs ? ` of ${run.total_epochs}` : ""}` : "Epoch —";
  $("#training-loss").textContent = run.loss != null ? `Loss ${Number(run.loss).toFixed(4)}` : "Loss —";
  $("#training-elapsed").textContent = run.elapsed_seconds ? `Elapsed ${secondsLabel(run.elapsed_seconds)}` : "Elapsed —";
  $("#cancel-training").hidden = done;
  $("#start-training").disabled = true;
  if (done) {
    state.trainingRunId = null;
    state.trainingVariantId = null;
    window.clearTimeout(state.trainingPoll);
    state.trainingPoll = null;
  }
}

async function pollTraining() {
  if (!state.trainingRunId) return;
  const runId = state.trainingRunId;
  const variant = state.trainingVariantId;
  try {
    const run = await api(variantPath(`/training-runs/${encodeURIComponent(runId)}`));
    if (state.trainingRunId !== runId || state.variantId !== variant) return;
    showTrainingProgress(run);
    if (state.trainingRunId) state.trainingPoll = window.setTimeout(pollTraining, 2000);
    else {
      showToast(String(run.status).toLowerCase().startsWith("comp") ? "Training complete. Checkpoint preserved." : "Training run ended.");
      await refreshStatus();
    }
  } catch (error) {
    if (state.trainingRunId !== runId || state.variantId !== variant) return;
    showToast(`Training status unavailable: ${error.message}`, "error");
    state.trainingPoll = window.setTimeout(pollTraining, 5000);
  }
}

async function cancelTraining() {
  if (!state.trainingRunId) return;
  try {
    await api(variantPath(`/training-runs/${encodeURIComponent(state.trainingRunId)}`), { method: "DELETE" });
    showToast("Cancellation requested.");
  } catch (error) {
    showToast(`Could not cancel training: ${error.message}`, "error");
  }
}

function updateGenerateButton() {
  const seedState = generationSeedState();
  const ready = state.variantId && $("#generation-text").value.trim() && $("#generation-checkpoint").value && seedState.valid;
  $("#generate-button").disabled = !ready;
}

function generationSeedState(showError) {
  const enabled = $("#generation-reproducible").checked;
  const input = $("#generation-seed");
  const error = $("#generation-seed-error");
  input.disabled = !enabled;
  if (!enabled) {
    input.removeAttribute("aria-invalid");
    error.hidden = true;
    $("#generation-seed-help").textContent = "A random seed will be used. Results can vary even when the text, checkpoint, and delivery stay the same.";
    return { enabled: false, valid: true, seed: null };
  }

  $("#generation-seed-help").textContent = "Seed 3407 matches the local evaluation starting seed. Text, checkpoint, and delivery must also stay the same.";
  const value = input.value.trim();
  const seed = Number(value);
  const valid = value !== "" && Number.isInteger(seed) && seed >= 0 && seed <= MAX_GENERATION_SEED;
  input.setAttribute("aria-invalid", String(!valid));
  if (valid) error.hidden = true;
  else if (showError === true) error.hidden = false;
  else if (showError === false) error.hidden = true;
  return { enabled: true, valid, seed };
}

function updateGenerationSeedControl() {
  generationSeedState(true);
  updateGenerateButton();
}

async function generateSpeech() {
  const text = $("#generation-text").value.trim();
  const checkpoint = $("#generation-checkpoint").value;
  if (!state.variantId) return showToast("Choose a voice variant before generating speech.", "error");
  if (!text || !checkpoint) return;
  const seedState = generationSeedState(true);
  if (!seedState.valid) {
    $("#generation-seed").focus();
    updateGenerateButton();
    return;
  }
  const delivery = $("#generation-delivery").value;
  const variant = state.variantId;
  const epoch = state.variantEpoch;
  const payload = { variant_id: variant, text, delivery, checkpoint_version: checkpoint };
  if (seedState.enabled) payload.seed = seedState.seed;
  const button = $("#generate-button");
  button.disabled = true; button.innerHTML = "Generating locally…";
  try {
    const result = await api("/generations", {
      method: "POST",
      accept: "audio/wav, application/json",
      body: payload,
    });
    if (state.variantId !== variant || state.variantEpoch !== epoch) return;
    if (state.outputUrl?.startsWith("blob:")) URL.revokeObjectURL(state.outputUrl);
    if (result instanceof Blob) state.outputUrl = URL.createObjectURL(result);
    else state.outputUrl = result.audio_url || result.url || `${API}/generations/${encodeURIComponent(result.id || result.generation_id)}/audio`;
    $("#generated-audio").src = state.outputUrl;
    $("#download-output").href = state.outputUrl;
    const deliveryLabel = $("#generation-delivery").selectedOptions[0]?.textContent || delivery;
    const seedLabel = seedState.enabled ? `Fixed seed ${seedState.seed}` : "Random seed";
    $("#output-name").textContent = `${variantName()} · Generated speech`;
    $("#download-output").download = `${safeId(variantName().toLowerCase().replaceAll(" ", "-")) || "voice"}-generated.wav`;
    $("#output-meta").textContent = `${generationCheckpointLabel(checkpoint)} · ${deliveryLabel} · ${seedLabel} · Local WAV${result.duration_seconds ? ` · ${secondsLabel(result.duration_seconds)}` : ""}`;
    $("#output-placeholder").hidden = true;
    $("#generated-output").hidden = false;
    showToast("Speech generated entirely on this Mac.");
  } catch (error) {
    if (state.variantId !== variant || state.variantEpoch !== epoch) return;
    showToast(`Generation failed: ${error.message}`, "error");
  } finally {
    button.innerHTML = '<span aria-hidden="true">▶</span> Generate speech';
    updateGenerateButton();
  }
}

function bindEvents() {
  for (const selector of ["#variant-select", "#mobile-variant-select"]) {
    $(selector).addEventListener("change", (event) => switchVariant(event.target.value));
  }
  for (const selector of ["#manage-variants", "#manage-variants-mobile"]) {
    $(selector).addEventListener("click", () => {
      renderVariantManager();
      $("#variant-dialog").showModal();
    });
  }
  $("#close-variant-dialog").addEventListener("click", () => $("#variant-dialog").close());
  $("#start-model-setup").addEventListener("click", () => openModelDownloadConsent("qwen"));
  $("#cancel-model-setup").addEventListener("click", () => cancelModelSetup("qwen"));
  $("#install-speech-checks").addEventListener("click", () => openModelDownloadConsent("whisper"));
  $("#install-training-speech-checks").addEventListener("click", () => openModelDownloadConsent("whisper"));
  $("#cancel-speech-checks").addEventListener("click", () => cancelModelSetup("whisper"));
  $("#cancel-training-speech-checks").addEventListener("click", () => cancelModelSetup("whisper"));
  $("#close-model-consent").addEventListener("click", closeModelDownloadConsent);
  $("#cancel-model-consent").addEventListener("click", closeModelDownloadConsent);
  $("#confirm-model-consent").addEventListener("click", confirmModelDownload);
  $("#model-consent-dialog").addEventListener("close", () => { state.pendingModelDownload = null; });
  $("#model-consent-dialog").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeModelDownloadConsent();
  });
  $("#variant-dialog").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) event.currentTarget.close();
  });
  $("#variant-create-form").addEventListener("submit", createVariant);
  $("#new-recording-session").addEventListener("click", newRecordingSession);
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
  $$(`[data-go-to]`).forEach((item) => item.addEventListener("click", () => setView(item.dataset.goTo)));
  $$("[data-prompt-kind]").forEach((button) => button.addEventListener("click", () => selectPromptKind(button.dataset.promptKind)));
  $("#custom-script-text").addEventListener("input", updateCustomScriptSubmit);
  $("#save-custom-script").addEventListener("click", saveCustomScript);
  $("#skip-prompt").addEventListener("click", skipPrompt);
  $("#record-button").addEventListener("click", () => state.capture ? stopRecording() : startRecording());
  $("#monitor-button").addEventListener("click", checkMicrophone);
  $("#discard-take").addEventListener("click", discardTake);
  $("#save-take").addEventListener("click", saveTake);
  $("#refresh-recordings").addEventListener("click", loadRecordings);
  $("#select-visible-recordings").addEventListener("click", selectVisibleRecordings);
  $("#clear-recording-selection").addEventListener("click", clearRecordingSelection);
  $("#approve-selected-recordings").addEventListener("click", () => bulkReviewRecordings("approved"));
  $("#quarantine-selected-recordings").addEventListener("click", () => bulkReviewRecordings("quarantined"));
  $("#reject-selected-recordings").addEventListener("click", () => bulkReviewRecordings("rejected"));
  $$("[data-review-filter]").forEach((button) => button.addEventListener("click", () => {
    state.reviewFilter = button.dataset.reviewFilter;
    $$("[data-review-filter]").forEach((item) => item.classList.toggle("active", item === button));
    renderRecordings();
  }));
  $("#freeze-dataset").addEventListener("click", freezeDataset);
  $("#dataset-version").addEventListener("change", () => { updateDatasetVersionMeta(); updateTrainButton(); updateRuntimeEstimate(); });
  $("#training-stage").addEventListener("change", updateRuntimeEstimate);
  $("#start-training").addEventListener("click", startTraining);
  $("#cancel-training").addEventListener("click", cancelTraining);
  $("#generation-text").addEventListener("input", (event) => {
    $("#character-count").textContent = event.target.value.length;
    updateGenerateButton();
  });
  $("#generation-checkpoint").addEventListener("change", updateGenerateButton);
  $("#generation-reproducible").addEventListener("change", updateGenerationSeedControl);
  $("#generation-seed").addEventListener("input", updateGenerationSeedControl);
  $("#generate-button").addEventListener("click", generateSpeech);
  $("#play-option-a").addEventListener("click", () => playBlindOption("A"));
  $("#play-option-b").addEventListener("click", () => playBlindOption("B"));
  $$(`[data-blind-choice]`).forEach((button) => button.addEventListener("click", () => submitBlindVote(button.dataset.blindChoice)));
  document.addEventListener("keydown", handleRecordShortcut);
  window.addEventListener("hashchange", () => setView(location.hash.slice(1) || "record"));
  window.addEventListener("beforeunload", () => {
    saveCurrentSession();
    closeCapture(state.capture);
    if (state.take?.url) URL.revokeObjectURL(state.take.url);
    if (state.outputUrl?.startsWith("blob:")) URL.revokeObjectURL(state.outputUrl);
    stopBlindAudio();
    window.clearTimeout(state.evaluationPoll);
    window.clearTimeout(state.setupPoll);
  });
}

async function init() {
  bindEvents();
  drawIdleMeter();
  updateSession();
  updateGenerationSeedControl();
  renderVariantControls();
  await loadVariants();
  setView(location.hash.slice(1) || "record");
}

document.addEventListener("DOMContentLoaded", init);
