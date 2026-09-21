const API_BASE_URL = (window.API_BASE_URL || "").replace(/\/$/, "");
const invariantsDialog = document.querySelector("#invariants-dialog");
const invariantsForm = document.querySelector("#invariants-form");
const invariantFields = document.querySelector("#invariants-fields");
const invariantError = document.querySelector("#invariant-error");
const saveInvariantsButton = document.querySelector("#save-invariants");
let invariantSettings = null;
let invariantSaving = false;
const sessionList = document.querySelector("#sessions");
const messages = document.querySelector("#messages");
const title = document.querySelector("#chat-title");
const status = document.querySelector("#status");
const form = document.querySelector("#message-form");
const input = document.querySelector("#message-input");
const submitButton = form.querySelector("button[type='submit']");
const newButton = document.querySelector("#new-session");
const clearDatabaseButton = document.querySelector("#clear-database");
const profileSelect = document.querySelector("#profile-select");
const newProfileButton = document.querySelector("#new-profile");
const editProfileButton = document.querySelector("#edit-profile");
const activeProfileName = document.querySelector("#active-profile-name");
const activeProfileStatus = document.querySelector("#active-profile-status");
const activeProfileSummary = document.querySelector("#active-profile-summary");
const activeProfileConstraints = document.querySelector("#active-profile-constraints");
const profileDialog = document.querySelector("#profile-dialog");
const profileForm = document.querySelector("#profile-form");
const profileDialogTitle = document.querySelector("#profile-dialog-title");
const profileIdInput = document.querySelector("#profile-id");
const profileNameInput = document.querySelector("#profile-name");
const profileDescriptionInput = document.querySelector("#profile-description");
const profileLanguageInput = document.querySelector("#profile-language");
const profileToneInput = document.querySelector("#profile-tone");
const profileDetailInput = document.querySelector("#profile-detail-level");
const profileFormatInput = document.querySelector("#profile-response-format");
const profileConstraintsInput = document.querySelector("#profile-constraints-input");
const closeProfileDialogButton = document.querySelector("#close-profile-dialog");
const cancelProfileButton = document.querySelector("#cancel-profile");
const deleteProfileButton = document.querySelector("#delete-profile");
const commandMenu = document.querySelector("#command-menu");
const followupQueue = document.querySelector("#followup-queue");
const queuedMessagesContainer = document.querySelector("#queued-messages");
const queueCount = document.querySelector("#queue-count");
const shortTermMemory = document.querySelector("#short-term-memory");
const workingMemory = document.querySelector("#working-memory");
const longTermMemory = document.querySelector("#long-term-memory");
const shortTermCount = document.querySelector("#short-term-count");
const workingCount = document.querySelector("#working-count");
const longTermCount = document.querySelector("#long-term-count");
const startTaskButton = document.querySelector("#start-task");
const refreshButton = document.querySelector("#refresh-session");
const taskPanel = document.querySelector("#task-panel");
const taskOverview = document.querySelector("#task-overview");
const taskToolbar = document.querySelector("#task-toolbar");
const memoryView = document.querySelector("#memory-view");
const inspectorTabs = document.querySelector("#inspector-tabs");
const processTab = document.querySelector("#process-tab");
const memoryTab = document.querySelector("#memory-tab");
const showProcessButton = document.querySelector("#show-task-process");
const advanceButton = document.querySelector("#task-advance");
const pauseButton = document.querySelector("#task-pause");
const resumeButton = document.querySelector("#task-resume");
const replanButton = document.querySelector("#task-replan");
const STAGES = { planning: "Планирование", awaiting_approval: "Утверждение", execution: "Выполнение", validation: "Проверка", done: "Готово" };
const ACTIONS = { generate_plan: "Сформировать план", approve_plan: "Утвердить план", execute_step: "Выполнить шаг", validate: "Проверить результат", none: "Задача завершена" };
let currentSession = null;
let taskRunning = false;
let controlBusy = false;
let inspectorMode = "memory";
let renderedTaskSessionId = null;
let taskOperation = null;
let taskTimer = null;
let taskError = "";
let sessions = [];
let profiles = [];
let currentProfileId = null;
let currentSessionId = null;
let currentMessages = [];
let memorySnapshot = { working: [], long_term: [] };
let busy = false;
let activeCommandIndex = 0;
let queuedMessages = [];

const memoryCommands = [
  { name: "/goal", layer: "working", category: "goal", description: "цель текущей задачи" },
  { name: "/constraint", layer: "working", category: "constraint", description: "ограничение текущей задачи" },
  { name: "/decision", layer: "working", category: "decision", description: "решение текущей задачи" },
  { name: "/profile", layer: "long_term", category: "profile", description: "устойчивый факт профиля" },
  { name: "/preference", layer: "long_term", category: "preference", description: "предпочтение пользователя" },
  { name: "/knowledge", layer: "long_term", category: "knowledge", description: "знание для будущих чатов" },
];

const profileLabels = {
  tone: {
    neutral: "нейтральный",
    friendly: "дружелюбный",
    formal: "формальный",
    technical: "технический",
  },
  detail_level: {
    brief: "кратко",
    balanced: "сбалансированно",
    detailed: "подробно",
  },
  response_format: {
    plain: "обычный текст",
    bullets: "списки",
    steps: "пошагово",
  },
  language: { ru: "русский", en: "English" },
};

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail;
    const message = typeof detail === "string"
      ? detail
      : detail?.message || (Array.isArray(detail) ? detail.map((item) => item.msg).join("; ") : "");
    throw new Error(message || "Backend не смог выполнить запрос");
  }
  return response.status === 204 ? null : response.json();
}

function invariantPayload() {
  const minimum = Number(document.querySelector("#min-emojis").value);
  const maximum = Number(document.querySelector("#max-sentences").value);
  const bounded = (value, previous) => Number.isInteger(value) && value >= 1 && value <= 10 ? value : previous;
  const emojiEnabled = document.querySelector("#emoji-enabled").checked;
  const sentencesEnabled = document.querySelector("#sentence-limit-enabled").checked;
  return {
    emoji_enabled: emojiEnabled,
    min_emojis: emojiEnabled ? minimum : bounded(minimum, invariantSettings?.min_emojis || 3),
    uppercase_enabled: document.querySelector("#uppercase-enabled").checked,
    sentence_limit_enabled: sentencesEnabled,
    max_sentences: sentencesEnabled ? maximum : bounded(maximum, invariantSettings?.max_sentences || 3),
    revision: invariantSettings?.revision,
  };
}

function renderInvariantPreview() {
  const settings = invariantPayload();
  document.querySelector("#min-emojis").disabled = !settings.emoji_enabled;
  document.querySelector("#max-sentences").disabled = !settings.sentence_limit_enabled;
  const sentences = ["Небо кажется голубым.", "Атмосфера рассеивает солнечный свет.", "Синий свет рассеивается сильнее красного."];
  const count = settings.sentence_limit_enabled ? Math.max(1, settings.max_sentences || 1) : 3;
  let example = sentences.slice(0, count).join(" ");
  if (settings.uppercase_enabled) example = example.toUpperCase();
  if (settings.emoji_enabled) example += " " + [..."💬✨🙂✅🌟🔹🟢📌🎯💡"].slice(0, Math.min(10, Math.max(1, settings.min_emojis || 1))).join(" ");
  document.querySelector("#invariant-preview-text").textContent = example;
}

function renderActiveInvariants() {
  const container = document.querySelector("#active-invariants");
  container.replaceChildren();
  const settings = invariantSettings;
  if (!settings) {
    document.querySelector("#invariant-count").textContent = "?";
    return;
  }
  const rules = [];
  if (settings.emoji_enabled) rules.push([`🙂 ≥ ${settings.min_emojis}`, "Обязательные эмодзи"]);
  if (settings.uppercase_enabled) rules.push(["АБВ", "Только верхний регистр"]);
  if (settings.sentence_limit_enabled) rules.push([`≤ ${settings.max_sentences} предл.`, "Лимит предложений"]);
  document.querySelector("#invariant-count").textContent = String(rules.length);
  for (const [text, name] of rules) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.textContent = text;
    chip.title = `${name} — открыть инварианты`;
    chip.addEventListener("click", openInvariants);
    container.append(chip);
  }
}

async function loadInvariants() {
  invariantSettings = await api("/api/invariants");
  renderActiveInvariants();
}

async function openInvariants() {
  if (invariantSaving) return;
  invariantError.hidden = true;
  invariantFields.disabled = true;
  saveInvariantsButton.disabled = true;
  if (!invariantsDialog.open) invariantsDialog.showModal();
  try {
    await loadInvariants();
    const settings = invariantSettings;
    document.querySelector("#emoji-enabled").checked = settings.emoji_enabled;
    document.querySelector("#min-emojis").value = settings.min_emojis;
    document.querySelector("#uppercase-enabled").checked = settings.uppercase_enabled;
    document.querySelector("#sentence-limit-enabled").checked = settings.sentence_limit_enabled;
    document.querySelector("#max-sentences").value = settings.max_sentences;
    renderInvariantPreview();
    invariantFields.disabled = false;
    saveInvariantsButton.disabled = false;
  } catch (error) {
    invariantError.textContent = error.message;
    invariantError.hidden = false;
  }
}

function closeInvariants() {
  if (!invariantSaving) invariantsDialog.close();
}

document.querySelector("#open-invariants").addEventListener("click", openInvariants);
document.querySelector("#close-invariants").addEventListener("click", closeInvariants);
document.querySelector("#cancel-invariants").addEventListener("click", closeInvariants);
invariantsForm.addEventListener("input", renderInvariantPreview);
invariantsDialog.addEventListener("cancel", (event) => { if (invariantSaving) event.preventDefault(); });
invariantsDialog.addEventListener("click", (event) => { if (event.target === invariantsDialog) closeInvariants(); });
invariantsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (invariantSaving || !invariantsForm.reportValidity()) return;
  const payload = invariantPayload();
  invariantSaving = true;
  saveInvariantsButton.disabled = true;
  invariantFields.disabled = true;
  saveInvariantsButton.textContent = "Сохраняем…";
  invariantError.hidden = true;
  try {
    invariantSettings = await api("/api/invariants", { method: "PUT", body: JSON.stringify(payload) });
    renderActiveInvariants();
    invariantsDialog.close();
    status.textContent = "Инварианты сохранены";
  } catch (error) {
    invariantError.textContent = error.message;
    invariantError.hidden = false;
  } finally {
    invariantSaving = false;
    saveInvariantsButton.disabled = false;
    invariantFields.disabled = false;
    saveInvariantsButton.textContent = "Сохранить";
  }
});

function setBusy(value) {
  busy = value;
  submitButton.textContent = value ? "В очередь" : "Отправить";
  newButton.disabled = value || activeProfile()?.onboarding_complete === false;
  clearDatabaseButton.disabled = value;
  profileSelect.disabled = value;
  newProfileButton.disabled = value;
  editProfileButton.disabled = value;
  syncControls();
  if (value) status.textContent = readyStatus();
}

function syncControls() {
  const task = currentSession?.task;
  const active = task && task.state !== "done";
  const blocked = busy || controlBusy;
  input.disabled = Boolean(controlBusy || taskRunning || active);
  form.querySelector('button[type="submit"]').disabled = Boolean(controlBusy || taskRunning || active);
  startTaskButton.disabled = Boolean(blocked || task || !activeProfile()?.onboarding_complete);
  newButton.disabled = blocked || activeProfile()?.onboarding_complete === false;
  profileSelect.disabled = blocked;
  newProfileButton.disabled = blocked;
  editProfileButton.disabled = blocked;
  startTaskButton.title = activeProfile()?.onboarding_complete ? "Создать пошаговую задачу" : "Сначала завершите интервью в обычном чате или отправьте /skip";
  clearDatabaseButton.disabled = blocked;
  refreshButton.disabled = blocked;
  form.hidden = Boolean(active);
  taskToolbar.hidden = !active;
  input.placeholder = "Сообщение или /команда…";
  if (!task) return;
  const allowed = task.allowed_actions;
  const nextAction = task.expected_action === "approve_plan" ? "approve" : task.expected_action;
  advanceButton.textContent = taskRunning ? "Выполняется…" : ACTIONS[task.expected_action];
  advanceButton.hidden = task.state === "done";
  advanceButton.disabled = blocked || !allowed.includes(nextAction);
  pauseButton.hidden = !allowed.includes("pause");
  pauseButton.disabled = controlBusy || (busy && !taskRunning);
  resumeButton.hidden = !allowed.includes("resume");
  resumeButton.disabled = blocked;
  replanButton.hidden = !allowed.includes("replan");
  replanButton.disabled = blocked;
  document.querySelector("#task-note").textContent = task.paused
    ? (taskRunning ? "Пауза запрошена. Текущий ответ завершится и сохранится; следующий шаг не начнётся." : "Состояние сохранено. «Продолжить» снимет паузу, затем можно выполнить ожидаемое действие.")
    : (task.state === "done" ? "Итог сохранён в истории и результатах задачи. Для новой задачи создайте новый чат." : task.state === "awaiting_approval" ? "Выполнение заблокировано до утверждения плана. Можно утвердить план или указать изменения." : "Одно нажатие выполняет одно действие. Состояние сохраняется после каждого действия.");
  renderTaskActivity(task);
}

function readyStatus() {
  if (busy) return taskRunning ? (currentSession?.task?.paused ? "Завершаю ответ…" : "Выполняется…") : "Агент отвечает…";
  if (currentSession?.task?.paused) return "На паузе";
  return currentSession?.task?.state === "done" ? "Задача завершена" : "Готов";
}

function setInspector(mode) {
  inspectorMode = currentSession?.task ? mode : "memory";
  taskPanel.hidden = inspectorMode !== "process";
  memoryView.hidden = inspectorMode !== "memory";
  for (const [tab, name] of [[processTab, "process"], [memoryTab, "memory"]]) {
    tab.setAttribute("aria-selected", String(inspectorMode === name));
    tab.tabIndex = inspectorMode === name ? 0 : -1;
  }
}

function taskIcon(name) {
  const paths = {
    planning: '<path d="M8 5h12M8 12h12M8 19h12M3 5h.01M3 12h.01M3 19h.01"/>',
    awaiting_approval: '<path d="M9 4h6M9 3h6v4H9zM9 5H5v16h14V5h-4M8 14l3 3 5-6"/>',
    execution: '<path d="m8 5 10 7-10 7z"/>',
    validation: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    done: '<path d="m5 12 4 4L19 6"/>',
    paused: '<path d="M8 5v14M16 5v14"/>',
    error: '<path d="M12 5v9M12 19h.01"/>',
  };
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.planning}</svg>`;
}

function renderTask(task) {
  taskOverview.hidden = !task;
  inspectorTabs.hidden = !task;
  if (!task) {
    renderedTaskSessionId = null;
    taskError = "";
    setInspector("memory");
    return;
  }
  if (renderedTaskSessionId !== currentSessionId) {
    renderedTaskSessionId = currentSessionId;
    taskError = "";
    setInspector("process");
  }
  document.querySelector("#task-title").textContent = task.task;
  document.querySelector("#task-title").title = task.task;
  document.querySelector("#task-progress").textContent = task.total ? `${task.step} / ${task.total} шагов` : "Плана пока нет";
  const current = document.querySelector("#task-current");
  current.textContent = task.current;
  current.title = task.current;
  const stageNames = Object.keys(STAGES);
  document.querySelector("#task-stages").replaceChildren(...stageNames.map((name, index) => {
    const badge = document.createElement("span");
    badge.className = "task-stage";
    badge.dataset.stage = name;
    const completed = index < stageNames.indexOf(task.state);
    if (name === task.state) {
      badge.classList.add("active");
      badge.setAttribute("aria-current", "step");
    } else if (completed) badge.classList.add("completed");
    const mark = document.createElement("span");
    mark.className = "stage-mark";
    mark.innerHTML = taskIcon(completed ? "done" : name);
    const label = document.createElement("span");
    label.textContent = STAGES[name];
    badge.append(mark, label);
    return badge;
  }));
  document.querySelector("#task-plan-details").hidden = !task.plan.length;
  document.querySelector("#task-plan-empty").hidden = Boolean(task.plan.length);
  document.querySelector("#task-plan").replaceChildren(...task.plan.map((item, index) => {
    const li = document.createElement("li");
    li.className = "task-step";
    li.dataset.step = index;
    const mark = document.createElement("span");
    mark.className = "step-mark";
    if (index < task.step) {
      li.classList.add("completed");
      mark.innerHTML = taskIcon("done");
    } else mark.textContent = index + 1;
    if (index === task.step && task.state === "execution") {
      li.classList.add("current");
      li.setAttribute("aria-current", "step");
    }
    const body = document.createElement("div");
    const text = document.createElement("p");
    text.textContent = item;
    text.title = item;
    const state = document.createElement("small");
    state.className = "step-status";
    state.textContent = index < task.step ? "Готово" : "В плане";
    body.append(text, state);
    li.append(mark, body);
    return li;
  }));
  document.querySelector("#task-criteria").replaceChildren(...task.criteria.map((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    return li;
  }));
  const results = [
    ...task.previous_results.map((item) => ({ ...item, title: `Предыдущий план: ${item.title}` })),
    ...task.done,
  ];
  if (task.result) results.push({ title: "Итоговый результат", output: task.result });
  document.querySelector("#task-results-details").hidden = !results.length;
  document.querySelector("#task-results-summary").textContent = `Сохранённые результаты (${results.length})`;
  document.querySelector("#task-results").replaceChildren(...results.map((item) => {
    const detail = document.createElement("details");
    detail.className = "saved-result";
    const summary = document.createElement("summary");
    summary.textContent = item.title;
    const text = document.createElement("p");
    text.textContent = item.output;
    detail.append(summary, text);
    return detail;
  }));
  renderTaskActivity(task);
}

function elapsedTaskTime() {
  const seconds = Math.floor((performance.now() - taskOperation.startedAt) / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function updateTaskElapsed() {
  if (!taskOperation) return;
  const elapsed = elapsedTaskTime();
  document.querySelector("#task-elapsed").textContent = elapsed;
  const inlineTime = messages.querySelector(".task-wait-time");
  if (inlineTime) inlineTime.textContent = elapsed;
}

function renderTaskActivity(task) {
  if (!task) return;
  const running = Boolean(taskOperation);
  const activity = document.querySelector("#task-activity");
  activity.classList.toggle("running", running);
  activity.classList.toggle("paused", task.paused);
  activity.classList.toggle("failed", Boolean(taskError));
  const titles = {
    generate_plan: "Составляю план",
    execute_step: `Выполняю шаг ${(taskOperation?.step ?? task.step) + 1} из ${taskOperation?.total ?? task.total}`,
    validate: "Проверяю результат",
  };
  const current = document.querySelector("#task-current");
  current.textContent = running ? (task.paused ? "Завершаю ответ перед паузой" : titles[taskOperation.expectedAction]) : task.current;
  current.title = running ? taskOperation.current : task.current;
  document.querySelector("#task-activity-title").textContent = running
    ? (task.paused ? "Завершаю ответ перед паузой" : titles[taskOperation.expectedAction])
    : (taskError ? "Действие не завершено" : task.paused ? "Задача на паузе" : task.state === "done" ? "Задача завершена" : "Ожидается действие");
  document.querySelector("#task-expected").textContent = running
    ? (taskOperation.expectedAction === "execute_step" ? taskOperation.current : "Ожидаем ответ модели; результат будет сохранён.")
    : (taskError || (task.paused ? "План и результаты сохранены." : ACTIONS[task.expected_action]));
  activity.querySelector(".activity-icon").innerHTML = taskIcon(taskError ? "error" : task.paused ? "paused" : task.state);
  const elapsed = document.querySelector("#task-elapsed");
  elapsed.hidden = !running;
  if (running) elapsed.textContent = elapsedTaskTime();
  taskOverview.classList.toggle("running", running);
  taskOverview.classList.toggle("paused", task.paused);
  for (const step of document.querySelectorAll(".task-step.current")) {
    step.classList.toggle("running", running);
    step.classList.toggle("paused", task.paused);
    step.querySelector(".step-status").textContent = running ? "Выполняется" : task.paused ? "На паузе" : "Следующий шаг";
  }
  let pending = messages.querySelector(".task-pending");
  if (!running) {
    pending?.remove();
    return;
  }
  if (!pending) {
    const nearBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 80;
    pending = document.createElement("article");
    pending.className = "message assistant task-pending";
    const label = document.createElement("span");
    label.textContent = "Агент · выполняет задачу";
    const card = document.createElement("div");
    card.className = "task-pending-card";
    const heading = document.createElement("strong");
    heading.textContent = titles[taskOperation.expectedAction];
    const time = document.createElement("small");
    time.className = "task-wait-time";
    time.setAttribute("aria-label", "Время ожидания");
    time.setAttribute("aria-live", "off");
    const text = document.createElement("p");
    text.textContent = taskOperation.expectedAction === "execute_step" ? taskOperation.current : "Ответ появится здесь после завершения действия.";
    const signal = document.createElement("div");
    signal.className = "activity-scan";
    signal.setAttribute("aria-hidden", "true");
    card.append(heading, time, text, signal);
    pending.append(label, card);
    messages.append(pending);
    if (nearBottom) requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
  }
  pending.classList.toggle("paused", task.paused);
  updateTaskElapsed();
}

function startTaskActivity(task) {
  taskError = "";
  taskOperation = { expectedAction: task.expected_action, current: task.current, step: task.step, total: task.total, startedAt: performance.now() };
  taskTimer = setInterval(updateTaskElapsed, 1000);
}

function stopTaskActivity() {
  clearInterval(taskTimer);
  taskTimer = null;
  taskOperation = null;
  messages.querySelector(".task-pending")?.remove();
}

function applySession(session) {
  // A pause request and a generation can return out of order.
  if (currentSession?.id === session.id && currentSession.task && session.task
      && session.task.revision < currentSession.task.revision) return;
  currentSession = session;
  currentSessionId = session.id;
  const { messages: ignoredMessages, ...summary } = session;
  sessions = [summary, ...sessions.filter((item) => item.id !== session.id)];
  title.textContent = session.title;
  renderSessions();
  renderTask(session.task);
  syncControls();
  renderMessages(session.messages);
  status.textContent = readyStatus();
}

function activeProfile() {
  return profiles.find((profile) => profile.id === currentProfileId) || null;
}

function renderActiveProfile() {
  const profile = activeProfile();
  if (!profile) return;
  activeProfileName.textContent = profile.name;
  newButton.disabled = busy || !profile.onboarding_complete;
  newButton.title = profile.onboarding_complete
    ? "Создать новый чат"
    : "Сначала завершите интервью";
  editProfileButton.disabled = busy;
  activeProfileStatus.textContent = profile.onboarding_complete
    ? "PROFILE · READY"
    : `INTERVIEW · ${profile.onboarding_step}/3`;
  const settings = [
    profileLabels.language[profile.language] || profile.language,
    profileLabels.tone[profile.tone] || profile.tone,
    profileLabels.detail_level[profile.detail_level] || profile.detail_level,
    profileLabels.response_format[profile.response_format] || profile.response_format,
  ];
  activeProfileSummary.textContent = profile.onboarding_complete
    ? (profile.description
      ? `${profile.description} · ${settings.join(" · ")}`
      : settings.join(" · "))
    : "Агент ещё собирает профиль из диалога";
  activeProfileConstraints.replaceChildren(...profile.constraints.map((constraint) => {
    const tag = document.createElement("small");
    tag.textContent = constraint;
    return tag;
  }));
  syncControls();
}

function renderProfiles() {
  profileSelect.replaceChildren(...profiles.map((profile) => {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.onboarding_complete
      ? profile.name
      : `${profile.name} · интервью`;
    return option;
  }));
  if (currentProfileId) profileSelect.value = currentProfileId;
  renderActiveProfile();
}

function rememberSelectedProfile() {
  try {
    localStorage.setItem("rubik-active-profile", currentProfileId);
  } catch (_) {
    // Persistence is optional when storage is blocked by the browser.
  }
}

function openProfileDialog(profile) {
  if (busy || controlBusy) return;
  profileDialogTitle.textContent = profile.onboarding_complete
    ? "Проверка профиля"
    : "Интервью не завершено";
  profileIdInput.value = profile.id;
  profileNameInput.value = profile.name;
  profileDescriptionInput.value = profile.description;
  profileLanguageInput.value = profile.language;
  profileToneInput.value = profile.tone;
  profileDetailInput.value = profile.detail_level;
  profileFormatInput.value = profile.response_format;
  profileConstraintsInput.value = profile.constraints.join("\n");
  deleteProfileButton.hidden = profile.id === "default";
  profileForm.querySelector("button[type='submit']").disabled = !profile.onboarding_complete;
  profileDialog.showModal();
  profileNameInput.focus();
}

function closeProfileDialog() {
  profileDialog.close();
  profileForm.reset();
}

function profilePayload() {
  return {
    name: profileNameInput.value.trim(),
    description: profileDescriptionInput.value.trim(),
    language: profileLanguageInput.value,
    tone: profileToneInput.value,
    detail_level: profileDetailInput.value,
    response_format: profileFormatInput.value,
    constraints: profileConstraintsInput.value
      .split("\n")
      .map((constraint) => constraint.trim())
      .filter(Boolean),
  };
}

function renderQueue() {
  queueCount.textContent = String(queuedMessages.length);
  followupQueue.hidden = queuedMessages.length === 0;
  queuedMessagesContainer.replaceChildren(...queuedMessages.map((queued, index) => {
    const item = document.createElement("article");
    const position = document.createElement("span");
    position.textContent = String(index + 1);
    const content = document.createElement("p");
    content.textContent = queued.content;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.title = "Убрать из очереди";
    remove.setAttribute("aria-label", `Убрать из очереди: ${queued.content}`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      queuedMessages = queuedMessages.filter((item) => item.id !== queued.id);
      renderQueue();
      status.textContent = queuedMessages.length
        ? `В очереди: ${queuedMessages.length}`
        : "Очередь очищена";
    });
    item.append(position, content, remove);
    return item;
  }));
}

function enqueueMessage(content) {
  queuedMessages.push({
    id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
    content,
  });
  renderQueue();
  status.textContent = `Агент отвечает · в очереди: ${queuedMessages.length}`;
}

function commandMatches() {
  const value = input.value;
  if (!value.startsWith("/") || /\s/.test(value)) return [];
  return memoryCommands.filter((command) => command.name.startsWith(value.toLowerCase()));
}

function hideCommandMenu() {
  commandMenu.hidden = true;
  commandMenu.replaceChildren();
  input.removeAttribute("aria-activedescendant");
}

function selectCommand(command) {
  input.value = `${command.name} `;
  hideCommandMenu();
  input.focus();
}

function renderCommandMenu() {
  const matches = commandMatches();
  if (!matches.length) {
    hideCommandMenu();
    return;
  }

  activeCommandIndex = Math.min(activeCommandIndex, matches.length - 1);
  const options = matches.map((command, index) => {
    const option = document.createElement("button");
    option.type = "button";
    option.id = `memory-command-${index}`;
    option.className = index === activeCommandIndex ? "command-option active" : "command-option";
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", String(index === activeCommandIndex));

    const name = document.createElement("strong");
    name.textContent = command.name;
    const description = document.createElement("span");
    description.textContent = command.description;
    const layer = document.createElement("small");
    layer.className = command.layer;
    layer.textContent = command.layer === "working" ? "WORKING" : "LONG-TERM";
    option.append(name, description, layer);
    option.addEventListener("mousedown", (event) => {
      event.preventDefault();
      selectCommand(command);
    });
    return option;
  });

  commandMenu.replaceChildren(...options);
  commandMenu.hidden = false;
  input.setAttribute("aria-activedescendant", options[activeCommandIndex].id);
}

function parsedMemoryCommand(content) {
  const command = memoryCommands.find((candidate) => (
    content === candidate.name || content.startsWith(`${candidate.name} `)
  ));
  if (!command) return null;
  return { ...command, content: content.slice(command.name.length).trim() };
}

async function executeMemoryCommand(command) {
  if (!command.content) {
    status.textContent = `Добавь текст после ${command.name}`;
    input.focus();
    return;
  }

  setBusy(true);
  status.textContent = "Сохраняю команду в память…";
  try {
    await api("/api/memory", {
      method: "POST",
      body: JSON.stringify({
        layer: command.layer,
        category: command.category,
        content: command.content,
        session_id: command.layer === "working" ? currentSessionId : null,
        profile_id: command.layer === "long_term" ? currentProfileId : null,
        source_session_id: currentSessionId,
        source_text: `${command.name} ${command.content}`,
      }),
    });
    hideCommandMenu();
    const [session, updatedSessions, memory] = await Promise.all([
      api(`/api/chat/sessions/${currentSessionId}`),
      api(`/api/chat/sessions?profile_id=${encodeURIComponent(currentProfileId)}`),
      api(`/api/memory?session_id=${encodeURIComponent(currentSessionId)}`),
    ]);
    sessions = updatedSessions;
    memorySnapshot = memory;
    applySession(session);
    status.textContent = command.layer === "working"
      ? `Сохранено в рабочую память: ${command.category}`
      : `Сохранено в долговременную память: ${command.category}`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    setBusy(false);
    input.focus();
    void drainQueue();
  }
}

function renderSessions() {
  sessionList.replaceChildren(...sessions.map((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = session.id === currentSessionId ? "session active" : "session";
    button.textContent = `${session.title}${session.task ? ` · ${session.task.paused ? "⏸" : STAGES[session.task.state]}` : ""}`;
    button.addEventListener("click", () => openSession(session.id));
    return button;
  }));
}

function renderMessages(items, updateState = true) {
  if (updateState) {
    currentMessages = items;
    renderMemory();
  }
  if (!items.length) {
    messages.innerHTML = '<div class="empty"><span>✦</span><h2>Чем помочь?</h2><p>Напишите первую задачу. Перед работой агент коротко познакомится с вами.</p></div>';
    return;
  }
  messages.replaceChildren(...items.map((message) => {
    const article = document.createElement("article");
    const kind = message.kind || "message";
    article.className = `message ${message.role} ${kind}`;
    if (message.refusal && message.role === "assistant") article.classList.add("invariant-refusal");
    const label = document.createElement("span");
    label.textContent = kind === "command"
      ? "Команда памяти"
      : (message.role === "user" ? "Вы" : (message.refusal ? "Отказ · Инварианты" : "Агент"));
    let content;
    if (kind === "pending") {
      content = document.createElement("div");
      content.className = "typing-indicator";
      content.setAttribute("aria-label", "Агент формирует ответ");
      content.append(
        document.createElement("i"),
        document.createElement("i"),
        document.createElement("i"),
      );
    } else {
      content = document.createElement("p");
      content.textContent = message.content;
    }
    article.append(label, content);
    return article;
  }));
  renderTaskActivity(currentSession?.task);
  requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
}

function emptyMemory(text) {
  const item = document.createElement("p");
  item.className = "memory-empty";
  item.textContent = text;
  return item;
}

function renderShortTermMemory() {
  const shortTermMessages = currentMessages.filter((message) => (
    (message.kind || "message") === "message" && !message.refusal
  ));
  shortTermCount.textContent = String(shortTermMessages.length);
  const recent = shortTermMessages.slice(-6);
  if (!recent.length) {
    shortTermMemory.replaceChildren(emptyMemory("Диалог пока пуст."));
    return;
  }
  shortTermMemory.replaceChildren(...recent.map((message) => {
    const item = document.createElement("article");
    item.className = "memory-item compact";
    const category = document.createElement("strong");
    category.textContent = message.role === "user" ? "user" : "assistant";
    const content = document.createElement("p");
    content.textContent = message.content;
    item.append(category, content);
    return item;
  }));
}

function memoryEntryElement(entry) {
  const item = document.createElement("article");
  item.className = "memory-item";
  const heading = document.createElement("div");
  const category = document.createElement("strong");
  category.textContent = entry.category;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "memory-delete";
  remove.title = "Удалить запись";
  remove.setAttribute("aria-label", `Удалить ${entry.category}`);
  remove.textContent = "×";
  remove.addEventListener("click", () => deleteMemory(entry));
  heading.append(category, remove);
  const content = document.createElement("p");
  content.textContent = entry.content;
  item.append(heading, content);
  return item;
}

function renderMemoryList(container, entries, emptyText) {
  if (!entries.length) {
    container.replaceChildren(emptyMemory(emptyText));
    return;
  }
  container.replaceChildren(...entries.map(memoryEntryElement));
}

function renderMemory() {
  renderShortTermMemory();
  workingCount.textContent = String(memorySnapshot.working.length);
  longTermCount.textContent = String(memorySnapshot.long_term.length);
  renderMemoryList(workingMemory, memorySnapshot.working, "Нет данных задачи.");
  renderMemoryList(longTermMemory, memorySnapshot.long_term, "Нет общих воспоминаний.");
}

async function loadMemory(sessionId) {
  memorySnapshot = await api(`/api/memory?session_id=${encodeURIComponent(sessionId)}`);
  renderMemory();
}

async function deleteMemory(entry) {
  try {
    await api(`/api/memory/${entry.layer}/${entry.id}`, { method: "DELETE" });
    await loadMemory(currentSessionId);
    status.textContent = "Запись памяти удалена";
  } catch (error) {
    status.textContent = error.message;
  }
}

async function openSession(sessionId) {
  if (busy || controlBusy) return;
  try {
    const [session, memory] = await Promise.all([
      api(`/api/chat/sessions/${sessionId}`),
      api(`/api/memory?session_id=${encodeURIComponent(sessionId)}`),
    ]);
    memorySnapshot = memory;
    applySession(session);
    input.focus();
  } catch (error) {
    status.textContent = error.message;
  }
}

async function sendChatMessage(content) {
  setBusy(true);
  const optimisticMessages = [
    ...currentMessages,
    { role: "user", kind: "message", content },
    { role: "assistant", kind: "pending", content: "" },
  ];
  renderMessages(optimisticMessages, false);
  try {
    const result = await api(`/api/chat/sessions/${currentSessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    });
    const [session, memory, loadedProfiles] = await Promise.all([
      api(`/api/chat/sessions/${currentSessionId}`),
      api(`/api/memory?session_id=${encodeURIComponent(currentSessionId)}`),
      api("/api/profiles"),
    ]);
    profiles = loadedProfiles;
    memorySnapshot = memory;
    sessions = sessions.filter((item) => item.id !== result.session.id);
    sessions.unshift(result.session);
    title.textContent = result.session.title;
    renderSessions();
    renderProfiles();
    applySession(session);
    status.textContent = "Готов";
  } catch (error) {
    renderMessages(currentMessages);
    status.textContent = error.message;
  } finally {
    setBusy(false);
    input.focus();
    void drainQueue();
  }
}

async function dispatchContent(content) {
  const memoryCommand = parsedMemoryCommand(content);
  if (memoryCommand) {
    await executeMemoryCommand(memoryCommand);
  } else {
    await sendChatMessage(content);
  }
}

async function drainQueue() {
  if (busy || !queuedMessages.length || !currentSessionId) return;
  const [next] = queuedMessages.splice(0, 1);
  renderQueue();
  await dispatchContent(next.content);
}

async function createSession() {
  if (busy || controlBusy) return;
  try {
    const session = await api("/api/chat/sessions", {
      method: "POST",
      body: JSON.stringify({ profile_id: currentProfileId }),
    });
    sessions.unshift(session);
    await openSession(session.id);
  } catch (error) {
    status.textContent = error.message;
  }
}

async function loadSessions() {
  try {
    sessions = await api(`/api/chat/sessions?profile_id=${encodeURIComponent(currentProfileId)}`);
    if (sessions.length) await openSession(sessions[0].id);
    else await createSession();
  } catch (error) {
    status.textContent = error.message;
  }
}

async function activateProfile(profileId) {
  currentProfileId = profileId;
  currentSessionId = null;
  currentSession = null;
  renderTask(null);
  currentMessages = [];
  memorySnapshot = { working: [], long_term: [] };
  queuedMessages = [];
  rememberSelectedProfile();
  renderProfiles();
  renderQueue();
  renderMessages([]);
  title.textContent = "Новый чат";
  await loadSessions();
}

async function loadProfiles() {
  try {
    profiles = await api("/api/profiles");
    let savedProfileId = null;
    try {
      savedProfileId = localStorage.getItem("rubik-active-profile");
    } catch (_) {
      // Use the default profile when storage is unavailable.
    }
    const selected = profiles.some((profile) => profile.id === savedProfileId)
      ? savedProfileId
      : profiles[0]?.id;
    if (!selected) throw new Error("Не удалось создать основной профиль");
    await activateProfile(selected);
  } catch (error) {
    status.textContent = error.message;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = input.value.trim();
  if (!content || !currentSessionId) return;
  const memoryCommand = parsedMemoryCommand(content);
  if (memoryCommand && !memoryCommand.content) {
    await executeMemoryCommand(memoryCommand);
    return;
  }
  input.value = "";
  hideCommandMenu();
  if (busy) {
    enqueueMessage(content);
    input.focus();
    return;
  }
  await dispatchContent(content);
});

input.addEventListener("keydown", (event) => {
  const matches = commandMatches();
  if (!commandMenu.hidden && matches.length) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      activeCommandIndex = (activeCommandIndex + direction + matches.length) % matches.length;
      renderCommandMenu();
      return;
    }
    if (event.key === "Tab" || event.key === "Enter") {
      event.preventDefault();
      selectCommand(matches[activeCommandIndex]);
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      hideCommandMenu();
      return;
    }
  }
  if (
    event.key === "Enter"
    && !event.shiftKey
    && !event.isComposing
  ) {
    event.preventDefault();
    if (input.value.trim()) form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  activeCommandIndex = 0;
  renderCommandMenu();
});

newButton.addEventListener("click", createSession);
profileSelect.addEventListener("change", async () => {
  if (busy || controlBusy) {
    profileSelect.value = currentProfileId;
    return;
  }
  await activateProfile(profileSelect.value);
});
newProfileButton.addEventListener("click", async () => {
  if (busy || controlBusy) return;
  setBusy(true);
  status.textContent = "Создаю автопрофиль…";
  try {
    const created = await api("/api/profiles/auto", { method: "POST" });
    profiles = await api("/api/profiles");
    setBusy(false);
    await activateProfile(created.id);
    status.textContent = "Напишите первую задачу — агент начнёт интервью";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    setBusy(false);
  }
});
editProfileButton.addEventListener("click", () => openProfileDialog(activeProfile()));
closeProfileDialogButton.addEventListener("click", closeProfileDialog);
cancelProfileButton.addEventListener("click", closeProfileDialog);
deleteProfileButton.addEventListener("click", async () => {
  const profile = activeProfile();
  if (!profile || profile.id === "default" || busy) return;
  const confirmed = window.confirm(
    `Удалить профиль «${profile.name}» вместе со всеми его чатами и памятью? Это действие нельзя отменить.`,
  );
  if (!confirmed) return;

  deleteProfileButton.disabled = true;
  setBusy(true);
  status.textContent = "Удаляю профиль…";
  try {
    await api(`/api/profiles/${encodeURIComponent(profile.id)}`, { method: "DELETE" });
    closeProfileDialog();
    profiles = await api("/api/profiles");
    const fallbackProfile = profiles[0];
    if (!fallbackProfile) throw new Error("После удаления не осталось профилей");
    setBusy(false);
    await activateProfile(fallbackProfile.id);
    status.textContent = "Профиль и связанные данные удалены";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    deleteProfileButton.disabled = false;
    setBusy(false);
  }
});
profileDialog.addEventListener("click", (event) => {
  if (event.target === profileDialog) closeProfileDialog();
});
profileForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const profileId = profileIdInput.value;
  const saveButton = profileForm.querySelector("button[type='submit']");
  saveButton.disabled = true;
  status.textContent = "Сохраняю профиль…";
  try {
    const saved = await api(`/api/profiles/${encodeURIComponent(profileId)}`, {
      method: "PUT",
      body: JSON.stringify(profilePayload()),
    });
    profiles = await api("/api/profiles");
    closeProfileDialog();
    await activateProfile(saved.id);
    status.textContent = "Профиль скорректирован";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    saveButton.disabled = false;
  }
});
clearDatabaseButton.addEventListener("click", async () => {
  if (busy || !window.confirm("Удалить чаты активного профиля и их рабочую память? Долговременная память сохранится.")) return;

  setBusy(true);
  try {
    await api(`/api/chat/sessions?profile_id=${encodeURIComponent(currentProfileId)}`, { method: "DELETE" });
    sessions = [];
    currentSessionId = null;
    currentSession = null;
    renderTask(null);
    title.textContent = "Новый чат";
    renderSessions();
    renderMessages([]);
    status.textContent = "История очищена";
  } catch (error) {
    status.textContent = error.message;
    return;
  } finally {
    setBusy(false);
  }
  await createSession();
});

startTaskButton.addEventListener("click", async () => {
  const task = input.value.trim();
  if (busy || controlBusy || currentSession?.task || !activeProfile()?.onboarding_complete) return;
  if (!task) {
    status.textContent = "Опишите цель и требования задачи в поле сообщения";
    input.focus();
    return;
  }
  setBusy(true);
  let failed = false;
  try {
    const session = await api(`/api/chat/sessions/${currentSessionId}/task`, {
      method: "POST", body: JSON.stringify({ task }),
    });
    input.value = "";
    applySession(session);
  } catch (error) {
    failed = true;
    status.textContent = error.message;
  } finally {
    setBusy(false);
    if (!failed) status.textContent = readyStatus();
  }
});

async function runTaskAction(action, content = "") {
  const task = currentSession?.task;
  if (!task || controlBusy || (busy && !(action === "pause" && taskRunning))) return;
  const isPause = action === "pause";
  if (isPause) controlBusy = true;
  else {
    taskRunning = ["advance", "generate_plan", "execute_step", "validate"].includes(action);
    taskError = "";
    if (taskRunning) startTaskActivity(task);
    setBusy(true);
  }
  syncControls();
  let failed = false;
  try {
    const session = await api(`/api/chat/sessions/${currentSessionId}/task/actions`, {
      method: "POST", body: JSON.stringify({ action, revision: task.revision, content }),
    });
    applySession(session);
  } catch (error) {
    failed = true;
    if (!isPause) taskError = error.message;
    try { applySession(await api(`/api/chat/sessions/${currentSessionId}`)); } catch { /* Keep the last saved snapshot visible. */ }
    status.textContent = error.message;
  } finally {
    if (isPause) controlBusy = false;
    else {
      stopTaskActivity();
      taskRunning = false;
      setBusy(false);
    }
    syncControls();
    if (!failed) status.textContent = readyStatus();
  }
}

advanceButton.addEventListener("click", () => runTaskAction(currentSession.task.expected_action === "approve_plan" ? "approve" : currentSession.task.expected_action));
pauseButton.addEventListener("click", () => runTaskAction("pause"));
resumeButton.addEventListener("click", () => runTaskAction("resume"));
replanButton.addEventListener("click", () => {
  const content = window.prompt("Что изменить в плане? Завершённые результаты сохранятся.");
  if (content?.trim()) runTaskAction("replan", content.trim());
});
refreshButton.addEventListener("click", () => openSession(currentSessionId));
processTab.addEventListener("click", () => setInspector("process"));
memoryTab.addEventListener("click", () => setInspector("memory"));
inspectorTabs.addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  const mode = event.key === "Home" ? "process" : event.key === "End" ? "memory" : inspectorMode === "process" ? "memory" : "process";
  setInspector(mode);
  (mode === "process" ? processTab : memoryTab).focus();
});
showProcessButton.addEventListener("click", () => {
  setInspector("process");
  processTab.focus({ preventScroll: true });
  if (window.matchMedia("(max-width: 820px)").matches) {
    taskPanel.closest(".context-panel").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
  }
});


renderMemory();
renderQueue();
loadInvariants().catch(() => { renderActiveInvariants(); });
loadProfiles();
