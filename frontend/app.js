const state = {
  sessionId: localStorage.getItem("enterprise-agent-session") || null,
  token: localStorage.getItem("enterprise-agent-token") || null,
  role: localStorage.getItem("enterprise-agent-role") || "user",
  authMode: "login",
  isBusy: false,
  isNewConversation: !localStorage.getItem("enterprise-agent-session"),
};

const loginOverlay = document.querySelector("#loginOverlay");
const loginForm = document.querySelector("#loginForm");
const authTitle = document.querySelector("#authTitle");
const authSubmitBtn = document.querySelector("#authSubmitBtn");
const switchAuthModeBtn = document.querySelector("#switchAuthModeBtn");
const usernameInput = document.querySelector("#usernameInput");
const passwordInput = document.querySelector("#passwordInput");
const loginError = document.querySelector("#loginError");
const currentUser = document.querySelector("#currentUser");
const logoutBtn = document.querySelector("#logoutBtn");
const newChatBtn = document.querySelector("#newChatBtn");
const conversationList = document.querySelector("#conversationList");
const chatList = document.querySelector("#chatList");
const chatForm = document.querySelector("#chatForm");
const queryInput = document.querySelector("#queryInput");
const sendBtn = document.querySelector("#sendBtn");
const ingestBtn = document.querySelector("#ingestBtn");
const forceIngestBtn = document.querySelector("#forceIngestBtn");
const ingestResult = document.querySelector("#ingestResult");
const knowledgePanel = document.querySelector("#knowledgePanel");
const serviceStatus = document.querySelector("#serviceStatus");
const backend = document.querySelector("#backend");
const docCount = document.querySelector("#docCount");
const agentMode = document.querySelector("#agentMode");

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...authHeader(),
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function authHeader() {
  return state.token ? { Authorization: `Bearer ${state.token}` } : {};
}

async function ensureAuthenticated() {
  if (!state.token) {
    showLogin();
    return;
  }
  try {
    const result = await api("/auth/me");
    setCurrentUser(result.data);
    hideLogin();
    await refreshStats();
    await loadConversations();
  } catch (error) {
    logout();
  }
}

function showLogin() {
  loginOverlay.classList.add("active");
}

function hideLogin() {
  loginOverlay.classList.remove("active");
}

function logout() {
  state.token = null;
  state.sessionId = null;
  state.isNewConversation = true;
  state.role = "user";
  localStorage.removeItem("enterprise-agent-token");
  localStorage.removeItem("enterprise-agent-session");
  localStorage.removeItem("enterprise-agent-role");
  currentUser.textContent = "未登录";
  applyRolePermissions();
  showLogin();
}

function addMessage(role, content, sources = [], meta = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  if (meta.status) {
    article.dataset.status = meta.status;
  }
  if (meta.id) {
    article.dataset.messageId = meta.id;
  }

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;

  if (sources.length) {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";
    sourceBox.textContent = `来源：${sources.map((item) => item.file_name || item.source || "unknown").join("、")}`;
    bubble.appendChild(sourceBox);
  }

  if (meta.status === "failed" && meta.retryable && meta.id) {
    appendRetryAction(bubble, meta.id);
  }

  article.appendChild(avatar);
  article.appendChild(bubble);
  chatList.appendChild(article);
  chatList.scrollTop = chatList.scrollHeight;
  return bubble;
}

function appendRetryAction(bubble, messageId) {
  const actions = document.createElement("div");
  actions.className = "message-actions";

  const retryBtn = document.createElement("button");
  retryBtn.type = "button";
  retryBtn.className = "retry-btn";
  retryBtn.textContent = "重试";
  retryBtn.addEventListener("click", () => retryMessage(messageId, bubble, retryBtn));

  actions.appendChild(retryBtn);
  bubble.appendChild(actions);
}

function setBubbleStatus(bubble, message) {
  bubble.classList.add("thinking");
  bubble.textContent = message;
}

function clearBubbleStatus(bubble) {
  bubble.classList.remove("thinking");
}

function setBusy(isBusy) {
  state.isBusy = isBusy;
  sendBtn.disabled = isBusy;
  sendBtn.textContent = isBusy ? "处理中" : "发送";
}

async function refreshStats() {
  try {
    await api("/health");
    serviceStatus.textContent = "在线";
    if (isAdmin()) {
      const result = await api("/knowledge/stats");
      backend.textContent = result.data.backend;
      docCount.textContent = result.data.document_count;
    } else {
      backend.textContent = "仅管理员可见";
      docCount.textContent = "仅管理员可见";
    }
  } catch (error) {
    serviceStatus.textContent = "异常";
    backend.textContent = "-";
    docCount.textContent = "-";
  }
}

async function ingestKnowledge(force) {
  const button = force ? forceIngestBtn : ingestBtn;
  button.disabled = true;
  ingestResult.textContent = force ? "正在强制重建索引..." : "正在增量更新知识库...";

  try {
    const result = await api("/knowledge/ingest/async", {
      method: "POST",
      body: JSON.stringify({ force }),
    });
    const taskId = result.data.task_id;
    ingestResult.textContent = `任务已提交：${taskId}`;
    const data = await pollIngestTask(taskId);
    ingestResult.textContent = `完成：扫描 ${data.scanned_files} 个文件，跳过 ${data.skipped_files} 个文件，入库 ${data.indexed_files} 个文件，新增 ${data.indexed_chunks} 个分片，删除 ${data.deleted_chunks} 个旧分片。`;
    await refreshStats();
  } catch (error) {
    ingestResult.textContent = `失败：${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function loadConversations() {
  const result = await api("/conversations");
  conversationList.innerHTML = "";
  result.data.forEach((item) => {
    const button = document.createElement("button");
    button.textContent = item.title || "新会话";
    button.addEventListener("click", () => openConversation(item.session_id));
    conversationList.appendChild(button);
  });
}

async function openConversation(sessionId) {
  const result = await api(`/conversations/${sessionId}/messages`);
  state.sessionId = sessionId;
  state.isNewConversation = false;
  localStorage.setItem("enterprise-agent-session", sessionId);
  chatList.innerHTML = "";
  if (!result.data.length) {
    renderWelcome();
    return;
  }
  result.data.forEach((message) => {
    const content = message.status === "pending" && !message.content
      ? "上次生成未完成，请重新提问或稍后刷新。"
      : message.content;
    addMessage(message.role, content, [], {
      id: message.id,
      status: message.status,
      retryable: message.retryable,
      errorMessage: message.error_message,
    });
  });
}

function newConversation() {
  state.sessionId = null;
  state.isNewConversation = true;
  localStorage.removeItem("enterprise-agent-session");
  chatList.innerHTML = "";
  renderWelcome();
}

function renderWelcome() {
  addMessage("assistant", "你好，我可以查询企业制度、生成流程申请草稿，也可以结合工具生成咨询报告。");
}

async function pollIngestTask(taskId) {
  while (true) {
    const result = await api(`/knowledge/tasks/${taskId}`);
    const data = result.data;
    if (data.status === "completed") {
      return data.result;
    }
    if (data.status === "failed") {
      throw new Error(data.error || "知识库任务失败");
    }
    ingestResult.textContent = `任务运行中：${taskId}`;
    await new Promise((resolve) => setTimeout(resolve, 1200));
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.isBusy) {
    return;
  }
  const query = queryInput.value.trim();
  if (!query) {
    return;
  }

  addMessage("user", query);
  queryInput.value = "";
  queryInput.style.height = "auto";
  const pendingBubble = addMessage("assistant", "");
  setBubbleStatus(pendingBubble, "正在检索知识库并分析问题...");
  setBusy(true);
  const outboundSessionId = state.isNewConversation ? null : state.sessionId;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader() },
      body: JSON.stringify({
        query,
        session_id: outboundSessionId,
        use_agent: agentMode.checked,
      }),
    });

    if (!response.ok || !response.body) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    await readChatStream(response, pendingBubble);
    await loadConversations();
  } catch (error) {
    pendingBubble.textContent = `请求失败：${error.message}`;
  } finally {
    setBusy(false);
    queryInput.focus();
  }
});

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  loginError.textContent = "";
  try {
    const path = state.authMode === "login" ? "/api/auth/login" : "/api/auth/register";
    const body = {
      username: usernameInput.value.trim(),
      password: passwordInput.value,
    };
    if (state.authMode === "register") {
      body.tenant_id = "default";
    }
    const result = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!result.ok) {
      const text = await result.text();
      throw new Error(state.authMode === "login" ? "用户名或密码错误" : text);
    }
    const payload = await result.json();
    state.token = payload.data.access_token;
    localStorage.setItem("enterprise-agent-token", state.token);
    setCurrentUser(payload.data);
    hideLogin();
    await refreshStats();
    await loadConversations();
  } catch (error) {
    loginError.textContent = error.message;
  }
});

logoutBtn.addEventListener("click", logout);
newChatBtn.addEventListener("click", newConversation);
switchAuthModeBtn.addEventListener("click", () => {
  state.authMode = state.authMode === "login" ? "register" : "login";
  const isRegister = state.authMode === "register";
  authTitle.textContent = isRegister ? "注册普通用户" : "登录企业知识智能体";
  authSubmitBtn.textContent = isRegister ? "注册并登录" : "登录";
  switchAuthModeBtn.textContent = isRegister ? "已有账号？返回登录" : "没有账号？注册普通用户";
  loginError.textContent = "";
});

function roleLabel(role) {
  return role === "admin" ? "管理员" : "用户";
}

function isAdmin() {
  return state.role === "admin";
}

function setCurrentUser(user) {
  state.role = user.role || "user";
  localStorage.setItem("enterprise-agent-role", state.role);
  currentUser.textContent = `${user.username} / ${roleLabel(state.role)}`;
  applyRolePermissions();
}

function applyRolePermissions() {
  const admin = isAdmin();
  if (knowledgePanel) {
    knowledgePanel.hidden = !admin;
  }
  ingestBtn.disabled = !admin;
  forceIngestBtn.disabled = !admin;
  if (!admin) {
    ingestResult.textContent = "";
  }
}

async function readChatStream(response, bubble) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answer = "";
  let assistantMessageId = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      const payload = JSON.parse(line);
      if (payload.event === "session") {
        state.sessionId = payload.data.session_id;
        state.isNewConversation = false;
        assistantMessageId = payload.data.assistant_message_id || assistantMessageId;
        if (assistantMessageId) {
          bubble.closest(".message").dataset.messageId = assistantMessageId;
        }
        localStorage.setItem("enterprise-agent-session", state.sessionId);
      }
      if (payload.event === "status") {
        if (!answer.trim()) {
          setBubbleStatus(bubble, payload.data.message || "正在处理...");
          chatList.scrollTop = chatList.scrollHeight;
        }
      }
      if (payload.event === "delta") {
        if (!answer.trim()) {
          clearBubbleStatus(bubble);
          bubble.textContent = "";
        }
        answer += payload.data.content || "";
        bubble.textContent = answer;
        chatList.scrollTop = chatList.scrollHeight;
      }
      if (payload.event === "done") {
        assistantMessageId = payload.data.assistant_message_id || assistantMessageId;
        appendSources(bubble, payload.data.sources || []);
      }
      if (payload.event === "error") {
        clearBubbleStatus(bubble);
        bubble.textContent = payload.data.message || "生成失败，请稍后重试。";
        const failedId = payload.data.assistant_message_id || assistantMessageId;
        const article = bubble.closest(".message");
        article.dataset.status = "failed";
        if (failedId && payload.data.retryable !== false) {
          article.dataset.messageId = failedId;
          appendRetryAction(bubble, failedId);
        }
        return;
      }
    }
  }

  if (!answer.trim()) {
    clearBubbleStatus(bubble);
    bubble.textContent = "没有返回内容。";
  }
}

async function retryMessage(messageId, bubble, retryBtn) {
  if (state.isBusy) {
    return;
  }
  setBusy(true);
  retryBtn.disabled = true;
  bubble.textContent = "正在重试生成回答...";
  bubble.classList.add("thinking");

  try {
    const result = await api(`/messages/${messageId}/retry`, { method: "POST" });
    clearBubbleStatus(bubble);
    bubble.textContent = result.data.answer || "没有返回内容。";
    appendSources(bubble, result.data.sources || []);
    const article = bubble.closest(".message");
    article.dataset.status = "success";
    article.dataset.messageId = result.data.assistant_message_id || messageId;
    await loadConversations();
  } catch (error) {
    clearBubbleStatus(bubble);
    bubble.textContent = `重试失败：${error.message}`;
    appendRetryAction(bubble, messageId);
  } finally {
    setBusy(false);
    queryInput.focus();
  }
}

function appendSources(bubble, sources) {
  if (!sources.length) {
    return;
  }
  const sourceBox = document.createElement("div");
  sourceBox.className = "sources";
  sourceBox.textContent = `来源：${sources.map((item) => item.file_name || item.source || "unknown").join("、")}`;
  bubble.appendChild(sourceBox);
}

queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = `${queryInput.scrollHeight}px`;
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (state.isBusy) {
      return;
    }
    chatForm.requestSubmit();
  }
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.example;
    queryInput.focus();
  });
});

ingestBtn.addEventListener("click", () => ingestKnowledge(false));
forceIngestBtn.addEventListener("click", () => ingestKnowledge(true));

applyRolePermissions();
ensureAuthenticated();
