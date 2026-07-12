const state = {
  sessionId: localStorage.getItem("enterprise-agent-session") || null,
  token: localStorage.getItem("enterprise-agent-token") || null,
  authMode: "login",
  isBusy: false,
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
    currentUser.textContent = `${result.data.username} / ${roleLabel(result.data.role)}`;
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
  localStorage.removeItem("enterprise-agent-token");
  localStorage.removeItem("enterprise-agent-session");
  currentUser.textContent = "未登录";
  showLogin();
}

function addMessage(role, content, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

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

  article.appendChild(avatar);
  article.appendChild(bubble);
  chatList.appendChild(article);
  chatList.scrollTop = chatList.scrollHeight;
  return bubble;
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
    const result = await api("/knowledge/stats");
    backend.textContent = result.data.backend;
    docCount.textContent = result.data.document_count;
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
  localStorage.setItem("enterprise-agent-session", sessionId);
  chatList.innerHTML = "";
  if (!result.data.length) {
    renderWelcome();
    return;
  }
  result.data.forEach((message) => addMessage(message.role, message.content));
}

function newConversation() {
  state.sessionId = null;
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

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader() },
      body: JSON.stringify({
        query,
        session_id: state.sessionId,
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
    currentUser.textContent = `${payload.data.username} / ${roleLabel(payload.data.role)}`;
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

async function readChatStream(response, bubble) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answer = "";

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
        appendSources(bubble, payload.data.sources || []);
      }
    }
  }

  if (!answer.trim()) {
    clearBubbleStatus(bubble);
    bubble.textContent = "没有返回内容。";
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

ensureAuthenticated();
