import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache

from AIRAGAgent.config.settings import settings
from AIRAGAgent.model.factory import chat_model, embed_model
from AIRAGAgent.utils.logger_handler import logger


@dataclass(frozen=True)
class QueryRoute:
    mode: str
    reason: str
    confidence: float
    retrieval_mode: str = "direct_policy"


CHAT_PATTERNS = [
    "你好",
    "您好",
    "hello",
    "hi",
    "嗨",
    "在吗",
    "你是谁",
    "你能做什么",
    "你可以做什么",
    "你有什么功能",
    "介绍一下你自己",
    "怎么使用你",
    "这个系统主要能帮我处理哪些",
    "这个系统能帮我处理哪些",
    "系统主要能帮我处理哪些",
    "你能帮我处理哪些",
    "你能处理哪些",
]

META_ANSWER_PATTERNS = [
    "没有相关制度依据",
    "没有相关依据",
    "没有找到依据",
    "未找到依据",
    "知识库里没有",
    "知识库没有",
    "资料里没有",
    "检索不到",
    "找不到资料",
    "你会怎么回答",
    "你应该怎么回答",
    "会怎么回答",
    "如何回答",
    "回答规范",
    "回答原则",
    "会不会编造",
    "是否会编造",
    "能不能编造",
]

META_SUBJECT_WORDS = ["你", "系统", "智能体", "助手", "知识库", "资料", "依据"]

CAPABILITY_WORDS = ["系统", "你", "智能体", "助手"]
CAPABILITY_INTENT_WORDS = [
    "能帮我处理哪些",
    "可以处理哪些",
    "能处理哪些",
    "主要能帮我",
    "有什么功能",
    "能做什么",
    "可以帮我做",
    "能帮我做",
    "能不能帮我",
    "是否可以帮我",
    "支持哪些",
    "支持什么",
    "能支持哪些",
    "可以支持哪些",
    "能做制度问答",
    "能生成",
    "能不能做",
    "可以帮我处理",
    "能处理",
    "能帮我查询",
    "会引用",
    "帮我查询制度还是生成申请",
    "怎么让你",
    "如何让你",
]

FLOW_CREATE_WORDS = [
    "起草",
    "生成",
    "写一份",
    "帮我写",
    "创建",
    "弄一个",
    "办一个",
    "整理申请",
    "整理一下申请",
    "申请草稿",
    "申请单",
    "提交申请",
    "发起申请",
    "拟一份",
]

FLOW_REQUEST_PREFIXES = [
    "我想申请",
    "我要申请",
    "我需要申请",
    "我想办理",
    "我要办理",
    "我需要办理",
    "帮我申请",
    "帮我办理",
]

FLOW_OBJECT_WORDS = [
    "申请",
    "草稿",
    "购买单",
    "采购单",
    "报销单",
    "申请内容",
    "模板",
    "报告",
    "说明",
    "汇报",
    "工作汇报",
    "审批",
    "流程",
    "采购",
    "报销",
    "请假",
    "权限",
    "离职",
    "入职",
]

POLICY_WORDS = [
    "制度",
    "规定",
    "标准",
    "要求",
    "材料",
    "证明",
    "流程",
    "审批",
    "手续",
    "怎么办",
    "怎么走",
    "怎么算",
    "哪些",
    "需要",
    "可以",
    "报销",
    "采购",
    "请假",
    "年假",
    "离职",
    "入职",
    "权限",
    "走人",
    "交接",
]

PROFESSIONAL_SIGNALS = [
    "制度",
    "规定",
    "标准",
    "要求",
    "材料",
    "证明",
    "流程",
    "审批",
    "手续",
    "依据",
    "哪一条",
    "风险",
    "合规",
    "违规",
    "费用标准",
    "报销标准",
    "适用条件",
    "所需材料",
    "交接",
    "离职交接",
    "交接手续",
]

AMBIGUOUS_PROCESS_PATTERNS = [
    "怎么处理",
    "怎么弄",
    "怎么走",
    "怎么推进",
    "接下来怎么",
    "下一步怎么",
    "下一步应该怎么",
    "下一步应该怎么走",
    "下一步应该怎么处理",
    "应该怎么处理",
    "应该怎么推进",
]

DOMAIN_KEYWORDS = {
    "reimbursement": ["报销", "差旅", "出差", "费用", "发票", "住宿", "交通", "餐补"],
    "leave_attendance": ["请假", "事假", "考勤", "病假", "年假", "调休"],
    "procurement": ["采购", "供应商", "合同", "办公设备"],
    "security": ["权限", "信息安全", "账号", "数据安全", "生产权限", "保密", "外发"],
    "onboarding": ["入职", "转正", "试用期", "离职", "招聘"],
    "administration": ["档案", "借阅", "仓库", "领料", "车辆", "会议", "印章"],
    "safety": ["安全整改", "安全生产", "环保检查"],
    "salary_performance": ["薪资", "绩效", "销售提成", "奖金"],
    "ticket_sop": ["工单", "客户", "SOP"],
}

ROUTE_PROMPT = """你是企业知识智能体的低置信路由裁决器。只返回 JSON，不要解释。
请在四类链路中选择最合适的一类：
- chat：询问系统能力、使用方式、能做什么、支持哪些场景、回答规范、没有依据时如何回答，或普通寒暄。不要为这类元问题检索企业制度。
- direct_policy：简单制度事实问答，只问单一规则、天数、是否需要、能不能、可不可以。
- professional_policy：查询制度流程、审批材料、适用条件、办理手续、制度依据、费用标准、安全要求。
- complex_process：要求生成、起草、创建、办理申请草稿或流程方案，需要审批路径、风险校验、草稿生成等多工具协同。

判定规则：
- 如果用户问“你/系统/助手在没有依据时怎么回答、会不会编造、如何引用依据”，必须返回 chat。
- 只有用户在询问某个真实业务制度、流程、材料、审批、标准时，才返回 direct_policy 或 professional_policy。
- 如果问题有业务对象但用户意图不清楚，例如既可能查制度、又可能生成申请、又可能做风险校验，则返回 needs_clarification=true。

用户问题：{query}

返回格式：
{{"retrieval_mode":"chat|direct_policy|professional_policy|complex_process","needs_clarification":false,"reason":"简短原因","confidence":0.0}}
"""

SEMANTIC_ROUTE_DESCRIPTIONS = {
    "chat": "询问系统能力、使用方式、能做什么、支持哪些场景、回答规范、没有制度依据时如何回答、是否会编造，或普通寒暄，不需要检索企业知识库。",
    "direct_policy": "简单制度事实问答，只问一个明确规则、天数、是否需要、能不能、可不可以，适合轻量向量检索。",
    "professional_policy": "专业制度查询，需要审批材料、适用条件、办理流程、制度依据、费用标准、安全要求或多个条件组合，适合元数据过滤和多路召回。",
    "complex_process": "需要生成、起草、创建申请草稿或流程方案，需要组合审批路径、制度检索、风险校验和草稿生成等多个工具。",
}

SEMANTIC_ROUTE_TO_MODE = {
    "chat": "chat",
    "direct_policy": "policy",
    "professional_policy": "policy",
    "complex_process": "flow",
}


def normalize_query(query: str) -> str:
    text = query.strip().lower()
    return re.sub(r"[\s，。！？、；；,.!?;:\"'“”‘’（）()【】\[\]{}<>《》]+", "", text)


def classify_query(query: str, use_llm_fallback: bool = True) -> QueryRoute:
    text = normalize_query(query)
    rule_route = classify_by_rules(query, text)
    if is_strong_rule_route(rule_route):
        logger.info(
            "[QueryRoute] mode=%s retrieval_mode=%s confidence=%.2f reason=%s query=%s",
            rule_route.mode,
            rule_route.retrieval_mode,
            rule_route.confidence,
            rule_route.reason,
            query,
        )
        return rule_route

    semantic_route = classify_by_semantic(query) if settings.SEMANTIC_ROUTER_ENABLED else None
    route = choose_route(rule_route, semantic_route)
    if route.reason == "ambiguous_business_process":
        logger.info(
            "[QueryRoute] mode=%s retrieval_mode=%s confidence=%.2f reason=%s query=%s",
            route.mode,
            route.retrieval_mode,
            route.confidence,
            route.reason,
            query,
        )
        return route
    if (
        route.confidence >= 0.72
        or not use_llm_fallback
        or not settings.ROUTER_LLM_FALLBACK_ENABLED
    ):
        logger.info(
            "[QueryRoute] mode=%s retrieval_mode=%s confidence=%.2f reason=%s query=%s",
            route.mode,
            route.retrieval_mode,
            route.confidence,
            route.reason,
            query,
        )
        return route

    llm_route = classify_by_llm(query)
    logger.info(
        "[QueryRoute] mode=%s retrieval_mode=%s confidence=%.2f reason=%s query=%s",
        llm_route.mode,
        llm_route.retrieval_mode,
        llm_route.confidence,
        llm_route.reason,
        query,
    )
    return llm_route


def is_strong_rule_route(route: QueryRoute) -> bool:
    return route.reason in {
        "chat_pattern",
        "short_chat_semantic_pattern",
        "meta_answer_policy_question",
        "capability_question",
        "explicit_create_action_with_flow_object",
        "explicit_request_intent_with_flow_object",
        "simple_fact_policy_signal",
    }


def classify_by_rules(query: str, normalized: str) -> QueryRoute:
    normalized_chat = {normalize_query(item) for item in CHAT_PATTERNS}
    if normalized in normalized_chat:
        return QueryRoute("chat", "chat_pattern", 0.98, "chat")

    if any(normalize_query(item) in normalized for item in CHAT_PATTERNS) and len(normalized) <= 14:
        return QueryRoute("chat", "short_chat_semantic_pattern", 0.9, "chat")

    if is_meta_answer_question(query):
        return QueryRoute("chat", "meta_answer_policy_question", 0.96, "chat")

    if any(word in query for word in CAPABILITY_WORDS) and any(word in query for word in CAPABILITY_INTENT_WORDS):
        return QueryRoute("chat", "capability_question", 0.92, "chat")

    domain = infer_domain(query)
    if is_simple_fact_question(query, normalized) and (
        domain
        or any(word in query for word in POLICY_WORDS)
        or any(word in query for word in ["协议", "申请", "来源"])
    ):
        return QueryRoute("policy", "simple_fact_policy_signal", 0.88, "direct_policy")

    if any(prefix in query for prefix in FLOW_REQUEST_PREFIXES) and any(word in query for word in FLOW_OBJECT_WORDS):
        return QueryRoute("flow", "explicit_request_intent_with_flow_object", 0.93, "complex_process")

    create_score = sum(1 for word in FLOW_CREATE_WORDS if word in query)
    object_score = sum(1 for word in FLOW_OBJECT_WORDS if word in query)
    if create_score >= 1 and object_score >= 1:
        return QueryRoute("flow", "explicit_create_action_with_flow_object", 0.95, "complex_process")

    policy_score = sum(1 for word in POLICY_WORDS if word in query)
    professional_score = sum(1 for word in PROFESSIONAL_SIGNALS if word in query)
    query_len = len(normalized)

    if domain and any(pattern in query for pattern in AMBIGUOUS_PROCESS_PATTERNS) and not has_explicit_policy_reference(query):
        return QueryRoute("policy", "ambiguous_business_process", 0.66, "professional_policy")

    if professional_score >= 2 or (domain and professional_score >= 1 and query_len >= 12):
        return QueryRoute("policy", "professional_policy_candidate", 0.74, "professional_policy")

    if policy_score >= 1:
        if is_simple_fact_question(query, normalized):
            return QueryRoute("policy", "simple_fact_policy_signal", 0.88, "direct_policy")
        return QueryRoute("policy", "direct_policy_candidate", 0.68, "direct_policy")

    return QueryRoute("policy", "ambiguous_default_direct_policy", 0.6, "direct_policy")


def is_simple_fact_question(query: str, normalized: str) -> bool:
    simple_patterns = ["几天", "多少天", "能不能", "可不可以", "可以吗", "需要吗", "要不要", "是否", "必须", "吗"]
    complex_patterns = ["哪些材料", "审批流程", "审批要求", "风险", "合规", "申请内容", "审批路径", "所需材料"]
    if any(pattern in query for pattern in complex_patterns):
        return False
    return len(normalized) <= 18 and any(pattern in query for pattern in simple_patterns)


def is_meta_answer_question(query: str) -> bool:
    if not any(pattern in query for pattern in META_ANSWER_PATTERNS):
        return False
    if any(word in query for word in META_SUBJECT_WORDS):
        return True
    return any(pattern in query for pattern in ["会怎么回答", "如何回答", "回答规范", "回答原则"])


def has_explicit_policy_reference(query: str) -> bool:
    return any(word in query for word in ["制度", "规定", "办法", "机制", "细则", "要求", "管理办法"])


def choose_route(rule_route: QueryRoute, semantic_route: QueryRoute = None) -> QueryRoute:
    if rule_route.reason == "ambiguous_business_process":
        return rule_route
    if semantic_route is None:
        return rule_route
    if semantic_margin(semantic_route) < 0.06 and rule_route.retrieval_mode != "chat":
        return rule_route
    if semantic_route.reason.startswith("semantic_route:chat") and rule_route.retrieval_mode != "chat":
        margin = semantic_margin(semantic_route)
        if margin < 0.05:
            return QueryRoute(
                mode=rule_route.mode,
                reason=f"semantic_chat_low_margin_keep_{rule_route.reason}",
                confidence=min(rule_route.confidence, 0.69),
                retrieval_mode=rule_route.retrieval_mode,
            )
    if (
        semantic_route.retrieval_mode == "complex_process"
        and rule_route.retrieval_mode == "professional_policy"
        and rule_route.reason in {"professional_policy_candidate", "ambiguous_business_process"}
    ):
        return rule_route
    if semantic_route.confidence >= settings.SEMANTIC_ROUTER_THRESHOLD:
        if rule_route.reason.endswith("_candidate") or rule_route.confidence < 0.84:
            return semantic_route
        if semantic_route.retrieval_mode != rule_route.retrieval_mode and semantic_route.confidence >= 0.78:
            return semantic_route
    return rule_route


def semantic_margin(route: QueryRoute) -> float:
    match = re.search(r"margin=([0-9.]+)", route.reason)
    return float(match.group(1)) if match else 1.0


@lru_cache(maxsize=512)
def classify_by_semantic(query: str) -> QueryRoute:
    try:
        query_vector = embed_model.embed_query(query)
        scored = []
        for retrieval_mode, description in SEMANTIC_ROUTE_DESCRIPTIONS.items():
            route_vector = _route_embedding(retrieval_mode, description)
            scored.append((cosine_similarity(query_vector, route_vector), retrieval_mode))
        scored.sort(reverse=True)
        best_score, retrieval_mode = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        confidence = max(0.0, min((best_score + 1) / 2, 1.0))
        margin = best_score - second_score
        if margin < 0.025:
            confidence = min(confidence, 0.7)
        return QueryRoute(
            mode=SEMANTIC_ROUTE_TO_MODE[retrieval_mode],
            reason=f"semantic_route:{retrieval_mode}:margin={margin:.3f}",
            confidence=confidence,
            retrieval_mode=retrieval_mode,
        )
    except Exception as exc:
        logger.warning("[QueryRoute] semantic route failed: %s", exc)
        return QueryRoute("policy", "semantic_failed", 0.0, "direct_policy")


@lru_cache(maxsize=16)
def _route_embedding(retrieval_mode: str, description: str):
    return embed_model.embed_query(f"{retrieval_mode}: {description}")


def cosine_similarity(left, right) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def infer_domain(query: str):
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            return domain
    return None


@lru_cache(maxsize=256)
def classify_by_llm(query: str) -> QueryRoute:
    try:
        response = chat_model.invoke(ROUTE_PROMPT.format(query=query))
        content = getattr(response, "content", str(response))
        data = json.loads(extract_json_object(content))
        retrieval_mode = data.get("retrieval_mode") or data.get("mode")
        if retrieval_mode in {"policy", "flow"}:
            retrieval_mode = "professional_policy" if retrieval_mode == "policy" else "complex_process"
        if retrieval_mode not in SEMANTIC_ROUTE_TO_MODE:
            raise ValueError(f"invalid retrieval_mode: {retrieval_mode}")
        confidence = max(0.0, min(float(data.get("confidence", 0.7)), 1.0))
        if data.get("needs_clarification"):
            return QueryRoute(
                SEMANTIC_ROUTE_TO_MODE[retrieval_mode],
                f"llm_clarify_needed:{data.get('reason', 'ambiguous')}",
                min(confidence, 0.69),
                retrieval_mode,
            )
        return QueryRoute(
            SEMANTIC_ROUTE_TO_MODE[retrieval_mode],
            str(data.get("reason", "llm_route")),
            confidence,
            retrieval_mode,
        )
    except Exception as exc:
        logger.warning("[QueryRoute] LLM fallback failed: %s", exc)
        return QueryRoute("policy", "llm_failed_default_direct_policy", 0.5, "direct_policy")


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no json object found")
    return text[start : end + 1]


def is_small_talk(query: str) -> bool:
    return classify_query(query).mode == "chat"


def needs_agent_workflow(query: str) -> bool:
    return classify_query(query).mode == "flow"


def needs_clarification(route: QueryRoute) -> bool:
    if not settings.CLARIFY_ROUTE_ENABLED:
        return False
    if route.mode == "chat":
        return False
    if route.reason in {
        "explicit_create_action_with_flow_object",
        "explicit_request_intent_with_flow_object",
        "simple_fact_policy_signal",
        "professional_policy_candidate",
        "direct_policy_candidate",
    }:
        return False
    if route.reason.startswith("llm_clarify_needed"):
        return True
    if route.reason.startswith("llm_failed"):
        return route.confidence < settings.CLARIFY_ROUTE_THRESHOLD
    return route.reason.startswith("ambiguous") and route.confidence < settings.CLARIFY_ROUTE_THRESHOLD
