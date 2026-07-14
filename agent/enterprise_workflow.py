import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from AIRAGAgent.agent.tools.agent_tools import (
    check_policy_risk,
    create_application_draft,
    fetch_external_data,
    fill_context_for_report,
    get_current_month,
    get_employee_department,
    get_employee_id,
    locate_policy_clause,
    query_approval_path,
    query_expense_standard,
    rag_summarize,
)
from AIRAGAgent.config.settings import settings
from AIRAGAgent.model.factory import chat_model
from AIRAGAgent.services.query_classifier import classify_query
from AIRAGAgent.utils.amount_parser import extract_amount
from AIRAGAgent.utils.json_guard import parse_json_array
from AIRAGAgent.utils.logger_handler import logger
from AIRAGAgent.utils.trace import elapsed_ms, log_trace, now_ms, short_text


class EnterpriseAgentState(TypedDict, total=False):
    query: str
    history: List[Dict[str, str]]
    intent: str
    plan: List[Dict[str, Any]]
    tool_results: Dict[str, str]
    review_notes: List[str]
    final_answer: str


ALLOWED_TOOLS = {
    "rag_summarize": {"query"},
    "locate_policy_clause": {"query", "top_k"},
    "query_approval_path": {"application_type", "amount", "department"},
    "query_expense_standard": {"expense_type", "city_level"},
    "check_policy_risk": {"application_type", "content", "amount"},
    "create_application_draft": {"application_type", "reason", "key_info"},
    "get_employee_id": set(),
    "get_employee_department": set(),
    "get_current_month": set(),
    "fetch_external_data": {"employee_id", "month"},
    "fill_context_for_report": set(),
}

TOOL_DESCRIPTIONS = """
- rag_summarize(query): 检索企业制度知识库并摘要回答。
- locate_policy_clause(query, top_k): 定位制度来源、条款或命中片段。
- query_approval_path(application_type, amount, department): 查询申请审批路径和所需材料。
- query_expense_standard(expense_type, city_level): 查询报销、住宿、交通、餐补、差旅等费用标准。
- check_policy_risk(application_type, content, amount): 校验流程、费用、权限、证明材料等合规风险。
- create_application_draft(application_type, reason, key_info): 生成内部申请草稿。
- get_employee_id(): 获取当前员工ID。
- get_employee_department(): 获取当前员工部门。
- get_current_month(): 获取当前月份。
- fetch_external_data(employee_id, month): 查询外部系统模拟统计数据。
- fill_context_for_report(): 填充报告生成上下文。
"""

PLANNER_PROMPT = """你是企业知识 Agent 的任务规划器。请根据用户问题和意图，从白名单工具中选择必要工具并生成执行计划。

要求：
1. 只能使用白名单工具。
2. 不要使用无关工具。
3. 流程/申请类问题通常需要审批路径、制度检索、风险校验、草稿生成。
4. 风险类问题通常需要风险校验和制度依据定位。
5. 费用标准类问题通常需要费用标准和制度依据定位。
6. 只输出 JSON，不要输出解释。

白名单工具：
{tool_descriptions}

用户问题：{query}
识别意图：{intent}
金额：{amount}

输出格式：
[
  {{"tool": "工具名", "args": {{"参数名": "参数值"}}}}
]
"""

FINAL_PROMPT = """你是企业知识智能体，请根据用户问题和工具执行结果，生成最终回答。

要求：
1. 不要重复用户问题。
2. 结构清晰，必要时分步骤输出。
3. 涉及制度依据时说明来源或依据摘要。
4. 涉及风险时明确风险等级和整改建议。
5. 如果工具结果包含“依据状态：制度依据不足”或“规则推断”，必须明确提示用户提交前需要人工确认。
6. 不要把规则推断包装成制度原文结论。

用户问题：
{query}

工具结果：
{tool_results}
"""


class EnterpriseKnowledgeWorkflow:
    def __init__(self):
        self.final_chain = PromptTemplate.from_template(FINAL_PROMPT) | chat_model | StrOutputParser()
        self.planner_chain = PromptTemplate.from_template(PLANNER_PROMPT) | chat_model | StrOutputParser()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(EnterpriseAgentState)
        graph.add_node("intent", self.intent_node)
        graph.add_node("planner", self.planner_node)
        graph.add_node("tool_executor", self.tool_executor_node)
        graph.add_node("risk_review", self.risk_review_node)
        graph.add_node("final_answer", self.final_answer_node)
        graph.set_entry_point("intent")
        graph.add_edge("intent", "planner")
        graph.add_edge("planner", "tool_executor")
        graph.add_edge("tool_executor", "risk_review")
        graph.add_edge("risk_review", "final_answer")
        graph.add_edge("final_answer", END)
        return graph.compile()

    def invoke(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        start = now_ms()
        log_trace(logger, "workflow_start", mode="sync", history_count=len(history or []), query=short_text(query))
        result = self.graph.invoke({"query": query, "history": history or []})
        answer = result.get("final_answer", "")
        log_trace(
            logger,
            "workflow_done",
            mode="sync",
            intent=result.get("intent"),
            tools=",".join((result.get("tool_results") or {}).keys()),
            answer_chars=len(answer or ""),
            elapsed_ms=elapsed_ms(start),
        )
        return answer

    def stream(self, query: str, history: Optional[List[Dict[str, str]]] = None):
        start = now_ms()
        log_trace(logger, "workflow_start", mode="stream", history_count=len(history or []), query=short_text(query))
        state: EnterpriseAgentState = {"query": query, "history": history or []}
        state = self.intent_node(state)
        state = self.planner_node(state)
        state = self.tool_executor_node(state)
        state = self.risk_review_node(state)
        first_chunk = True
        for chunk in self.final_chain.stream(
            {
                "query": state["query"],
                "tool_results": json.dumps(state.get("tool_results", {}), ensure_ascii=False, indent=2),
            }
        ):
            if chunk:
                if first_chunk:
                    log_trace(
                        logger,
                        "workflow_first_chunk",
                        intent=state.get("intent"),
                        tools=",".join((state.get("tool_results") or {}).keys()),
                        elapsed_ms=elapsed_ms(start),
                    )
                    first_chunk = False
                yield str(chunk)
        log_trace(
            logger,
            "workflow_done",
            mode="stream",
            intent=state.get("intent"),
            tools=",".join((state.get("tool_results") or {}).keys()),
            elapsed_ms=elapsed_ms(start),
        )

    def intent_node(self, state: EnterpriseAgentState) -> EnterpriseAgentState:
        query = state["query"]
        route = classify_query(query)
        if route.mode == "flow":
            intent = "flow_generation"
        elif any(word in query for word in ["月报", "报告", "统计"]):
            intent = "report_generation"
        elif any(word in query for word in ["风险", "合规", "违规", "长期权限", "生产权限"]):
            intent = "risk_check"
        elif any(word in query for word in ["住宿", "交通", "餐补", "差旅", "费用标准", "报销标准"]):
            intent = "expense_standard"
        elif any(word in query for word in ["条款", "依据", "来源", "哪一条"]):
            intent = "policy_trace"
        else:
            intent = "policy_qa"
        logger.info("[EnterpriseWorkflow] intent=%s query=%s", intent, query)
        log_trace(logger, "workflow_intent", intent=intent, route_mode=route.mode, query=short_text(query))
        return {**state, "intent": intent}

    def _is_flow_generation_query(self, query: str) -> bool:
        create_words = ["生成", "草稿", "申请单", "帮我写", "创建", "起草", "提交申请", "发起申请"]
        action_phrases = ["我要申请", "我想申请", "帮我办理", "需要办理", "办理申请"]
        if any(word in query for word in create_words + action_phrases):
            return True
        if "流程" in query and any(word in query for word in ["怎么申请", "如何申请", "怎么办理", "如何办理"]):
            return True
        return False

    def planner_node(self, state: EnterpriseAgentState) -> EnterpriseAgentState:
        query = state["query"]
        intent = state["intent"]
        amount = self._extract_amount(query)
        fallback_plan = self._rule_based_plan(query, intent, amount)
        plan = fallback_plan

        if settings.AGENT_LLM_PLANNER_ENABLED and intent not in {"policy_qa", "policy_trace"}:
            llm_plan = self._llm_plan(query, intent, amount)
            if llm_plan:
                plan = llm_plan
            else:
                logger.info("[EnterpriseWorkflow] fallback to rule-based plan")

        logger.info("[EnterpriseWorkflow] plan=%s", plan)
        log_trace(
            logger,
            "workflow_plan",
            intent=intent,
            planner="llm_or_rule" if settings.AGENT_LLM_PLANNER_ENABLED else "rule",
            amount=amount,
            tools=",".join(step.get("tool", "") for step in plan),
            steps=len(plan),
        )
        return {**state, "plan": plan}

    def _rule_based_plan(self, query: str, intent: str, amount: float) -> List[Dict[str, Any]]:
        plan_map = {
            "policy_qa": [{"tool": "rag_summarize", "args": {"query": query}}],
            "policy_trace": [{"tool": "locate_policy_clause", "args": {"query": query}}],
            "expense_standard": [
                {"tool": "query_expense_standard", "args": {"expense_type": query}},
                {"tool": "locate_policy_clause", "args": {"query": query}},
            ],
            "risk_check": [
                {"tool": "check_policy_risk", "args": {"application_type": query, "content": query, "amount": amount}},
                {"tool": "locate_policy_clause", "args": {"query": query}},
            ],
            "flow_generation": [
                {"tool": "query_approval_path", "args": {"application_type": query, "amount": amount}},
                {"tool": "rag_summarize", "args": {"query": f"{query} 制度要求 审批流程 所需材料"}},
                {"tool": "check_policy_risk", "args": {"application_type": query, "content": query, "amount": amount}},
                {
                    "tool": "create_application_draft",
                    "args": {"application_type": query, "reason": query, "key_info": f"金额：{amount}"},
                },
            ],
            "report_generation": [
                {"tool": "get_employee_id", "args": {}},
                {"tool": "get_current_month", "args": {}},
                {"tool": "fetch_external_data", "args": {}},
                {"tool": "fill_context_for_report", "args": {}},
            ],
        }
        return plan_map.get(intent, plan_map["policy_qa"])

    def _llm_plan(self, query: str, intent: str, amount: float) -> List[Dict[str, Any]]:
        try:
            raw = self.planner_chain.invoke(
                {
                    "query": query,
                    "intent": intent,
                    "amount": amount,
                    "tool_descriptions": TOOL_DESCRIPTIONS,
                }
            )
            parsed = self._parse_plan_json(raw)
            return self._validate_plan(parsed, query, intent, amount)
        except Exception as exc:
            logger.warning("[EnterpriseWorkflow] LLM planner failed: %s", exc)
            return []

    def _parse_plan_json(self, raw: str) -> List[Dict[str, Any]]:
        try:
            payload = parse_json_array(raw)
        except Exception as exc:
            logger.warning("[EnterpriseWorkflow] failed to parse LLM plan JSON: %s", exc)
            return []
        return [step for step in payload if isinstance(step, dict)]

    def _validate_plan(
        self,
        plan: List[Dict[str, Any]],
        query: str,
        intent: str,
        amount: float,
    ) -> List[Dict[str, Any]]:
        validated = []
        used_tools = set()
        for step in plan[:6]:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool") or step.get("name") or step.get("tool_name")
            if tool_name not in ALLOWED_TOOLS or tool_name in used_tools:
                logger.warning("[EnterpriseWorkflow] ignored invalid or duplicated tool: %s", tool_name)
                continue
            args = step.get("args") or step.get("arguments") or step.get("parameters") or {}
            if not isinstance(args, dict):
                args = {}
            allowed_args = ALLOWED_TOOLS[tool_name]
            cleaned_args = {key: value for key, value in args.items() if key in allowed_args}
            cleaned_args = self._complete_tool_args(tool_name, cleaned_args, query, amount)
            validated.append({"tool": tool_name, "args": cleaned_args})
            used_tools.add(tool_name)

        if not validated:
            return []
        if intent == "flow_generation":
            required_order = ["query_approval_path", "rag_summarize", "check_policy_risk", "create_application_draft"]
            validated = self._ensure_required_tools(validated, required_order, query, amount)
        if intent == "risk_check":
            validated = self._ensure_required_tools(validated, ["check_policy_risk", "locate_policy_clause"], query, amount)
        if intent == "expense_standard":
            validated = self._ensure_required_tools(validated, ["query_expense_standard", "locate_policy_clause"], query, amount)
        return validated

    def _complete_tool_args(self, tool_name: str, args: Dict[str, Any], query: str, amount: float) -> Dict[str, Any]:
        args = {
            key: self._stringify_arg(value) if key not in {"amount", "top_k"} else value
            for key, value in args.items()
        }
        if "amount" in args:
            args["amount"] = self._coerce_float(args.get("amount"), default=amount)
        if "top_k" in args:
            args["top_k"] = self._coerce_int(args.get("top_k"), default=3, minimum=1, maximum=10)
        if tool_name in {"rag_summarize", "locate_policy_clause"}:
            args.setdefault("query", query)
        elif tool_name == "query_approval_path":
            args.setdefault("application_type", query)
            args.setdefault("amount", amount)
        elif tool_name == "query_expense_standard":
            args.setdefault("expense_type", query)
        elif tool_name == "check_policy_risk":
            args.setdefault("application_type", query)
            args.setdefault("content", query)
            args.setdefault("amount", amount)
        elif tool_name == "create_application_draft":
            args.setdefault("application_type", query)
            args.setdefault("reason", query)
            args.setdefault("key_info", f"金额：{amount}")
        return args

    def _coerce_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _coerce_int(self, value: Any, default: int = 3, minimum: int = 1, maximum: int = 10) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(minimum, min(number, maximum))

    def _stringify_arg(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _ensure_required_tools(
        self,
        plan: List[Dict[str, Any]],
        required_tools: List[str],
        query: str,
        amount: float,
    ) -> List[Dict[str, Any]]:
        by_tool = {step["tool"]: step for step in plan}
        merged = []
        for tool_name in required_tools:
            step = by_tool.get(tool_name)
            if not step:
                step = {"tool": tool_name, "args": self._complete_tool_args(tool_name, {}, query, amount)}
            merged.append(step)
        for step in plan:
            if step["tool"] not in required_tools:
                merged.append(step)
        return merged

    def tool_executor_node(self, state: EnterpriseAgentState) -> EnterpriseAgentState:
        results = {}
        context = {}
        for step in state.get("plan", []):
            tool_name = step["tool"]
            args = dict(step.get("args", {}))
            tool_start = now_ms()
            try:
                args = self._enrich_args_from_context(tool_name, args, state["query"], results)
                if tool_name == "fetch_external_data":
                    args.setdefault("employee_id", context.get("employee_id", "E1001"))
                    args.setdefault("month", context.get("month", get_current_month.invoke({})))
                log_trace(
                    logger,
                    "workflow_tool_start",
                    tool=tool_name,
                    args_keys=",".join(sorted(args.keys())),
                )
                result = self._invoke_tool(tool_name, args)
                results[tool_name] = result
                log_trace(
                    logger,
                    "workflow_tool_done",
                    tool=tool_name,
                    result_chars=len(result or ""),
                    elapsed_ms=elapsed_ms(tool_start),
                )
                if tool_name == "get_employee_id":
                    context["employee_id"] = result
                if tool_name == "get_current_month":
                    context["month"] = result
            except Exception as exc:
                logger.error("[EnterpriseWorkflow] tool %s failed: %s", tool_name, exc, exc_info=True)
                log_trace(
                    logger,
                    "workflow_tool_failed",
                    tool=tool_name,
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=elapsed_ms(tool_start),
                )
                results[tool_name] = self._tool_failure_fallback(tool_name, args, state["query"], exc, results)
        return {**state, "tool_results": results}

    def _tool_failure_fallback(
        self,
        tool_name: str,
        args: Dict[str, Any],
        query: str,
        exc: Exception,
        results: Dict[str, str],
    ) -> str:
        error = f"{type(exc).__name__}: {exc}"
        try:
            if tool_name == "rag_summarize":
                clause = locate_policy_clause.invoke({"query": args.get("query") or query, "top_k": 3})
                return f"工具 rag_summarize 执行失败，已降级为条款定位结果。\n失败原因：{error}\n{clause}"
            if tool_name == "locate_policy_clause":
                summary = rag_summarize.invoke({"query": args.get("query") or query})
                return f"工具 locate_policy_clause 执行失败，已降级为制度摘要结果。\n失败原因：{error}\n{summary}"
            if tool_name == "query_approval_path":
                return (
                    "审批路径工具执行失败，已启用保守兜底：请先提交直属主管审核，"
                    "再由对应部门负责人、财务或总经理按实际制度确认。提交前必须人工核对制度原文。"
                    f"\n失败原因：{error}"
                )
            if tool_name == "query_expense_standard":
                basis = locate_policy_clause.invoke({"query": f"{query} 报销标准 费用标准 发票", "top_k": 3})
                return f"费用标准工具执行失败，已降级检索制度依据。\n失败原因：{error}\n{basis}"
            if tool_name == "check_policy_risk":
                return (
                    "风险校验工具执行失败，已启用保守兜底：当前事项需要人工复核制度依据、审批权限、"
                    "金额阈值、附件材料和是否存在先执行后补审批风险。"
                    f"\n失败原因：{error}"
                )
            if tool_name == "create_application_draft":
                return (
                    f"申请草稿工具执行失败，已生成最小草稿。\n"
                    f"申请事项：{args.get('application_type') or query}\n"
                    f"申请原因：{args.get('reason') or query}\n"
                    f"关键信息：{args.get('key_info') or '资料未明确，提交前请补充制度依据和附件。'}\n"
                    f"失败原因：{error}"
                )
            if tool_name == "fetch_external_data":
                return f"外部数据工具执行失败，已跳过外部数据补充。失败原因：{error}"
            if tool_name == "get_employee_id":
                return "E1001"
            if tool_name == "get_employee_department":
                return "研发部"
            if tool_name == "get_current_month":
                return get_current_month.invoke({})
            if tool_name == "fill_context_for_report":
                return "报告上下文工具执行失败，已跳过上下文补充。"
        except Exception as fallback_exc:
            logger.error(
                "[EnterpriseWorkflow] fallback for tool %s failed: %s",
                tool_name,
                fallback_exc,
                exc_info=True,
            )
            return (
                f"工具 {tool_name} 执行失败，兜底也失败。"
                f"原始错误：{error}；兜底错误：{type(fallback_exc).__name__}: {fallback_exc}。"
                "请人工核对制度依据后再处理。"
            )
        return f"工具 {tool_name} 执行失败：{error}。请人工核对制度依据后再处理。"

    def _enrich_args_from_context(
        self,
        tool_name: str,
        args: Dict[str, Any],
        query: str,
        results: Dict[str, str],
    ) -> Dict[str, Any]:
        if tool_name == "check_policy_risk":
            basis = results.get("rag_summarize")
            if basis:
                args["content"] = self._join_limited([args.get("content", query), basis], limit=900)

        if tool_name == "create_application_draft":
            key_parts = [str(args.get("key_info") or "")]
            if results.get("query_approval_path"):
                key_parts.append(f"审批路径：{results['query_approval_path']}")
            if results.get("rag_summarize"):
                key_parts.append(f"制度摘要：{results['rag_summarize']}")
            if results.get("check_policy_risk"):
                key_parts.append(f"风险校验：{results['check_policy_risk']}")
            args["key_info"] = self._join_limited(key_parts, limit=1200)
        return args

    def _join_limited(self, parts: List[str], limit: int) -> str:
        text = "\n".join(part for part in parts if part)
        return text[:limit]

    def risk_review_node(self, state: EnterpriseAgentState) -> EnterpriseAgentState:
        query = state["query"]
        results = dict(state.get("tool_results", {}))
        review_notes = []
        if state.get("intent") in {"flow_generation", "expense_standard"} and "check_policy_risk" not in results:
            amount = self._extract_amount(query)
            args = {"application_type": query, "content": query, "amount": amount}
            try:
                results["check_policy_risk"] = check_policy_risk.invoke(args)
            except Exception as exc:
                logger.error("[EnterpriseWorkflow] risk review tool failed: %s", exc, exc_info=True)
                results["check_policy_risk"] = self._tool_failure_fallback(
                    "check_policy_risk", args, query, exc, results
                )
        risk_result = results.get("check_policy_risk", "")
        if "风险等级：高" in risk_result:
            review_notes.append("检测到高风险事项，最终回答需要提示用户补充材料或先进行人工确认。")
        if state.get("intent") in {"flow_generation", "risk_check"} and "locate_policy_clause" not in results:
            if not results.get("rag_summarize") or "未检索" in results.get("rag_summarize", ""):
                args = {"query": query, "top_k": 3}
                try:
                    results["locate_policy_clause"] = locate_policy_clause.invoke(args)
                    review_notes.append("制度摘要不足，已补充执行条款定位。")
                except Exception as exc:
                    logger.error("[EnterpriseWorkflow] clause fallback failed: %s", exc, exc_info=True)
                    results["locate_policy_clause"] = self._tool_failure_fallback(
                        "locate_policy_clause", args, query, exc, results
                    )
        if review_notes:
            results["review_notes"] = "\n".join(review_notes)
        log_trace(
            logger,
            "workflow_risk_review",
            intent=state.get("intent"),
            review_notes=len(review_notes),
            tools=",".join(results.keys()),
        )
        return {**state, "tool_results": results, "review_notes": review_notes}

    def final_answer_node(self, state: EnterpriseAgentState) -> EnterpriseAgentState:
        start = now_ms()
        answer = self.final_chain.invoke(
            {
                "query": state["query"],
                "tool_results": json.dumps(state.get("tool_results", {}), ensure_ascii=False, indent=2),
            }
        )
        log_trace(
            logger,
            "workflow_final_answer",
            intent=state.get("intent"),
            answer_chars=len(answer or ""),
            elapsed_ms=elapsed_ms(start),
        )
        return {**state, "final_answer": answer}

    def _invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        tools = {
            "rag_summarize": rag_summarize,
            "get_employee_id": get_employee_id,
            "get_employee_department": get_employee_department,
            "get_current_month": get_current_month,
            "fetch_external_data": fetch_external_data,
            "create_application_draft": create_application_draft,
            "query_approval_path": query_approval_path,
            "locate_policy_clause": locate_policy_clause,
            "check_policy_risk": check_policy_risk,
            "query_expense_standard": query_expense_standard,
            "fill_context_for_report": fill_context_for_report,
        }
        tool = tools[tool_name]
        return tool.invoke(args)

    def _extract_amount(self, text: str) -> float:
        return extract_amount(text)
