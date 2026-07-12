import csv
import json
from datetime import datetime

from langchain_core.tools import tool

from AIRAGAgent.rag.rag_service import RagSummarizeService
from AIRAGAgent.utils.config_handler import agent_config
from AIRAGAgent.utils.logger_handler import logger
from AIRAGAgent.utils.path_tool import get_abs_path

rag_service = None
external_data = {}
DEMO_EMPLOYEE_ID = "E1001"
DEMO_DEPARTMENT = "研发部"


def get_rag_service() -> RagSummarizeService:
    global rag_service
    if rag_service is None:
        rag_service = RagSummarizeService()
    return rag_service


@tool(description="从企业制度知识库中检索参考资料，适用于报销、请假、采购、入职、信息安全、工单SOP等制度问答")
def rag_summarize(query: str) -> str:
    return get_rag_service().rag_summarize(query)


@tool(description="获取当前员工ID，以纯字符串形式返回。演示环境固定返回E1001")
def get_employee_id() -> str:
    return DEMO_EMPLOYEE_ID


@tool(description="获取当前员工所属部门，以纯字符串形式返回。演示环境固定返回研发部")
def get_employee_department() -> str:
    return DEMO_DEPARTMENT


@tool(description="获取当前月份，格式为YYYY-MM，以纯字符串形式返回")
def get_current_month() -> str:
    return datetime.now().strftime("%Y-%m")


def generate_external_data():
    if external_data:
        return

    external_data_path = get_abs_path(agent_config["external_data_path"])
    with open(external_data_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            employee_id = row["employee_id"]
            month = row["month"]
            external_data.setdefault(employee_id, {})
            external_data[employee_id][month] = {
                "employee_id": employee_id,
                "employee_name": row["employee_name"],
                "department": row["department"],
                "month": month,
                "top_questions": row["top_questions"],
                "pending_requests": row["pending_requests"],
                "risk_notes": row["risk_notes"],
                "suggestion": row["suggestion"],
            }


@tool(description="从外部系统中获取指定员工在指定月份的制度咨询统计记录，入参为employee_id和month，未检索到则返回空字符串")
def fetch_external_data(employee_id: str, month: str) -> str:
    generate_external_data()

    try:
        return json.dumps(external_data[employee_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到员工：{employee_id}在{month}的制度咨询统计记录")
        return ""


@tool(description="根据申请类型、申请原因和关键字段生成企业内部流程申请草稿，适用于报销、请假、采购、权限开通等场景")
def create_application_draft(application_type: str, reason: str, key_info: str) -> str:
    return (
        f"申请类型：{application_type}\n"
        f"申请人：{DEMO_EMPLOYEE_ID}\n"
        f"所属部门：{DEMO_DEPARTMENT}\n"
        f"申请原因：{reason}\n"
        f"关键信息：{key_info}\n"
        "审批建议：请申请人补充对应制度要求的附件，并提交直属主管初审。"
    )


@tool(description="根据申请类型、金额和部门判断企业内部审批路径，适用于采购、报销、请假、权限开通等流程类问题")
def query_approval_path(application_type: str, amount: float = 0, department: str = DEMO_DEPARTMENT) -> str:
    app_type = application_type.strip()
    steps = ["直属主管"]

    if any(word in app_type for word in ["采购", "合同"]):
        steps.append("部门负责人")
        if amount >= 5000:
            steps.append("财务审核")
        if amount >= 10000:
            steps.append("总经理审批")
        required = "采购申请说明、预算依据、供应商信息、报价单"
    elif any(word in app_type for word in ["报销", "差旅", "费用"]):
        steps.append("财务审核")
        if amount >= 3000:
            steps.append("部门负责人复核")
        required = "发票、付款凭证、费用明细、审批单"
    elif any(word in app_type for word in ["请假", "病假", "年假"]):
        if "病假" in app_type:
            steps.append("HR备案")
        required = "请假事由、请假时间、必要证明材料"
    elif any(word in app_type for word in ["权限", "账号", "数据"]):
        steps.extend(["系统负责人", "信息安全管理员"])
        required = "权限用途、有效期、最小权限说明、负责人确认"
    else:
        steps.append("部门负责人")
        required = "申请说明、相关附件"

    return (
        f"申请类型：{application_type}\n"
        f"申请金额：{amount}\n"
        f"所属部门：{department}\n"
        f"建议审批路径：{' → '.join(steps)}\n"
        f"建议准备材料：{required}"
    )


@tool(description="定位企业制度条款依据，返回来源文件、分片序号和命中片段，适用于需要可追溯依据的问题")
def locate_policy_clause(query: str, top_k: int = 3) -> str:
    docs = get_rag_service().retrieve_documents(query, top_k=top_k)
    if not docs:
        return "未定位到明确制度条款。"

    items = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "-")
        preview = doc.page_content.strip().replace("\n", " ")[:260]
        items.append(f"{index}. 来源：{source}；分片：{chunk_index}；片段：{preview}")
    return "\n".join(items)


@tool(description="根据申请内容进行制度风险校验，识别超标准报销、长期权限、缺少证明、越级审批等风险，并给出整改建议")
def check_policy_risk(application_type: str, content: str, amount: float = 0) -> str:
    risks = []
    text = f"{application_type} {content}"

    if amount >= 10000 and any(word in text for word in ["采购", "报销", "费用"]):
        risks.append(("高", "金额较高，可能需要财务和总经理审批", "补充预算依据、报价单和业务必要性说明。"))
    if any(word in text for word in ["长期权限", "永久权限", "数据库权限", "生产权限"]):
        risks.append(("高", "权限申请可能违反最小权限和临时授权原则", "明确权限范围和有效期，改为临时授权并增加负责人审批。"))
    if "病假" in text and not any(word in text for word in ["证明", "病历", "诊断"]):
        risks.append(("中", "病假申请可能缺少必要证明材料", "补充医院证明、病历或诊断材料。"))
    if any(word in text for word in ["先执行", "后补", "紧急采购", "补审批"]):
        risks.append(("中", "存在事后补审批或流程倒置风险", "说明紧急原因，并补充直属主管确认记录。"))
    if not risks:
        risks.append(("低", "未发现明显制度风险", "按对应制度流程提交审批并保留附件。"))

    risk_lines = [f"- 风险等级：{level}；原因：{reason}；建议：{advice}" for level, reason, advice in risks]
    return "风险校验结果：\n" + "\n".join(risk_lines)


@tool(description="查询费用或报销标准，适用于住宿、交通、餐补、差旅、发票等费用标准问题")
def query_expense_standard(expense_type: str, city_level: str = "普通城市") -> str:
    standards = {
        "住宿": {"一线城市": "不超过600元/晚", "普通城市": "不超过400元/晚"},
        "交通": {"一线城市": "按实际合规票据报销", "普通城市": "按实际合规票据报销"},
        "餐补": {"一线城市": "不超过100元/天", "普通城市": "不超过80元/天"},
        "差旅": {"一线城市": "按住宿、交通、餐补分类执行", "普通城市": "按住宿、交通、餐补分类执行"},
    }
    matched_type = next((key for key in standards if key in expense_type), expense_type)
    standard = standards.get(matched_type, {}).get(city_level)
    if not standard and matched_type in standards:
        standard = standards[matched_type].get("普通城市")
    if not standard:
        standard = "未在规则表中配置明确标准，请以制度检索结果为准。"

    basis = get_rag_service().rag_summarize(f"{expense_type} {city_level} 报销标准 发票 要求")
    return (
        f"费用类型：{expense_type}\n"
        f"城市级别：{city_level}\n"
        f"规则表标准：{standard}\n"
        f"制度依据摘要：{basis}"
    )


@tool(description="无入参。调用后触发中间件将后续提示词切换到制度咨询月报生成场景")
def fill_context_for_report():
    return "fill_context_for_report已调用"
