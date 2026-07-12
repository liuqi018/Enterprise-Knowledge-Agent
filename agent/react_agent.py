from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk

from AIRAGAgent.model.factory import chat_model
from AIRAGAgent.agent.tools.agent_tools import(
    rag_summarize,
    get_employee_id,
    get_employee_department,
    get_current_month,
    fetch_external_data,
    create_application_draft,
    query_approval_path,
    locate_policy_clause,
    check_policy_risk,
    query_expense_standard,
    fill_context_for_report,
)
from AIRAGAgent.agent.tools.middleware import minotor_tool,log_before_model,report_prompt_switch
from AIRAGAgent.utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.agent=create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                rag_summarize,
                get_employee_id,
                get_employee_department,
                get_current_month,
                fetch_external_data,
                create_application_draft,
                query_approval_path,
                locate_policy_clause,
                check_policy_risk,
                query_expense_standard,
                fill_context_for_report,
            ],
            middleware=[minotor_tool,log_before_model,report_prompt_switch],
        )
    def execute_stream(self,query:str,history_messages=None):
        messages = []
        for message in history_messages or []:
            messages.append({"role": message["role"], "content": message["content"]})
        messages.append({"role":"user","content":query})
        input_dict={

            "messages":messages
        }
        # stream_mode="messages" yields LLM message chunks instead of full graph state snapshots.
        try:
            for event in self.agent.stream(input_dict, stream_mode="messages", context={"report": False}):
                message = event[0] if isinstance(event, tuple) else event
                if not isinstance(message, (AIMessage, AIMessageChunk)):
                    continue
                if getattr(message, "tool_call_chunks", None):
                    continue
                content = self._message_content_to_text(message.content)
                if not content or content.strip() == query.strip():
                    continue
                yield content
            return
        except Exception:
            # Some LangChain versions do not support token message streaming for agents.
            # Fall back to value snapshots, but still only emit AI-message deltas.
            previous_content = ""
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
                latest_message = chunk["messages"][-1]
                if not isinstance(latest_message, AIMessage):
                    continue
                content = self._message_content_to_text(latest_message.content)
                if not content or content.strip() == query.strip():
                    continue
                if content.startswith(previous_content):
                    delta = content[len(previous_content):]
                else:
                    delta = content
                previous_content = content
                if delta.strip():
                    yield delta

    def _message_content_to_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content or "")
if __name__=="__main__":
    agent=ReactAgent()
    for chunk in agent.execute_stream("给我生成本月制度咨询报告"):
        print(chunk,end="",flush=True)
