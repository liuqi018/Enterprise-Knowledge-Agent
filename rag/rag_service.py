import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from AIRAGAgent.config.settings import settings
from AIRAGAgent.knowledge.loader import iter_source_files, load_documents
from AIRAGAgent.knowledge.service import KnowledgeBaseService
from AIRAGAgent.knowledge.splitter import split_documents
from AIRAGAgent.model.factory import chat_model
from AIRAGAgent.schemas import RagResponse
from AIRAGAgent.services.query_classifier import classify_query
from AIRAGAgent.utils.logger_handler import logger
from AIRAGAgent.utils.path_tool import get_abs_path
from AIRAGAgent.utils.trace import elapsed_ms, log_trace, now_ms, short_text


UNSUPPORTED_SPECIFIC_TERMS = [
    "\u8fdc\u7a0b\u529e\u516c",
    "\u8fdc\u7a0b\u529e\u516c\u8865\u8d34",
    "\u5496\u5561\u673a",
    "\u4e2a\u4eba\u7f51\u76d8",
    "\u5ba0\u7269",
    "\u5e26\u732b",
    "\u661f\u5ea7",
    "\u8840\u578b",
    "\u5fae\u4fe1",
    "\u53e3\u5934\u786e\u8ba4",
    "\u5916\u90e8\u4ee3\u7801\u6258\u7ba1",
    "\u5b9e\u4e60\u751f\u5355\u72ec\u5ba1\u6279",
    "\u6c38\u4e0d\u8fc7\u671f",
    "\u6c38\u4e45\u4e0d\u8fc7\u671f",
    "\u751f\u65e5\u5f53\u5929",
    "\u81ea\u52a8\u83b7\u5f97",
    "\u5e26\u85aa\u5047",
    "\u652f\u4ed8\u5b9d",
    "\u4e2a\u4eba\u652f\u4ed8\u5b9d",
]

SPECIFIC_UNSUPPORTED_PATTERN = re.compile(
    r"(\d+\s*(?:\u5143|\u5929|\u4e07\u5143)|\u6bcf\u5468[一二三四五六日天]|"
    r"\u4ee3\u66ff\u7cfb\u7edf\u5ba1\u6279|\u5355\u72ec\u5ba1\u6279)"
)

NO_DIRECT_BASIS_ANSWER = (
    "\u77e5\u8bc6\u5e93\u4e2d\u6ca1\u6709\u660e\u786e\u4f9d\u636e\u3002"
    "\u68c0\u7d22\u8d44\u6599\u672a\u76f4\u63a5\u652f\u6301\u8be5\u95ee\u9898\u4e2d\u7684\u5177\u4f53\u6761\u4ef6\u6216\u8bf4\u6cd5\uff0c"
    "\u8bf7\u8865\u5145\u5236\u5ea6\u540d\u79f0\u3001\u4e1a\u52a1\u573a\u666f\u6216\u54a8\u8be2\u5bf9\u5e94\u90e8\u95e8\u3002"
)


ANSWER_PROMPT = """你是企业制度知识库助手。只能基于“制度资料”回答用户问题。

证据使用规则：
1. 只有当资料与问题主题完全不相关，或没有任何可用于回答问题的片段时，才回答“知识库中没有明确依据。”
2. 如果资料只能支持部分答案，不要拒答；先回答“资料明确的部分”，再说明“资料未明确的部分”。
3. 不得编造制度名称、审批节点、材料清单、金额、天数、责任部门、例外条件或处罚标准。
4. 问什么答什么：不要默认扩展申请草稿、风险清单、材料清单或完整流程，除非用户问题明确需要。
5. 依据必须标注资料编号和来源文件，例如“依据：[资料1] xxx.txt”。

用户问题：{question}

制度资料：
{context}
"""

DIRECT_POLICY_PROMPT = """你是企业制度问答助手。只能基于给定的制度资料回答，不得编造。
回答前先判断资料是否直接支持用户问题：
- 如果资料中有直接片段能够回答问题，即使不完整，也必须回答可确认部分，不要说“没有明确依据”。
- 只有资料完全无法支持问题时，才输出：“知识库中没有明确依据。”
- 金额、天数、审批角色、责任部门、例外条件必须来自资料原文。
- 不要补充用户没有问的流程、材料、风险或申请草稿。

输出格式：
结论：用 1-2 句话直接回答问题。
依据：标注资料编号和来源文件；如果章节存在，也写出章节。

用户问题：{question}

制度资料：
{context}
"""

POLICY_ANSWER_PROMPT = """你是制造企业制度知识库助手。只能基于给定制度资料回答用户问题，不能编造资料中没有的制度、金额、天数、审批角色、责任部门或例外条件。

证据判断：
- 如果资料中有直接片段能回答问题，即使只是一个条款、一句话或一组记录要求，也必须回答可确认部分。
- 如果资料只支持部分答案，使用“资料明确：...”和“资料未明确：...”区分，不要整体拒答。
- 只有资料与问题完全无关，或无法形成任何有效答案时，才回答：“知识库中没有明确依据。请补充制度名称、业务场景或咨询对应部门。”

回答策略：
- 问题很泛或缺少对象时，不要直接拒答；先说明需要补充的关键信息，再列出资料中已经明确的通用规则或限制。
- 问“能不能先执行再补流程/先补签/先补检/先上线”时，优先检查资料是否已有审批、审核、评审、检验、记录、签字、发货或发布条件；只要有相关要求，就给出基于资料的保守结论，并说明资料未明确的例外条件。
- 问“是否列出全部/每个阶段/完整清单”时，不要编造完整清单；列出资料中已有项目，并明确资料不能证明已覆盖全部。
- 问敏感资料、薪酬、权限、研发资料外发时，优先回答访问范围、保密限制、审批/授权要求；资料未明确的权限边界单独写在“资料未明确”。
- 问职责/部门：只回答责任主体和依据，不要展开完整流程。
- 问是否可以/是否需要：先给明确结论，再列依据。
- 问包含什么/覆盖哪些：列出资料中出现的要点，不要说“完整流程未明确”作为开头。
- 问信息安全控制点/管理手册覆盖范围：优先提取资料中的安全目标、控制对象和控制措施，例如业务中断防护、数据丢失防护、敏感信息泄密防护、风险评估、风险处置、访问/权限控制、备份、保密、文件记录管理；只列资料支持的项目。
- 问流程/步骤：按资料列步骤；缺失环节写“资料未明确”，不要自行补齐。
- 问申请草稿：只有问题明确要求“生成草稿”时才生成草稿，否则只说明应包含的信息。

输出格式：
结论：直接回答用户问题。
依据与要点：列出资料明确支持的要求、职责、记录或控制点。
资料未明确：仅列出与问题直接相关但资料没有说明的内容；如果没有，可省略。
来源：标注资料编号、来源文件和章节标题；没有章节时只写来源文件。

约束：
- 不输出 JSON。
- 不要使用“通常、一般、可能”等词补充资料外内容。
- 不要默认输出“事项概述、办理步骤、所需材料、风险提示、申请草稿”等固定模板。
- 回答控制在 450 字以内。

用户问题：{question}

制度资料：
{context}
"""

PROCESS_ANSWER_PROMPT = """你是制造企业流程合规 Agent。只能基于给定制度资料生成流程方案、风险提示或申请草稿。

证据判断：
- 如果资料能支持部分流程，不要整体拒答；直接列出资料明确的步骤，并单独说明缺失环节。
- 只有资料与问题完全无关时，才回答“知识库中没有明确依据。”
- 不得补造审批链、材料清单、金额阈值、时限或责任人。

输出格式：
处理原则：如果用户问“能否先执行再补流程/先补签/先补检/先上线”，先依据资料中的审批、审核、评审、检验、记录或签字要求给出保守结论；资料只支持部分时，必须区分“资料明确”和“资料未明确”。
事项概述：用 1 句话说明用户要办理或处理的事项。
办理步骤：最多 6 步，只列资料明确的申请、上报、检查、审核、批准、执行、记录等环节。
资料未明确：列出资料缺失的关键环节，例如具体审批人、表单、时限、金额阈值。
责任/材料/记录：只列资料明确出现的责任部门、责任人、凭证、单据、附件、记录或报告。
合规风险：只列资料直接支持的风险；不要把常识风险当成制度要求。
申请草稿：仅当用户明确要求生成草稿时输出；否则不要输出该项。
来源：标注资料编号、来源文件和章节标题。

用户问题：{question}

制度资料：
{context}
"""

MULTI_QUERY_PROMPT = """请把用户的复杂流程问题改写为 3 个适合企业制度知识库检索的中文查询。
要求：
1. 覆盖适用条件、审批步骤、所需材料、风险注意事项。
2. 只返回 JSON 数组，不要解释。

用户问题：{question}
"""


class RagSummarizeService:
    _vector_store_ready = False
    _shared_bm25_documents_by_tenant: Dict[str, List[Document]] = {}
    _shared_bm25_manifest_by_tenant: Dict[str, Dict[str, str]] = {}
    _shared_bm25_index_by_tenant: Dict[str, Dict[str, Any]] = {}

    def __init__(self):
        self.knowledge_service = KnowledgeBaseService()
        self.vector_store = self.knowledge_service.vector_store
        self.prompt_template = PromptTemplate.from_template(ANSWER_PROMPT)
        self.chain = self.prompt_template | chat_model | StrOutputParser()
        self.answer_chains = {
            "direct_policy": PromptTemplate.from_template(DIRECT_POLICY_PROMPT) | chat_model | StrOutputParser(),
            "professional_policy": PromptTemplate.from_template(POLICY_ANSWER_PROMPT) | chat_model | StrOutputParser(),
            "complex_process": PromptTemplate.from_template(PROCESS_ANSWER_PROMPT) | chat_model | StrOutputParser(),
            "chat": PromptTemplate.from_template(POLICY_ANSWER_PROMPT) | chat_model | StrOutputParser(),
        }
        self.multi_query_chain = PromptTemplate.from_template(MULTI_QUERY_PROMPT) | chat_model | StrOutputParser()
        self._bm25_documents_by_tenant = self._shared_bm25_documents_by_tenant
        self._bm25_manifest_by_tenant = self._shared_bm25_manifest_by_tenant
        self._bm25_index_by_tenant = self._shared_bm25_index_by_tenant
        self._bm25_failed_sources: Dict[str, str] = {}
        self.bm25_corpus_path = get_abs_path(settings.BM25_CORPUS_PATH)

    def ensure_vector_store_ready(self) -> None:
        if RagSummarizeService._vector_store_ready:
            return
        if self.vector_store.count() > 0:
            RagSummarizeService._vector_store_ready = True
            return
        logger.warning("[RAG] vector store is empty, ingesting enterprise documents")
        self.knowledge_service.ingest(force=True)
        RagSummarizeService._vector_store_ready = True

    def classify_intent(self, query: str) -> str:
        route = classify_query(query)
        return route.retrieval_mode

    def rewrite_query(self, query: str, intent: str) -> List[str]:
        if intent == "complex_process":
            return self.process_rule_rewrite(query)
        if intent == "professional_policy":
            return self.policy_rule_rewrite(query)
        return [query.strip()]

    def process_rule_rewrite(self, query: str) -> List[str]:
        variants = [
            query.strip(),
            f"{query} 审批流程 所需材料 制度要求",
            f"{query} 风险注意事项 预算依据 报价单",
        ]
        return self.unique_list(variants)[:3]

    def policy_rule_rewrite(self, query: str) -> List[str]:
        variants = [query.strip()]
        compact = query.replace("怎么", "如何").replace("哪些", "什么")
        if compact not in variants:
            variants.append(compact)
        domain = self.infer_query_domain(query)
        if domain:
            variants.append(f"{query} 企业制度 规则 标准 适用条件")
        if any(word in query for word in ["流程", "申请", "审批", "办理"]):
            variants.append(f"{query} 审批步骤 所需材料 注意事项")
        return self.unique_list(variants)[:4]

    def llm_multi_query_rewrite(self, query: str) -> List[str]:
        try:
            raw = self.multi_query_chain.invoke({"question": query})
            parsed = json.loads(self.extract_json_array(raw))
            variants = [query.strip()] + [str(item).strip() for item in parsed if str(item).strip()]
            return self.unique_list(variants)[:4]
        except Exception as exc:
            logger.warning("[QueryRewrite] LLM rewrite failed, fallback to policy rules: %s", exc)
            return self.policy_rule_rewrite(query)

    def retrieve_documents(self, query: str, top_k: int = None, tenant_id: str = None, trace_id: str = None) -> List[Document]:
        retrieve_start = now_ms()
        self.ensure_vector_store_ready()
        tenant_id = tenant_id or "global"
        intent = self.classify_intent(query)
        metadata_filter = self.build_metadata_filter(query, intent)
        recalled_lists: List[Tuple[str, List[Document]]] = []
        log_trace(
            logger,
            "rag_retrieve_start",
            trace_id=trace_id,
            tenant_id=tenant_id,
            intent=intent,
            top_k=top_k,
            metadata_filter=metadata_filter,
            query=short_text(query),
        )

        if intent == "direct_policy":
            target_top_k = top_k or self.answer_top_k(intent)
            vector_k = max(settings.VECTOR_RECALL_TOP_K, target_top_k * 2)
            selected = self.vector_recall(query.strip(), vector_k, metadata_filter=None, trace_id=trace_id)
            selected = self.domain_aware_rerank(query, selected)
            selected = selected[:target_top_k]
            rank_mode = "vector_only"
            compressed = self.compress_context(query, selected, intent)
            logger.info(
                "[RAG] intent=%s queries=1 metadata_filter=None recall_lists=1 fused=%s filtered=%s selected=%s rank_mode=%s",
                intent,
                len(selected),
                len(selected),
                len(compressed),
                rank_mode,
            )
            log_trace(
                logger,
                "rag_retrieve_done",
                trace_id=trace_id,
                intent=intent,
                queries=1,
                recall_lists=1,
                fused=len(selected),
                filtered=len(selected),
                selected=len(compressed),
                rank_mode=rank_mode,
                elapsed_ms=elapsed_ms(retrieve_start),
            )
            return compressed

        queries = self.rewrite_query(query, intent)
        target_top_k = top_k or self.answer_top_k(intent)
        if intent == "professional_policy":
            queries = [query.strip()]
            vector_k = max(settings.VECTOR_RECALL_TOP_K, target_top_k * 2)
            bm25_k = max(settings.BM25_RECALL_TOP_K, target_top_k * 2)
        else:
            queries = queries[:2]
            vector_k = max(settings.VECTOR_RECALL_TOP_K, target_top_k * 2)
            bm25_k = max(settings.BM25_RECALL_TOP_K, target_top_k * 2)

        for rewritten in queries:
            recalled_lists.append(("vector", self.vector_recall(rewritten, vector_k, metadata_filter, trace_id=trace_id)))
            recalled_lists.append(("bm25", self.bm25_recall(rewritten, bm25_k, metadata_filter, tenant_id=tenant_id, trace_id=trace_id)))
            if metadata_filter:
                fallback_k = max(target_top_k, 3)
                recalled_lists.append(("vector_fallback", self.vector_recall(rewritten, fallback_k, metadata_filter=None, trace_id=trace_id)))
                recalled_lists.append(("bm25_fallback", self.bm25_recall(rewritten, fallback_k, metadata_filter=None, tenant_id=tenant_id, trace_id=trace_id)))

        fused = self.weighted_reciprocal_rank_fusion(recalled_lists)
        filtered = self.light_filter(query, fused)
        filtered = self.strict_domain_filter(query, filtered, metadata_filter)
        filtered = self.domain_aware_rerank(query, filtered)
        if intent == "complex_process" and settings.ENABLE_COMPLEX_RERANK and settings.RERANK_PROVIDER != "none":
            selected = self.rerank(query, filtered)
            rank_mode = "rerank"
        else:
            selected = filtered[:target_top_k]
            rank_mode = "weighted_rrf"
        compressed = self.compress_context(query, selected[:target_top_k], intent)
        logger.info(
            "[RAG] intent=%s queries=%s metadata_filter=%s recall_lists=%s fused=%s filtered=%s selected=%s rank_mode=%s",
            intent,
            len(queries),
            metadata_filter,
            len(recalled_lists),
            len(fused),
            len(filtered),
            len(selected),
            rank_mode,
        )
        log_trace(
            logger,
            "rag_retrieve_done",
            trace_id=trace_id,
            intent=intent,
            queries=len(queries),
            metadata_filter=metadata_filter,
            recall_lists=len(recalled_lists),
            fused=len(fused),
            filtered=len(filtered),
            selected=len(compressed),
            rank_mode=rank_mode,
            elapsed_ms=elapsed_ms(retrieve_start),
        )
        return compressed

    def answer_top_k(self, intent: str) -> int:
        if intent == "direct_policy":
            return 2
        if intent == "complex_process":
            return 4
        return 3

    def vector_recall(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
        trace_id: str = None,
    ) -> List[Document]:
        start = now_ms()
        docs = self.vector_store.similarity_search(query, k=k, metadata_filter=metadata_filter)
        for rank, doc in enumerate(docs, start=1):
            doc.metadata = {**doc.metadata, "recall_route": "vector", "vector_rank": rank}
        log_trace(
            logger,
            "rag_vector_recall",
            trace_id=trace_id,
            k=k,
            results=len(docs),
            metadata_filter=metadata_filter,
            elapsed_ms=elapsed_ms(start),
            query=short_text(query),
        )
        return docs

    def bm25_recall(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
        tenant_id: str = "default",
        trace_id: str = None,
    ) -> List[Document]:
        start = now_ms()
        index_tenant = self._bm25_tenant_for_query(tenant_id)
        index = self.load_bm25_index(index_tenant)
        documents = index.get("documents", [])
        if metadata_filter:
            candidate_indexes = [
                doc_index
                for doc_index, doc in enumerate(documents)
                if self._metadata_match(doc, metadata_filter)
            ]
        else:
            candidate_indexes = list(range(len(documents)))
        if not candidate_indexes:
            log_trace(
                logger,
                "rag_bm25_recall",
                trace_id=trace_id,
                tenant_id=tenant_id,
                index_tenant=index_tenant,
                k=k,
                results=0,
                reason="no_candidates",
                elapsed_ms=elapsed_ms(start),
                query=short_text(query),
            )
            return []

        query_terms = self.tokenize(query)
        if not query_terms:
            log_trace(
                logger,
                "rag_bm25_recall",
                trace_id=trace_id,
                tenant_id=tenant_id,
                index_tenant=index_tenant,
                k=k,
                results=0,
                reason="empty_terms",
                elapsed_ms=elapsed_ms(start),
                query=short_text(query),
            )
            return []

        tokenized_docs = index.get("tokenized_docs", [])
        doc_freq = index.get("doc_freq", {})
        avg_len = index.get("avg_len", 0.0)
        doc_count = index.get("doc_count", len(documents))
        scores = []
        for doc_index in candidate_indexes:
            tokens = tokenized_docs[doc_index]
            score = self.bm25_score(query_terms, tokens, doc_freq, doc_count, avg_len)
            if score > 0:
                scores.append((score, doc_index))

        scores.sort(reverse=True)
        result = []
        for rank, (score, doc_index) in enumerate(scores[:k], start=1):
            doc = documents[doc_index]
            result.append(Document(
                page_content=doc.page_content,
                metadata={
                **doc.metadata,
                "recall_route": "bm25",
                "bm25_rank": rank,
                "bm25_score": round(score, 4),
                },
            ))
        log_trace(
            logger,
            "rag_bm25_recall",
            trace_id=trace_id,
            tenant_id=tenant_id,
            index_tenant=index_tenant,
            k=k,
            candidates=len(candidate_indexes),
            scored=len(scores),
            results=len(result),
            metadata_filter=metadata_filter,
            elapsed_ms=elapsed_ms(start),
            query=short_text(query),
        )
        return result

    def _bm25_tenant_for_query(self, tenant_id: str) -> str:
        if tenant_id == "global":
            return "global"
        global_manifest = self._bm25_manifest_snapshot()
        cached_global = self._bm25_index_by_tenant.get("global")
        if cached_global and cached_global.get("manifest") == global_manifest:
            return "global"
        if not self._tenant_has_bm25_records(tenant_id):
            logger.info("[BM25] tenant=%s uses shared global corpus", tenant_id)
            return "global"
        return tenant_id

    def _bm25_manifest_snapshot(self) -> Dict[str, str]:
        if not os.path.exists(self.bm25_corpus_path):
            return {}
        stat = os.stat(self.bm25_corpus_path)
        return {self.bm25_corpus_path: f"{stat.st_size}:{stat.st_mtime_ns}"}

    def _tenant_has_bm25_records(self, tenant_id: str) -> bool:
        if not os.path.exists(self.bm25_corpus_path):
            return False
        with open(self.bm25_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if f'"tenant_id": "{tenant_id}"' in line or f'"tenant_id":"{tenant_id}"' in line:
                    return True
        return False

    def load_bm25_index(self, tenant_id: str) -> Dict[str, Any]:
        documents = self.load_bm25_documents(tenant_id)
        manifest = self._bm25_manifest_by_tenant.get(tenant_id, {})
        cached = self._bm25_index_by_tenant.get(tenant_id)
        if cached and cached.get("manifest") == manifest:
            return cached

        tokenized_docs = [self.tokenize(doc.page_content) for doc in documents]
        doc_freq = defaultdict(int)
        for tokens in tokenized_docs:
            for term in set(tokens):
                doc_freq[term] += 1
        avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(len(tokenized_docs), 1)
        index = {
            "manifest": dict(manifest),
            "documents": documents,
            "tokenized_docs": tokenized_docs,
            "doc_freq": dict(doc_freq),
            "avg_len": avg_len,
            "doc_count": len(documents),
        }
        self._bm25_index_by_tenant[tenant_id] = index
        logger.info("[BM25] index ready tenant=%s docs=%s", tenant_id, len(documents))
        return index

    def load_bm25_documents(self, tenant_id: str) -> List[Document]:
        cached_documents = self.load_bm25_documents_from_cache(tenant_id)
        if cached_documents is not None:
            return cached_documents
        logger.warning("[BM25] corpus cache missing, skip lexical recall: %s", self.bm25_corpus_path)
        return []

        source_files = iter_source_files(self.knowledge_service.data_path, ("txt", "pdf", "docx", "doc"), calculate_md5=False)
        manifest = {source.path: self._file_signature(source.path) for source in source_files}
        if (
            tenant_id in self._bm25_documents_by_tenant
            and manifest == self._bm25_manifest_by_tenant.get(tenant_id)
        ):
            return self._bm25_documents_by_tenant[tenant_id]

        documents: List[Document] = []
        for source in source_files:
            signature = manifest[source.path]
            if self._bm25_failed_sources.get(source.path) == signature:
                continue
            try:
                chunks = split_documents(load_documents(source))
                for chunk in chunks:
                    chunk.metadata = {**chunk.metadata, "tenant_id": tenant_id}
                documents.extend(chunks)
            except Exception as exc:
                self._bm25_failed_sources[source.path] = signature
                logger.warning("[BM25] failed to load %s: %s", source.path, exc)

        self._bm25_manifest_by_tenant[tenant_id] = manifest
        self._bm25_documents_by_tenant[tenant_id] = documents
        return documents

    def load_bm25_documents_from_cache(self, tenant_id: str) -> Optional[List[Document]]:
        if not os.path.exists(self.bm25_corpus_path):
            return None

        stat = os.stat(self.bm25_corpus_path)
        manifest = {self.bm25_corpus_path: f"{stat.st_size}:{stat.st_mtime_ns}"}
        if (
            tenant_id in self._bm25_documents_by_tenant
            and manifest == self._bm25_manifest_by_tenant.get(tenant_id)
        ):
            return self._bm25_documents_by_tenant[tenant_id]

        documents: List[Document] = []
        with open(self.bm25_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("tenant_id") != tenant_id:
                    continue
                page_content = record.pop("page_content", "")
                if page_content:
                    documents.append(Document(page_content=page_content, metadata=record))

        self._bm25_manifest_by_tenant[tenant_id] = manifest
        self._bm25_documents_by_tenant[tenant_id] = documents
        return documents

    def _file_signature(self, path: str) -> str:
        stat = os.stat(path)
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def weighted_reciprocal_rank_fusion(
        self,
        ranked_lists: List[Tuple[str, List[Document]]],
    ) -> List[Document]:
        scores: Dict[Tuple[str, int], float] = defaultdict(float)
        documents: Dict[Tuple[str, int], Document] = {}
        routes: Dict[Tuple[str, int], set] = defaultdict(set)
        weights = {
            "vector": settings.VECTOR_RRF_WEIGHT,
            "bm25": settings.BM25_RRF_WEIGHT,
            "vector_fallback": settings.VECTOR_RRF_WEIGHT * 0.45,
            "bm25_fallback": settings.BM25_RRF_WEIGHT * 0.35,
        }

        for route, ranked_docs in ranked_lists:
            weight = weights.get(route, 1.0)
            for rank, doc in enumerate(ranked_docs, start=1):
                key = self.document_key(doc)
                scores[key] += weight / (settings.RRF_K + rank)
                documents[key] = doc
                routes[key].add(route)

        fused = []
        for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            doc = documents[key]
            doc.metadata = {
                **doc.metadata,
                "recall_route": "+".join(sorted(routes[key])),
                "weighted_rrf_score": round(score, 6),
            }
            fused.append(doc)
        return fused

    def light_filter(self, query: str, documents: List[Document]) -> List[Document]:
        query_terms = set(self.tokenize(query))

        def keep(doc: Document) -> bool:
            content_terms = set(self.tokenize(doc.page_content))
            return bool(query_terms & content_terms) or float(doc.metadata.get("weighted_rrf_score", 0)) > 0

        kept = [doc for doc in documents if keep(doc)]
        return kept[: settings.LIGHT_FILTER_TOP_N]

    def strict_domain_filter(
        self,
        query: str,
        documents: List[Document],
        metadata_filter: Optional[Dict[str, str]],
    ) -> List[Document]:
        if not metadata_filter or metadata_filter.get("policy_domain") != "procurement":
            return documents

        strict = [doc for doc in documents if self.is_procurement_relevant(query, doc)]
        if strict:
            return strict
        return documents

    def is_procurement_relevant(self, query: str, doc: Document) -> bool:
        metadata = doc.metadata or {}
        file_name = str(metadata.get("file_name") or metadata.get("source") or "")
        section = str(metadata.get("section_title") or "")
        content = doc.page_content or ""
        combined = f"{file_name} {section} {content[:800]}"

        procurement_terms = ["采购", "请购", "供应商", "询价", "报价", "验收", "物资", "办公用品", "仓库"]
        approval_terms = ["审批", "批准", "审核", "总经理", "副总", "财务总监", "部门经理", "分管领导", "核准", "请购单"]
        amount_terms = ["金额", "万元", "5万", "五万", "50000", "5000", "超过", "以上", "固定资产", "低值易耗品"]
        polluted_file_terms = ["岗位职责", "销售管理", "合同管理", "劳动合同", "劳务合同", "组织架构"]

        file_has_procurement = any(term in file_name for term in ["采购", "请购", "仓库", "办公用品", "物资"])
        procurement_hits = sum(1 for term in procurement_terms if term in combined)
        approval_hits = sum(1 for term in approval_terms if term in combined)
        amount_hits = sum(1 for term in amount_terms if term in combined)

        if any(term in file_name for term in polluted_file_terms) and not file_has_procurement:
            return procurement_hits >= 2 and approval_hits >= 1
        if any(term in query for term in ["审批", "批准", "审核", "金额", "万元", "5万", "5000", "超过"]):
            return (
                file_has_procurement and approval_hits >= 1
                or procurement_hits >= 2 and approval_hits >= 1
                or procurement_hits >= 1 and approval_hits >= 1 and amount_hits >= 1
            )
        return file_has_procurement or procurement_hits >= 2

    def domain_aware_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        query_terms = set(self.tokenize(query))
        domain_terms = self.query_domain_terms(query)
        ranked = []
        for rank, doc in enumerate(documents, start=1):
            metadata = doc.metadata or {}
            source_text = " ".join(
                str(metadata.get(key, ""))
                for key in ["file_name", "source", "section_title", "policy_domain", "document_type"]
            )
            content_text = doc.page_content or ""
            combined = f"{source_text} {content_text[:600]}"
            domain_hit = sum(1 for term in domain_terms if term and term in combined)
            source_domain_hit = sum(1 for term in domain_terms if term and term in source_text)
            content_terms = set(self.tokenize(content_text[:800]))
            overlap = len(query_terms & content_terms)
            rrf_score = float(metadata.get("weighted_rrf_score", 0) or 0)
            procurement_priority = self.procurement_priority_score(query, source_text, combined)
            document_type_priority = self.document_type_priority_score(metadata)
            score = (
                1.0 / (rank + 1)
                + rrf_score
                + source_domain_hit * 0.18
                + domain_hit * 0.08
                + min(overlap, 8) * 0.015
                + procurement_priority
                + document_type_priority
            )
            ranked.append((score, rank, doc))
        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        result = []
        for new_rank, (score, _, doc) in enumerate(ranked, start=1):
            doc.metadata = {**doc.metadata, "domain_rerank_score": round(score, 6), "domain_rerank_rank": new_rank}
            result.append(doc)
        return result

    def document_type_priority_score(self, metadata: Dict[str, Any]) -> float:
        document_type = metadata.get("document_type")
        if document_type == "policy":
            return 0.18
        if document_type in {"contract_template", "job_description", "form_template"}:
            return -0.18
        return 0.0

    def procurement_priority_score(self, query: str, source_text: str, combined: str) -> float:
        if self.infer_query_domain(query) != "procurement":
            return 0.0

        score = 0.0
        if any(term in source_text for term in ["采购管理制度", "采购部工作流程", "采购管理流程"]):
            score += 0.45
        if "采购" in source_text:
            score += 0.2
        if any(term in combined for term in ["请购单", "采购申请", "采购核准权限"]):
            score += 0.25
        if any(term in query for term in ["审批", "批准", "审核", "金额", "万元", "超过"]):
            if any(term in combined for term in ["审批", "批准", "审核", "核准", "总经理", "副总", "财务总监", "分管领导"]):
                score += 0.35
        if any(term in source_text for term in ["岗位职责", "销售管理", "合同管理", "仓库管理"]):
            score -= 0.35
        return score

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []
        if settings.RERANK_PROVIDER == "cross_encoder":
            reranked = self.cross_encoder_rerank(query, documents)
            if reranked:
                return reranked
        if settings.RERANK_PROVIDER == "dashscope":
            reranked = self.dashscope_rerank(query, documents)
            if reranked:
                return reranked
        logger.warning("[Rerank] external reranker unavailable, using weighted RRF order")
        return documents[: settings.RERANK_TOP_K]

    def dashscope_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        try:
            import dashscope

            response = dashscope.TextReRank.call(
                model=settings.DASHSCOPE_RERANK_MODEL,
                query=query,
                documents=[doc.page_content for doc in documents],
                top_n=min(settings.RERANK_TOP_K, len(documents)),
                return_documents=False,
            )
            output = getattr(response, "output", None) or {}
            results = output.get("results") or []
            if not results:
                logger.warning("[Rerank] DashScope returned empty results")
                return []
            reranked = []
            for item in results:
                index = item["index"]
                doc = documents[index]
                doc.metadata = {**doc.metadata, "rerank_provider": "dashscope", "rerank_score": item.get("relevance_score")}
                reranked.append(doc)
            return reranked
        except Exception as exc:
            logger.warning("[Rerank] DashScope rerank failed: %s", exc)
            return []

    def cross_encoder_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(settings.CROSS_ENCODER_MODEL)
            pairs = [(query, doc.page_content) for doc in documents]
            scores = model.predict(pairs)
            ranked = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
            result = []
            for score, doc in ranked[: settings.RERANK_TOP_K]:
                doc.metadata = {**doc.metadata, "rerank_provider": "cross_encoder", "rerank_score": float(score)}
                result.append(doc)
            return result
        except Exception as exc:
            logger.warning("[Rerank] CrossEncoder rerank failed: %s", exc)
            return []

    def compress_context(self, query: str, documents: List[Document], intent: str = "professional_policy") -> List[Document]:
        query_terms = set(self.tokenize(query))
        priority_terms = {
            "材料", "审批", "流程", "要求", "风险", "注意", "证明", "标准", "条件", "申请",
            "发票", "合同", "报价", "验收", "归还", "保管", "记录", "上报", "处罚", "考核",
            "职责", "责任", "部门", "岗位", "控制", "控制点", "覆盖", "范围", "区别", "差异",
            "事件", "问题", "变更", "配置", "权限", "访问", "交接班", "台帐", "出库", "入库",
            "库存", "供应商", "评估", "审核", "确认", "处理建议", "根原因",
            "业务中断", "数据丢失", "敏感信息", "泄密", "备份", "保密", "访问控制", "风险处置",
            "安全目标", "安全方针", "管理体系", "控制措施",
        }
        chars_per_doc = self.context_chars_per_doc(intent)
        compressed = []
        for doc in documents:
            sentences = re.split(r"(?<=[。！？；\n])", doc.page_content)
            scored = []
            for index, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_terms = set(self.tokenize(sentence))
                score = 0
                score += 3 * len(query_terms & sentence_terms)
                score += 2 * sum(1 for term in priority_terms if term in sentence)
                if index <= 1:
                    score += 1
                if score > 0:
                    scored.append((score, index, sentence))
            scored.sort(key=lambda item: (-item[0], item[1]))
            selected = [sentence for _, _, sentence in scored[:6]]
            text = "".join(selected) if selected else doc.page_content
            text = text[:chars_per_doc]
            compressed.append(
                Document(
                    page_content=text,
                    metadata={**doc.metadata, "compressed": True, "rag_intent": intent},
                )
            )
        return compressed

    def context_chars_per_doc(self, intent: str) -> int:
        if intent == "direct_policy":
            return min(settings.COMPRESSED_CONTEXT_CHARS_PER_DOC, 350)
        if intent == "complex_process":
            return min(settings.COMPRESSED_CONTEXT_CHARS_PER_DOC, 650)
        return min(settings.COMPRESSED_CONTEXT_CHARS_PER_DOC, 500)

    def build_metadata_filter(self, query: str, intent: str) -> Optional[Dict[str, str]]:
        metadata_filter = {}
        if intent != "professional_policy":
            return None
        if self.has_composite_domain_signal(query):
            return None
        domain = self.infer_query_domain(query)
        if domain:
            metadata_filter["policy_domain"] = domain
        return metadata_filter or None

    def has_composite_domain_signal(self, query: str) -> bool:
        primary_groups = [
            ["培训", "学习", "企业文化", "团建"],
            ["环保", "整改", "环境"],
            ["权限", "账号", "信息安全", "数据安全", "保密"],
            ["招聘", "面试", "候选人", "岗位职责"],
            ["奖惩", "奖励", "违规", "纪律", "激励"],
            ["目标责任", "责任书", "工作计划", "工作汇报", "总结"],
        ]
        process_or_money_terms = ["报销", "费用", "申请", "材料", "审批", "流程", "验收", "上报", "处理", "共享", "外发"]
        return any(any(term in query for term in group) for group in primary_groups) and any(
            term in query for term in process_or_money_terms
        )

    def infer_query_domain(self, query: str) -> Optional[str]:
        domain_keywords = {
            "reimbursement": ["报销", "差旅", "费用", "发票"],
            "leave_attendance": ["请假", "事假", "考勤", "病假", "年假", "调休", "旷工", "迟到", "早退", "缺勤"],
            "procurement": ["采购", "供应商", "合同", "入库", "出库", "库存", "仓库", "物资", "验收"],
            "security": ["权限", "信息安全", "账号", "数据安全", "登录", "保密", "外发", "共享"],
            "onboarding": ["入职", "转正", "试用期", "离职", "人事", "录用"],
            "ticket_sop": ["工单", "客户", "SOP", "售后"],
        }
        for domain, keywords in domain_keywords.items():
            if any(keyword in query for keyword in keywords):
                return domain
        return None

    def query_domain_terms(self, query: str) -> List[str]:
        domain_terms = {
            "leave_attendance": ["请假", "事假", "休假", "考勤", "调休", "年假", "病假", "迟到", "早退", "旷工", "缺勤", "打卡"],
            "reimbursement": ["报销", "差旅", "费用", "发票", "票据", "凭证", "借款"],
            "procurement": ["采购", "供应商", "合同", "报价", "验收", "入库", "出库", "仓库", "库存", "物资", "领用", "盘点"],
            "security": [
                "权限", "账号", "信息安全", "数据安全", "保密", "临时权限", "系统", "登录", "异常登录",
                "访问控制", "备份", "数据丢失", "敏感信息", "泄密", "业务中断", "风险处置", "控制措施",
            ],
            "confidential": ["保密", "秘密", "文件", "资料", "共享", "外发", "外部", "对外"],
            "onboarding": ["入职", "转正", "离职", "试用期", "录用", "人事", "交接"],
            "recruitment": ["招聘", "面试", "候选人", "岗位", "职责", "录用"],
            "reward_discipline": ["奖惩", "奖励", "惩罚", "处罚", "违规", "纪律", "员工守则", "行为规范", "激励"],
            "work_report": ["工作计划", "工作汇报", "总结", "月报", "目标责任", "责任书", "指标"],
            "training": ["培训", "学习", "企业文化", "团队", "团建"],
            "environment": ["环保", "整改", "检查", "环境", "验收", "上报", "异常"],
            "sales": ["销售", "提成", "奖金", "业绩"],
        }
        terms = set(self.tokenize(query))
        selected: List[str] = []
        for candidates in domain_terms.values():
            if any(term in query or term in terms for term in candidates):
                selected.extend(candidates)
        selected.extend(term for term in terms if len(term) >= 2)
        return self.unique_list(selected)

    def filter_by_metadata(
        self,
        documents: List[Document],
        metadata_filter: Optional[Dict[str, str]],
    ) -> List[Document]:
        if not metadata_filter:
            return documents
        return [
            doc
            for doc in documents
            if self._metadata_match(doc, metadata_filter)
        ]

    def _metadata_match(self, doc: Document, metadata_filter: Dict[str, str]) -> bool:
        return all(doc.metadata.get(key) == value for key, value in metadata_filter.items())

    def bm25_score(
        self,
        query_terms: List[str],
        doc_terms: List[str],
        doc_freq: Dict[str, int],
        doc_count: int,
        avg_len: float,
    ) -> float:
        k1 = 1.5
        b = 0.75
        term_counts = Counter(doc_terms)
        doc_len = len(doc_terms)
        score = 0.0

        for term in query_terms:
            if term_counts[term] == 0:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            numerator = term_counts[term] * (k1 + 1)
            denominator = term_counts[term] + k1 * (1 - b + b * doc_len / max(avg_len, 1))
            score += idf * numerator / denominator
        return score

    def tokenize(self, text: str) -> List[str]:
        words = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())
        bigrams = [text[index : index + 2] for index in range(max(len(text) - 1, 0))]
        chinese_bigrams = [term for term in bigrams if re.match(r"^[\u4e00-\u9fff]{2}$", term)]
        return words + chinese_bigrams

    def document_key(self, doc: Document) -> Tuple[str, int]:
        return (doc.metadata.get("source", ""), int(doc.metadata.get("chunk_index", 0)))

    def unique_list(self, items: List[str]) -> List[str]:
        result = []
        for item in items:
            if item and item not in result:
                result.append(item)
        return result

    def extract_json_array(self, text: str) -> str:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no json array found")
        return text[start : end + 1]

    def format_context(self, documents: List[Document]) -> str:
        blocks = []
        total = 0
        max_context_chars = self.max_context_chars(self.intent_from_documents(documents))
        for index, doc in enumerate(documents, start=1):
            source = doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
            section = doc.metadata.get("section_title") or "未标注章节"
            policy_domain = doc.metadata.get("policy_domain") or "未标注制度域"
            text = doc.page_content.strip()
            block = f"[资料{index}] 来源：{source}\n章节：{section}\n制度域：{policy_domain}\n正文：{text}"
            total += len(block)
            if total > max_context_chars:
                break
            blocks.append(block)
        return "\n\n".join(blocks)

    def max_context_chars(self, intent: str) -> int:
        if intent == "direct_policy":
            return min(settings.MAX_CONTEXT_CHARS, 900)
        if intent == "complex_process":
            return min(settings.MAX_CONTEXT_CHARS, 2600)
        return min(settings.MAX_CONTEXT_CHARS, 1800)

    def sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        result = []
        for doc in documents:
            result.append(
                {
                    "source": doc.metadata.get("source"),
                    "file_name": doc.metadata.get("file_name"),
                    "document_type": doc.metadata.get("document_type"),
                    "policy_domain": doc.metadata.get("policy_domain"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "section_title": doc.metadata.get("section_title"),
                    "section_index": doc.metadata.get("section_index"),
                    "recall_route": doc.metadata.get("recall_route"),
                    "weighted_rrf_score": doc.metadata.get("weighted_rrf_score"),
                    "rerank_provider": doc.metadata.get("rerank_provider"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "compressed": doc.metadata.get("compressed"),
                    "preview": doc.page_content[:160],
                }
            )
        return result

    def rag_summarize(self, query: str) -> str:
        return self.answer(query).answer

    def intent_from_documents(self, documents: List[Document]) -> str:
        for doc in documents:
            intent = doc.metadata.get("rag_intent")
            if intent:
                return intent
        return "professional_policy"

    def answer_chain_for_intent(self, intent: str):
        return self.answer_chains.get(intent) or self.answer_chains["professional_policy"]

    def should_use_no_direct_basis_fallback(self, query: str, documents: List[Document]) -> bool:
        if not documents:
            return True
        context = " ".join(
            f"{doc.page_content} {(doc.metadata or {}).get('file_name', '')} {(doc.metadata or {}).get('section_title', '')}"
            for doc in documents[:5]
        )
        unsupported_terms = [
            term for term in UNSUPPORTED_SPECIFIC_TERMS
            if term in query and term not in context
        ]
        if unsupported_terms:
            return True
        for match in SPECIFIC_UNSUPPORTED_PATTERN.findall(query):
            compact = str(match).replace(" ", "")
            if compact and compact not in context.replace(" ", ""):
                return True
        return False

    def generate_answer(self, query: str, documents: List[Document], trace_id: str = None) -> str:
        intent = self.intent_from_documents(documents)
        if self.should_use_no_direct_basis_fallback(query, documents):
            log_trace(
                logger,
                "rag_answer_no_direct_basis",
                trace_id=trace_id,
                intent=intent,
                documents=len(documents),
                query=short_text(query),
            )
            return NO_DIRECT_BASIS_ANSWER
        start = now_ms()
        answer = self.answer_chain_for_intent(intent).invoke(
            {
                "question": query,
                "context": self.format_context(documents),
            }
        )
        log_trace(
            logger,
            "rag_answer_generated",
            trace_id=trace_id,
            intent=intent,
            documents=len(documents),
            answer_chars=len(answer or ""),
            elapsed_ms=elapsed_ms(start),
        )
        return answer

    def stream_answer(self, query: str, top_k: int = None, tenant_id: str = None, trace_id: str = None):
        total_start = now_ms()
        documents = self.retrieve_documents(query, top_k=top_k, tenant_id=tenant_id, trace_id=trace_id)
        if not documents:
            log_trace(logger, "rag_stream_done", trace_id=trace_id, documents=0, elapsed_ms=elapsed_ms(total_start))
            yield "知识库中没有明确依据。", []
            return

        sources = self.sources(documents)
        intent = self.intent_from_documents(documents)
        first_chunk = True
        for chunk in self.answer_chain_for_intent(intent).stream(
            {
                "question": query,
                "context": self.format_context(documents),
            }
        ):
            if chunk:
                if first_chunk:
                    log_trace(
                        logger,
                        "rag_stream_first_chunk",
                        trace_id=trace_id,
                        intent=intent,
                        documents=len(documents),
                        elapsed_ms=elapsed_ms(total_start),
                    )
                    first_chunk = False
                yield str(chunk), sources
        log_trace(
            logger,
            "rag_stream_done",
            trace_id=trace_id,
            intent=intent,
            documents=len(documents),
            sources_count=len(sources),
            elapsed_ms=elapsed_ms(total_start),
        )

    def answer(self, query: str, top_k: int = None, tenant_id: str = None, trace_id: str = None) -> RagResponse:
        total_start = now_ms()
        documents = self.retrieve_documents(query, top_k=top_k, tenant_id=tenant_id, trace_id=trace_id)
        if not documents:
            log_trace(logger, "rag_answer_done", trace_id=trace_id, documents=0, elapsed_ms=elapsed_ms(total_start))
            return RagResponse(answer="知识库中没有明确依据。", sources=[])

        answer = self.generate_answer(query, documents, trace_id=trace_id)
        sources = self.sources(documents)
        log_trace(
            logger,
            "rag_answer_done",
            trace_id=trace_id,
            documents=len(documents),
            sources_count=len(sources),
            answer_chars=len(answer or ""),
            elapsed_ms=elapsed_ms(total_start),
        )
        return RagResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    service = RagSummarizeService()
    print(service.rag_summarize("病假超过一天需要什么证明？"))
