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


ANSWER_PROMPT = """你是企业知识智能体，请只基于给定制度资料回答问题。

要求：
1. 回答要直接、结构清晰。
2. 如果资料不足，明确说明“知识库中没有明确依据”。
3. 涉及流程时，按步骤输出。
4. 涉及制度依据时，标注来源文件。

用户问题：{question}

制度资料：
{context}
"""

DIRECT_POLICY_PROMPT = """你是企业制度问答助手。请只基于给定资料，用短答案回答。

输出要求：
1. 结论：1-2 句话，必须直接回应问题。
2. 依据：1 句话，写来源文件名。
3. 如果没有完全匹配依据，但有相近制度依据，请给“审慎结论”，并说明需要按审批/报销/材料要求人工确认。
4. 只有资料完全无关时，才回答“知识库中没有明确依据”。

用户问题：{question}

制度资料：
{context}
"""

POLICY_ANSWER_PROMPT = """你是企业制度问答助手。请只基于给定资料，短结构化回答。

输出格式，每项 1-2 条，禁止长篇展开：
1. 结论：直接回答。
2. 材料/条件：列关键材料或适用条件；没有则写“资料未明确”。
3. 审批/流程：列关键审批节点或流程；没有则写“资料未明确”。
4. 注意事项：列关键风险、时限或禁止事项；没有则写“资料未明确”。
5. 依据：来源文件名。

要求：
- 不要编造资料中没有的信息。
- 必须尽量使用“材料、审批、流程、要求、风险、依据”等明确表述。
- 总字数控制在 300 字以内。

用户问题：{question}

制度资料：
{context}
"""

PROCESS_ANSWER_PROMPT = """你是企业流程智能体。请只基于给定资料，生成简洁可执行的流程方案或申请草稿。

输出格式：
1. 事项概述：1 句话。
2. 办理步骤：最多 4 步。
3. 所需材料：最多 5 项；资料未说明则写“资料未明确”。
4. 审批路径：最多 4 个节点；资料未说明则写“资料未明确”。
5. 风险提示：最多 3 条。
6. 申请草稿：100 字以内。
7. 依据：来源文件名。

要求：
- 不要编造资料中没有的硬性规则。
- 对不确定内容明确提示需要人工确认。
- 总字数控制在 500 字以内。

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

    def retrieve_documents(self, query: str, top_k: int = None, tenant_id: str = None) -> List[Document]:
        self.ensure_vector_store_ready()
        tenant_id = tenant_id or "global"
        intent = self.classify_intent(query)
        metadata_filter = self.build_metadata_filter(query, intent)
        recalled_lists: List[Tuple[str, List[Document]]] = []

        if intent == "direct_policy":
            target_top_k = top_k or self.answer_top_k(intent)
            vector_k = max(settings.VECTOR_RECALL_TOP_K, target_top_k * 2)
            selected = self.vector_recall(query.strip(), vector_k, metadata_filter=None)
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
            recalled_lists.append(("vector", self.vector_recall(rewritten, vector_k, metadata_filter)))
            recalled_lists.append(("bm25", self.bm25_recall(rewritten, bm25_k, metadata_filter, tenant_id=tenant_id)))
            if metadata_filter:
                fallback_k = max(target_top_k, 3)
                recalled_lists.append(("vector_fallback", self.vector_recall(rewritten, fallback_k, metadata_filter=None)))
                recalled_lists.append(("bm25_fallback", self.bm25_recall(rewritten, fallback_k, metadata_filter=None, tenant_id=tenant_id)))

        fused = self.weighted_reciprocal_rank_fusion(recalled_lists)
        filtered = self.light_filter(query, fused)
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
    ) -> List[Document]:
        docs = self.vector_store.similarity_search(query, k=k, metadata_filter=metadata_filter)
        for rank, doc in enumerate(docs, start=1):
            doc.metadata = {**doc.metadata, "recall_route": "vector", "vector_rank": rank}
        return docs

    def bm25_recall(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
        tenant_id: str = "default",
    ) -> List[Document]:
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
            return []

        query_terms = self.tokenize(query)
        if not query_terms:
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
                for key in ["file_name", "source", "section_title", "policy_domain"]
            )
            content_text = doc.page_content or ""
            combined = f"{source_text} {content_text[:600]}"
            domain_hit = sum(1 for term in domain_terms if term and term in combined)
            source_domain_hit = sum(1 for term in domain_terms if term and term in source_text)
            content_terms = set(self.tokenize(content_text[:800]))
            overlap = len(query_terms & content_terms)
            rrf_score = float(metadata.get("weighted_rrf_score", 0) or 0)
            score = (
                1.0 / (rank + 1)
                + rrf_score
                + source_domain_hit * 0.18
                + domain_hit * 0.08
                + min(overlap, 8) * 0.015
            )
            ranked.append((score, rank, doc))
        ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        result = []
        for new_rank, (score, _, doc) in enumerate(ranked, start=1):
            doc.metadata = {**doc.metadata, "domain_rerank_score": round(score, 6), "domain_rerank_rank": new_rank}
            result.append(doc)
        return result

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
            "leave_attendance": ["请假", "考勤", "病假", "年假", "调休", "旷工", "迟到", "早退", "缺勤"],
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
            "leave_attendance": ["请假", "休假", "考勤", "调休", "年假", "病假", "迟到", "早退", "旷工", "缺勤", "打卡"],
            "reimbursement": ["报销", "差旅", "费用", "发票", "票据", "凭证", "借款"],
            "procurement": ["采购", "供应商", "合同", "报价", "验收", "入库", "出库", "仓库", "库存", "物资", "领用", "盘点"],
            "security": ["权限", "账号", "信息安全", "数据安全", "保密", "临时权限", "系统", "登录", "异常登录"],
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
            text = doc.page_content.strip()
            block = f"[资料{index}] 来源：{source}\n{text}"
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

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        intent = self.intent_from_documents(documents)
        return self.answer_chain_for_intent(intent).invoke(
            {
                "question": query,
                "context": self.format_context(documents),
            }
        )

    def stream_answer(self, query: str, top_k: int = None, tenant_id: str = None):
        documents = self.retrieve_documents(query, top_k=top_k, tenant_id=tenant_id)
        if not documents:
            yield "知识库中没有明确依据。", []
            return

        sources = self.sources(documents)
        intent = self.intent_from_documents(documents)
        for chunk in self.answer_chain_for_intent(intent).stream(
            {
                "question": query,
                "context": self.format_context(documents),
            }
        ):
            if chunk:
                yield str(chunk), sources

    def answer(self, query: str, top_k: int = None, tenant_id: str = None) -> RagResponse:
        documents = self.retrieve_documents(query, top_k=top_k, tenant_id=tenant_id)
        if not documents:
            return RagResponse(answer="知识库中没有明确依据。", sources=[])

        answer = self.generate_answer(query, documents)
        return RagResponse(answer=answer, sources=self.sources(documents))


if __name__ == "__main__":
    service = RagSummarizeService()
    print(service.rag_summarize("病假超过一天需要什么证明？"))
