import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from AIRAGAgent.config.settings import settings
from AIRAGAgent.knowledge.cleaner import is_low_value_chunk


TITLE_PATTERN = re.compile(
    r"^\s*("
    r"第[一二三四五六七八九十百\d]+[章节条款部分]|"
    r"[一二三四五六七八九十]+[、.)）]|"
    r"\d+(?:\.\d+)*[、.)）]|"
    r"[（(][一二三四五六七八九十\d]+[）)]"
    r")\s*.+"
)

TITLE_SUFFIXES = ("制度", "办法", "流程", "规定", "职责", "标准", "方案", "细则", "规范", "指引")


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """Split documents by section first, then split long sections with overlap."""
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    chunks: List[Document] = []

    for document in documents:
        sections = split_into_sections(document.page_content)
        for section_index, (section_title, section_text) in enumerate(sections):
            section_chunks = split_section(section_text, chunk_size, chunk_overlap)
            for local_index, chunk_text in enumerate(section_chunks):
                metadata = {
                    **document.metadata,
                    "section_title": section_title,
                    "section_index": section_index,
                    "section_chunk_index": local_index,
                }
                if is_low_value_chunk(chunk_text, metadata):
                    continue
                chunks.append(Document(page_content=chunk_text, metadata=metadata))

    for index, chunk in enumerate(chunks):
        chunk.metadata = {**chunk.metadata, "chunk_index": index}
    return chunks


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    lines = [line.strip() for line in text.splitlines()]
    sections: List[Tuple[str, List[str]]] = []
    current_title = "正文"
    current_lines: List[str] = []

    for line in lines:
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if is_section_title(line) and current_lines:
            sections.append((current_title, current_lines))
            current_title = line[:80]
            current_lines = [line]
            continue

        if is_section_title(line) and not current_lines:
            current_title = line[:80]
        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    result = []
    for title, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if section_text:
            result.append((title, section_text))
    return result or [("正文", text.strip())]


def is_section_title(line: str) -> bool:
    if len(line) > 80:
        return False
    if TITLE_PATTERN.match(line):
        return True
    if line.endswith(TITLE_SUFFIXES) and len(line) <= 40:
        return True
    return False


def split_section(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", ".", "!", "?", " ", ""],
        length_function=len,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]
