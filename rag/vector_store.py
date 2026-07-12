import os.path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from AIRAGAgent.model.factory import embed_model
from AIRAGAgent.utils.config_handler import chroma_config
from AIRAGAgent.utils.file_handler import txt_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
from AIRAGAgent.utils.logger_handler import logger
from AIRAGAgent.utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store=Chroma(
            collection_name=chroma_config["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_config["persist_directory"]),
        )
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"],
            length_function=len,
        )
    def get_retriver(self):
        return self.vector_store.as_retriever(search_kwargs={"k":chroma_config["k"]})

    def count_documents(self) -> int:
        return self.vector_store._collection.count()

    def load_document(self, force: bool = False):
        """
        从数据文件夹内读取文件，转为向量存入向量库
        要计算文件的MD5做去重
        :return:
        """

        def check_md5_hex(md5_for_check:str):
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                #创建文件
                open(get_abs_path(chroma_config["md5_hex_store"]), "w",encoding="utf-8").close()
                return False #md5没处理过
            with open(get_abs_path(chroma_config["md5_hex_store"]),"r",encoding="utf-8") as f:
                for line in f.readlines():
                    line=line.strip()
                    if line==md5_for_check:  #md5处理过
                        return True
                return False

        def save_md5_hex(md5_for_check:str):
            with open(get_abs_path(chroma_config["md5_hex_store"]),"a",encoding="utf-8") as f:
                f.write(md5_for_check+"\n")

        #将文件内容变成langchain里面的document对象

        def get_file_documents(read_path:str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path:list[str]=listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"]),
        )

        for path  in allowed_files_path:
            md5_hex=get_file_md5_hex(path)
            if not force and check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents:list[Document]=get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue
                split_document:list[Document]=self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue
                #将内容存入向量库
                self.vector_store.add_documents(split_document)

                #记录已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as e:
                #exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}",exc_info=True)
                continue


if __name__=="__main__":
    vs=VectorStoreService()
    vs.load_document()
    retriever=vs.get_retriver()
    res=retriever.invoke("入职和转正流程")
    for r in res:
        print(r.page_content)
        print("-"*20)

