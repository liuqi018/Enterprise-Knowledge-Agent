"""
yaml
k:v
"""
import yaml
from AIRAGAgent.utils.path_tool import get_abs_path

def load_rag_config(config_path:str=get_abs_path("config/rag.yml"),encoding:str="utf-8") -> dict:
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_chroma_config(config_path:str=get_abs_path("config/chroma.yml"),encoding:str="utf-8") -> dict:
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)


def load_prompts_config(config_path:str=get_abs_path("config/prompts.yml"),encoding:str="utf-8") -> dict:
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_agent_config(config_path:str=get_abs_path("config/agent.yml"),encoding:str="utf-8") -> dict:
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.load(f,Loader=yaml.FullLoader)

def load_access_control_config(config_path: str = get_abs_path("config/access_control.yml"), encoding: str = "utf-8") -> dict:
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}

rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()


if __name__ == "__main__":
    print(rag_config)
    print(chroma_config)
    print(prompts_config)
