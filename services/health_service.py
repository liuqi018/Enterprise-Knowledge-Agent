def health_check():
    """
    系统健康检查（企业项目常见）
    后面可以扩展：
    - Redis状态
    - DB状态
    - LLM状态
    """
    return {
        "status": "ok",
        "service": "ai-agent-platform",
        "version": "1.0.0"
    }