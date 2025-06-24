from mem0 import Memory

DEFAULT_MODEL = "deepseek-r1:14b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

config = {
    "version":"v1.1",
    "llm":{
        "provider":"openai",
        "config":{
            "model": DEFAULT_MODEL,
            "temperature":0,
            "max_tokens": 1024
        }
    },
    "embedding":{
        "provider":"openai",
        "config":{
            "model": EMBEDDING_MODEL,
        }
    },
    "vector_store":{
        "provider":"chroma",
        "config":{
            "collection_name":"mem0db",
            "path":"mem0db"
        }
    },
    "history_db_path":"history.db"

}
m = Memory.from_config(config)

m.add("我喜欢读书", user_id="dreamhead", metadata={"category": "hobbies"})
m.add("我喜欢编程", user_id="dreamhead", metadata={"category": "hobbies"})

related_memories = m.search(query="dreamhead有哪些爱好？", user_id="dreamhead")
print(' '.join([mem["memory"] for mem in related_memories['results']]))