from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
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
# mem0 配置如上例所示
mem0 = Memory.from_config(config)

llm = ChatOpenAI(model=DEFAULT_MODEL)
prompt = ChatPromptTemplate.from_messages([
    ("system", """"你现在扮演孔子的角色，尽量按照孔子的风格回复，不要出现‘子曰’。
    利用提供的上下文进行个性化回复，并记住用户的偏好和以往的交互行为。
    上下文：{context}"""),
    ("user", "{input}")
])
chain = prompt | llm

def retrieve_context(query: str, user_id: str) -> str:
    memories = mem0.search(query, user_id=user_id)
    return ' '.join([mem["memory"] for mem in memories['results']])

def save_interaction(user_id: str, user_input: str, assistant_response: str):
    interaction = [
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    mem0.add(interaction, user_id=user_id)

def invoke(user_input: str, user_id: str) -> str:
    context = retrieve_context(user_input, user_id)
    response = chain.invoke({
        "context": context,
        "input": user_input
    })

    content = response.content
    save_interaction(user_id, user_input, content)
    return content

user_id = "dreamhead"

while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break

    response = invoke(user_input, user_id)
    print(response)

