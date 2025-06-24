import time

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI


DEFAULT_MODEL = "deepseek-r1:14b"
api_key = "EMPTY"
base_url = "http://127.0.0.1:11434/v1"
set_llm_cache(InMemoryCache())
model = ChatOpenAI(model=DEFAULT_MODEL, api_key=api_key, base_url=base_url)
start_time = time.time()
response = model.invoke("给我讲一句笑话")
end_time = time.time()
print(response.content)
print(f"第一次调用耗时: {end_time - start_time}秒")

start_time = time.time()
response = model.invoke("给我讲个一句话笑话")
end_time = time.time()
print(response.content)
print(f"第二次调用耗时: {end_time - start_time}秒")