from typing import List

import tiktoken
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


def str_token_count(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def tiktoken_counter(message:List[BaseMessage]) -> int:
    num_tokens = 3
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in message:
        if isinstance(msg,HumanMessage):
            role = "user"
        elif isinstance(msg,AIMessage):
            role = "assistant"
        elif isinstance(msg,ToolMessage):
            role = "tool"
        elif isinstance(msg,SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported massage type {msg.__class__}")
        num_tokens += tokens_per_message + str_token_count( role) + str_token_count(msg.content)
        if msg.name:
            num_tokens += tokens_per_name + str_token_count(msg.name)
        return num_tokens


model = ""
api_key = ""
base_url = ""
chat_model = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你现在扮演孔子的角色，尽量按照孔子的风格回复，不要出现‘子曰’",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
)


with_message_history = RunnableWithMessageHistory(
    trimmer | prompt | chat_model,
    get_session_history
)

config = {"configurable": {"session_id": "dreamhead"}}

while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break
    stream = with_message_history.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    for chunk in stream:
        print(chunk.content, end='', flush=True)
    print()