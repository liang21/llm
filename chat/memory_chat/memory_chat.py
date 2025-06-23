from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

model = ""
api_key = ""
base_url = ""
chat_model = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
store = dict()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chat_model,
    get_session_history,
)
config = {"configurable": {"session_id": "dreamhead"}}
while True:
    user_input = input("You:>")
    if user_input.lower() == "exit":
        break
    stream = with_message_history.stream([HumanMessage(content=user_input)], config=config)
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    print()
