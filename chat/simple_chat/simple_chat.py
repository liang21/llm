from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


model = ""
api_key = ""
base_url = ""
chat_model = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)
while True:
    user_input = input("You:>")
    if user_input.lower() == "exit":
        break
    stream = chat_model.stream([HumanMessage(content=user_input)])
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    print()
