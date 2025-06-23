from operator import itemgetter
from typing import List

import tiktoken
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage, trim_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = TextLoader("text.txt")
docs = loader.load()
print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits)

model = ""
api_key = ""
base_url = ""

vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="vectordb"
)

retriever = vectorstore.as_retriever(search_type="similarity")

def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))

def tiktoken_counter(messages: List[BaseMessage]) -> int:
    num_tokens = 3
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
                tokens_per_message
                + str_token_counter(role)
                + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens

trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chat_model = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Context: {context}""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

context = itemgetter("question") | retriever | format_docs
first_step = RunnablePassthrough.assign(context=context)
chain = first_step | prompt | trimmer | chat_model

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "dreamhead"}}

while True:
    user_input = input("You:> ")
    if user_input.lower() == 'exit':
        break

    if user_input.strip() == "":
        continue

    stream = with_message_history.stream(
        {"question": user_input},
        config=config
    )
    for chunk in stream:
        print(chunk.content, end='', flush=True)