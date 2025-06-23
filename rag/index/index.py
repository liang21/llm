from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("text.txt")
docs = loader.load()
print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits)
vectorstore = Chroma(
    collection_name="my_collection",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./db",
)

vectorstore.add_documents(splits)