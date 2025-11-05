import uuid
from fake_useragent import UserAgent
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from chromadb import PersistentClient
from config import CHROMA_DIR, COLLECTION_NAME
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

ua = UserAgent()
from chromadb import Client
client = Client()

def process_urls(urls):
    # Create collection fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)

    loader = WebBaseLoader(
        web_paths=urls,
    )
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(data)

    for doc in docs:
        collection.add(
            documents=[doc.page_content],
            metadatas=doc.metadata,
            ids=[str(uuid.uuid4())]
        )

    return len(docs)

def ask_question(query: str):
    # Instantiate the embedding model
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5" # A robust, compact model
    )
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
       
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        temperature=0.5,
        model_name="llama-3.3-70b-versatile"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain(query)

    sources = []
    for doc in result["source_documents"]:
        src = doc.metadata.get("source")
        if src and src not in sources:
            sources.append(src)

    return {
        "answer": result["result"],
        "sources": sources
    }
