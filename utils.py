import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure


# -------------------------
# Connect to Weaviate
# -------------------------
def connect_weaviate():
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )
    return client


# -------------------------
# Embeddings model
# -------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -------------------------
# Process PDF into chunks
# -------------------------
def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)
    return raw_text, chunks


# -------------------------
# Ingest into Weaviate and return LangChain VectorStore
# -------------------------
'''def ingest_to_weaviate(client, embeddings, chunks, doc_domain="general"):
    # If collection exists, delete it (fresh doc ingestion)
    if client.collections.exists("Document"):
        client.collections.delete("Document")

    # Create collection
    client.collections.create(
        name="Document",
        description="Stores document chunks",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="domain", data_type=DataType.TEXT),
        ],
    )

    # Insert chunks with embeddings
    with client.batch.dynamic() as batch:
        for chunk in chunks:
            embedding = embeddings.embed_query(chunk)
            batch.add_object(
                collection="Document",
                properties={
                    "content": chunk,
                    "domain": doc_domain,
                },
                vector=embedding,
            )

    # Wrap the Weaviate collection into a LangChain VectorStore
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name="Document",
        text_key="content",
        embedding=embeddings,
    )

    return vectorstore'''


def ingest_to_weaviate(client, embeddings, chunks, doc_domain="general"):
    collection_name = "Document"

    # If collection exists, delete it
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    # Create collection
    client.collections.create(
        name=collection_name,
        description="Stores document chunks",
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="domain", data_type=DataType.TEXT),
        ],
    )

    collection = client.collections.get(collection_name)

    # Insert chunks with embeddings
    for chunk in chunks:
        embedding = embeddings.embed_query(chunk)
        collection.data.insert(
            properties={
                "content": chunk,
                "domain": doc_domain,
            },
            vector=embedding,
        )

    # Use new LangChain integration
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name=collection_name,
        text_key="content",
        embedding=embeddings,
    )

    return vectorstore