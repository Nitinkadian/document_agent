import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from utils import connect_weaviate, get_embeddings, process_pdf, ingest_to_weaviate


# Load .env
load_dotenv()

# Streamlit UI
st.title("🤖 AI Agent with Weaviate + LangChain")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
doc_domain = st.selectbox("Select document domain", ["general", "invoice", "insurance", "finance"])
target_language = st.selectbox(
    "Select translation language (for Translator tool)",
    ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Hindi"]
)

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    # --- Connect + embeddings ---
    client = connect_weaviate()
    embeddings = get_embeddings()

    # --- Process PDF ---
    raw_text, chunks = process_pdf(uploaded_file)
    vectorstore = ingest_to_weaviate(client, embeddings, chunks, doc_domain=doc_domain)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # --- LLM ---
    llm = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            task="conversational",  # important for chat models
            temperature=0.3,
            max_new_tokens=512,
        )
    )

    # --- Prompts ---
    extractor_prompts = {
        "general": "Extract key entities (like names, dates, amounts, IDs, etc.) from the following text:\n\n{input}",
        "invoice": "Extract invoice-specific fields: Invoice Number, Date, Due Date, Vendor, Customer, Line Items, Total Amount, Taxes, Payment Terms from the following text:\n\n{input}",
        "insurance": "Extract insurance-specific fields: Policy Number, Claim Number, Claimant Name, Date of Loss, Premium Amount, Coverage Type, Beneficiary, Contact Info from the following text:\n\n{input}",
        "finance": "Extract finance-specific fields: Transaction ID, Account Number, Balance, Transaction Date, Currency, Amount, Sender, Receiver, Description from the following text:\n\n{input}",
    }

    summarizer_prompt = PromptTemplate.from_template("Summarize the following text:\n\n{input}")
    translator_prompt = PromptTemplate.from_template(
        f"Translate the following text into {target_language}:\n\n{{input}}"
    )
    extractor_prompt = PromptTemplate.from_template(extractor_prompts[doc_domain])

    # --- Chains ---
    summarizer_chain = LLMChain(llm=llm, prompt=summarizer_prompt)
    translator_chain = LLMChain(llm=llm, prompt=translator_prompt)
    extractor_chain = LLMChain(llm=llm, prompt=extractor_prompt)
    rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

    # --- Tools ---
    tools = [
        Tool(
            name="RAG Retriever",
            func=lambda q: rag_chain.invoke({"question": q})["answer"],
            description="Answer user queries from the uploaded document using retrieval."
        ),
        Tool(
            name="Summarizer",
            func=lambda text: summarizer_chain.invoke({"input": text})["text"],
            description="Summarize the uploaded document or a given text."
        ),
        Tool(
            name="Translator",
            func=lambda text: translator_chain.invoke({"input": text})["text"],
            description=f"Translate the document or text into {target_language}."
        ),
        Tool(
            name="Extractor",
            func=lambda text: extractor_chain.invoke({"input": text})["text"],
            description=f"Extract structured fields for {doc_domain} documents."
        )
    ]

    # --- Agent ---
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-conversational-react-description",  # better suited for chat models
        verbose=True,
        memory=memory
    )

    # --- Chat UI ---
    user_input = st.chat_input("💬 Ask me anything: summarize, translate, extract, or query the doc")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🤔 Thinking..."):
            response = agent.run(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)