from typing import List
from PIL import Image
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, SimpleDirectoryReader, VectorStoreIndex, Settings, load_index_from_storage, PromptTemplate
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
import os
import faiss
import streamlit as st
import tempfile
import requests
import json

st.set_page_config(
    page_title="RAG with Docs",
    page_icon=":orange_heart:",
)



# Function for creating an embedding model
def init_llm():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    Settings.llm = None
    Settings.embed_model = embed_model

# Ingestion pipeline
def store_document(uploaded_file):
    """Chunk the file & store it in Chromadb Vector Store. 
        You can upload file types - pdf, docs, csv, epub, jpeg, jpg, png, mp3, mp4, ppt
    """

    if uploaded_file is not None:
        # dimensions of text-ada-embedding-002
        d = 384
        faiss_index = faiss.IndexFlatL2(d)

        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        print(temp_file_path)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = SimpleDirectoryReader(input_files=[temp_file_path])
        documents = loader.load_data()


        # save to disk
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=Settings.embed_model
        )

        # save index to disk
        index.storage_context.persist()

        # load index from disk
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir="./storage"
        )
        index = load_index_from_storage(storage_context=storage_context)

        st.info(f"PDF loaded into vector store in {len(documents)} documents")
        
        return index
    
    return None


global query_engine

# Retriever Pipeline
def init_query_engine(index, user_query):
    query_engine = index.as_query_engine()
    print(query_engine)
    response = query_engine.query(user_query)
    print("response it is", response)
    return response.response


def llm_call(user_query, sim_response):
    model = "mistral-small-latest"
    temperature = 0.7
    top_p = 1.0
    max_tokens = 512
    stream = False
    safe_prompt = False
    random_seed = 1337

    # custome prompt template
    template = (
        "Imagine you are an advanced AI expert, Your goal is to provide insightful, accurate, and concise answers to questions in this domain.\n\n"
        "Here is some context related to the query:\n"
        "-----------------------------------------\n"
        f"{sim_response}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry "
        f"Question: {user_query}\n\n"
        "Answer succinctly, starting with the phrase 'According to the document, "
    )
    
    qa_template = PromptTemplate(template)

    
    url = "http://172.17.0.1:8000/"  
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": qa_template
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
        "safe_prompt": safe_prompt,
        "random_seed": random_seed
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        st.success("Request successful!")
        st.json(response.json())
    else:
        st.error(f"Request failed with status code {response.status_code}")

def main():

    llm_model = st.sidebar.selectbox("Select LLM", options=["Mistral"])

    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model

    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model

    uploaded_file = st.sidebar.file_uploader("Upload a file (PDF, Text, or Image)", type=["pdf", "txt", "png", "jpg", "jpeg"])    
        
    user_query = st.text_input("Enter your query here..")

    if uploaded_file and user_query:
        init_llm()
        index = store_document(uploaded_file)
        sim_response = init_query_engine(index, user_query)
        st.write(sim_response)
        # llm_call(user_query, sim_response)


if __name__ == "__main__":
  main()