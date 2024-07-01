import os
import uuid
import shutil
import time
import pandas as pd
import openai
import streamlit as st
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set up OpenAI API key
openai.api_key = st.secrets['openai_key']

def process_pdf(pdf_path: str) -> dict:
    # Generate a unique persistence directory for each PDF
    persist_directory = f"chroma_db_{uuid.uuid4()}"
    try:
        # Ingest PDF documents
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("No content extracted from the PDF")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        
        # Create a new Chroma instance with a unique persistence directory
        vectordb = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)

        # Set up retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Set up ChatOpenAI model
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        def query_documents(query: str) -> List[str]:
            result = qa_chain({"query": query})
            return result["result"], [doc.page_content for doc in result["source_documents"]]

        question = "Who is the witness, what committee is this hearing for (format exactly as either 'Energy and Commerce', 'Ways and Means', or 'Budget'), the date, and summarize the witness's statement? Use gender neutral language in referring to the witness. Return each answer separated by | with strictly only the answer. Do not rephrase the question. An example output is 'John Doe | Energy and Commerce | 2024-03-01 | John Doe talked about stuff'"
        answer, sources = query_documents(question)
        
        # Clean up: delete the Chroma collection and persist changes
        vectordb.delete_collection()
        vectordb.persist()

        return {
            "pdf_path": pdf_path,
            "answer": answer,
            "persist_directory": persist_directory,
            "success": True
        }
    except Exception as e:
        return {
            "pdf_path": pdf_path,
            "answer": f"ERROR: Unable to process PDF: {str(e)}",
            "persist_directory": persist_directory,
            "success": False
        }

def cleanup_directories(directories):
    for directory in directories:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(directory)
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    time.sleep(1)
                else:
                    pass

def main():
    st.title("PDF Testimony Summarizer")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner('Processing PDF...'):
            # Save the uploaded file temporarily
            temp_file_path = os.path.join("/tmp", f"{uuid.uuid4()}.pdf")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Process the PDF
            result = process_pdf(temp_file_path)

            if result["success"]:
                answers = result["answer"].split("|")
                if len(answers) == 4:
                    st.write("**Witness:**", answers[0].strip())
                    st.write("**Committee:**", answers[1].strip())
                    st.write("**Date:**", answers[2].strip())
                    st.write("**Summary:**", answers[3].strip())
                else:
                    st.error("Unexpected answer format. Unable to display the results.")
            else:
                st.error(result["answer"])

            # Cleanup temporary file and directories
            os.remove(temp_file_path)
            cleanup_directories([result['persist_directory']])
    
if __name__ == "__main__":
    main()
