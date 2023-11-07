from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import streamlit as st
import os,csv
import PyPDF2
import textract  # You'll need to install the textract library
from langchain.chains import ConversationalRetrievalChain
import pandas as pd

# Use the API key
# OPENAI_API_KEY = os.getenv("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")

st.title("Q&A Chatbot")
st.write("Upload a PDF or text file and ask questions about its content.")

uploaded_file = st.file_uploader("Choose a file (PDF or text)", type=["pdf", "txt"])

if uploaded_file:
    # Initialize a variable to store the extracted text
    raw_text = ""

    if uploaded_file.type == "application/pdf":
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    elif uploaded_file.type == "text/plain":
        # Save the uploaded text file to a temporary location
        with open("temp.txt", "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Read the text file using textract
        raw_text = textract.process("temp.txt", encoding='utf-8').decode('utf-8')
    


    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Check the content of the texts list
    st.write("Number of text chunks:", len(texts))

    # Download embeddings
    embeddings = OpenAIEmbeddings(openai_api_key="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")

    # Create the document search
    docsearch = FAISS.from_texts(texts, embeddings)

    # Create the conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0, openai_api_key="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"),
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    # Initialize chat history list
    chat_history = []

    # Get the user's query
    query = st.text_input("Ask a question about the uploaded document:")

    # Add a generate button
    generate_button = st.button("Generate Answer")

    if generate_button and query:
        with st.spinner("Generating answer..."):
            result = qa({"question": query, "chat_history": chat_history})

            answer = result["answer"]
            source_documents = result['source_documents']

            # Combine the answer and source_documents into a single response
            response = {
                "answer": answer,
                "source_documents": source_documents
            }
            st.write("Response:", response)
