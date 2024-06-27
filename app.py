import streamlit as st
from PyPDF2 import PdfReader

import faiss

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

#used for embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import google.generativeai as genai

#vector store FAISS
from langchain_community.vectorstores  import FAISS

#
from langchain_google_genai import ChatGoogleGenerativeAI

#do the chat 
from langchain.chains.question_answering import load_qa_chain


#do promt temlating
from langchain.prompts import PromptTemplate

#dot env environment var
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#to load pdf files from left side , to read pdf text 

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf) #
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

#text into chunks

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


#chunks to vector
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index") #saving vector locally in faiss_index folder

def get_conversational_chain():
    prompt_template="""
    answer the question in a detailed form  from the provided context,
    ,make sure to provide context just say , "answer is not available in the context", don't provide wrong answer
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer: 
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True) 

    docs=new_db.similarity_search(user_question)

    chain=get_conversational_chain()

    response=chain(
        {"input_documents":docs,"question":user_question}
        ,return_only_outputs=True)
    print(response)
    st.write("Reply: ",response["output_text"])


def main():
    st.set_page_config("chat with pdfs")
    st.header("chat with pdfs using google gemni")
    
    user_question=st.text_input("Enter your query")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("upload your pdf file",accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("processing....."):
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__=="__main__":
    main()
   