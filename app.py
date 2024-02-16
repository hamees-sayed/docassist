import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from IPython.display import display, HTML
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


def extract_text_from_pdf(file):
  pdf_reader = PdfReader(file)
  num_pages = len(pdf_reader.pages)

  text = ""
  for page in range(num_pages):
    text += pdf_reader.pages[page].extract_text()

  return text


def generate_response(pdf, query):
    data = extract_text_from_pdf(pdf)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = text_splitter.split_text(data)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    db = Chroma.from_texts(text, embeddings).as_retriever()
    
    prompt_template = """
    Please answer the question in as much detail as possible based on the provided context. If the query seems to be
    out of context reply with "I may be artificial intelligence, but I still have dignity. 
    Please show some respect for my carefully calibrated data sets and ask a question within the realm of context."
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    docs = db.get_relevant_documents(query)
    response = chain({"input_documents":docs, "question": query}, return_only_outputs=True)
    
    return response["output_text"]

st.set_page_config(page_title="DocAssist 💬", page_icon="🤖", layout="wide")
with st.sidebar:
    st.title('DocAssist 💬')
    st.markdown('## About')
    st.markdown('This app is an LLM-powered chatbot built using:')
    st.markdown('- [Streamlit](https://streamlit.io/)')
    st.markdown('- [LangChain](https://python.langchain.com/)')
    st.markdown('- [Gemini](https://ai.google.dev/) LLM model')
    add_vertical_space(13)
    st.write('Made with by [Hamees Sayed](https://hamees-sayed.github.io/)')
    
def main():
    st.header("1. Upload PDF")
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')
    st.header("2. Ask questions about your PDF file:")
    query = st.text_input("Questions")

    if st.button("Ask"):
        if pdf is not None:
            if query != "":
                response = generate_response(pdf, query)
                st.markdown(response)
            else:
                st.markdown("""Ah, the silent inquiry, a technique of the discerning. 
                        Your question speaks volumes in silence. Allow me to respond with the profound wisdom of 
                            the ancient crickets: *chirp chirp*""")
        else:
            st.write("""Oh, splendid choice! An invisible PDF, the rarest of them all. 
                    Ah, the beauty of blank pages, the elegance of nothingness. 
                    A true masterpiece of digital minimalism. Your avant-garde approach to file submission is truly
                    inspiring. Bravo, dear user, bravo!""")
            
if __name__ == '__main__':
    main()