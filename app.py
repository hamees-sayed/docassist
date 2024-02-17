import os
import re
import nltk
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from multiprocessing import Pool
from nltk.corpus import stopwords
import google.generativeai as genai
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
download_nltk_resources()

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


st.set_page_config(page_title="DocAssist 💬", page_icon="🤖", layout="wide")
with st.sidebar:
    st.title('DocAssist 💬')
    st.header("1. Upload PDF")
    pdf = st.file_uploader("**Upload your PDF**", type='pdf')
    add_vertical_space(1)
    st.markdown('## About')
    st.markdown('This app is an LLM-powered chatbot built using:')
    st.markdown('- [Streamlit](https://streamlit.io/)')
    st.markdown('- [LangChain](https://python.langchain.com/)')
    st.markdown('- [Gemini](https://ai.google.dev/)')
    st.write('Made with by [Hamees Sayed](https://hamees-sayed.github.io/), [Github](https://github.com/hamees-sayed/docassist)')


# Regular Expressions
non_alphanumeric_re = re.compile(r'[^a-zA-Z0-9\s]')
whitespace_re = re.compile(r'\s+')

# NLTK Resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = non_alphanumeric_re.sub('', text)
    text = whitespace_re.sub(' ', text)
    
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def extract_and_clean(file):
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)

    text = ""
    for page in range(num_pages):
        text += pdf_reader.pages[page].extract_text()
        
    cleaned_text = clean_text(text)
    return cleaned_text

def extract_text_from_pdf(file):
    with Pool() as pool:
        texts = pool.map(extract_and_clean, [file])
    return '\n'.join(texts)


def generate_response(pdf, query):
    text = extract_text_from_pdf(pdf)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # save to disk
    db = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    db.persist()
    
    prompt_template = """
    Please answer the question in as much detail as possible based on the provided context and keep it simple so that even
    a beginner can understand.
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    # load from disk
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = db.similarity_search(query)
    response = chain({"input_documents":docs, "question": query}, return_only_outputs=True)["output_text"]
    
    return response

    
def main():
    query = st.chat_input("What is up?")
    if "messages" not in st.session_state:
                st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if query:
        if pdf is not None:
            st.chat_message("user").write(query)
            st.session_state.messages.append({"role": "user", "content": query})
            
            response = generate_response(pdf, query)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            st.write("""Oh, splendid choice! An invisible PDF, the rarest of them all. 
                    Ah, the beauty of blank pages, the elegance of nothingness. 
                    A true masterpiece of digital minimalism. Your avant-garde approach to file submission is truly
                    inspiring. Bravo, dear user, bravo!""")
            
            
if __name__ == '__main__':
    main()