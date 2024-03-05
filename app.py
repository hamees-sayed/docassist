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
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Initialize NLTK resources
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

# Clean text
def clean_text(text):
    non_alphanumeric_re = re.compile(r'[^a-zA-Z0-9\s]')
    whitespace_re = re.compile(r'\s+')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = text.lower()
    text = non_alphanumeric_re.sub('', text)
    text = whitespace_re.sub(' ', text)
    
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Extract and clean text from PDF
def extract_and_clean(file):
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page in range(num_pages):
        text += pdf_reader.pages[page].extract_text()
        
    cleaned_text = clean_text(text)
    return cleaned_text

# Extract text from PDF
def extract_text_from_pdf(file):
    with Pool() as pool:
        texts = pool.map(extract_and_clean, [file])
    return '\n'.join(texts)

# Generate response
def generate_response(pdf, query, history, temperature=0.5):
    text = extract_text_from_pdf(pdf)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    db = Chroma.from_texts(chunks, embeddings, persist_directory="./chroma_db")
    db.persist()
    context = "\n".join([message["content"] for message in history])
    
    prompt_template = """
    You are helpful assistant, a teacher and a friend, please answer the question in as much detail as possible based on the provided 
    context and the history of the conversation and keep it simple so that even a beginner can understand. And also converse 
    with the user if necessary. Thank you.
    \n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature)
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = db.similarity_search(query)
    response = chain({"input_documents":docs, "question": query, "context":context}, return_only_outputs=True)["output_text"]
    
    return response

# Main function
def main():
    # Load NLTK resources
    initialize_nltk()
    
    # Load environment variables
    load_dotenv()
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Set up Streamlit page configuration
    st.set_page_config(page_title="DocAssist 💬", page_icon="🤖", layout="wide")
    
    # Streamlit sidebar
    with st.sidebar:
        st.title('DocAssist 💬')
        st.header("1. Upload PDF")
        pdf = st.file_uploader("**Upload your PDF**", type='pdf')
        temperature = st.slider(
            "Select the creativity temperature for the AI",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
        st.markdown('## About')
        st.markdown('This app is an LLM-powered chatbot built using:')
        st.markdown('- [Streamlit](https://streamlit.io/)')
        st.markdown('- [LangChain](https://python.langchain.com/)')
        st.markdown('- [Gemini](https://ai.google.dev/)')
        st.write('Made by [Hamees Sayed](https://hamees-sayed.github.io/) - [Source](https://github.com/hamees-sayed/docassist)')
    
    # Chatbot functionality
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
            response = generate_response(pdf, query, st.session_state.messages, temperature)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            st.write("""Oh, splendid choice! An invisible PDF, the rarest of them all. 
                    Ah, the beauty of blank pages, the elegance of nothingness. 
                    A true masterpiece of digital minimalism. Your avant-garde approach to file submission is truly
                    inspiring. Bravo, dear user, bravo!""")

# Entry point
if __name__ == '__main__':
    main()
