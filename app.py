import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAI

# Hàm để đọc và trích xuất text từ các file PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Hàm để đọc text từ các file TXT
def get_text_from_files(text_files):
    text = ""
    for text_file in text_files:
        text += text_file.read().decode("utf-8")
    return text

# Hàm để chia text thành các đoạn nhỏ (chunks)
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Hàm để tạo vector store từ các đoạn text
def get_vectorstore(text_chunks):
    # Sử dụng mô hình embedding miễn phí từ Hugging Face
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    
    # Tạo vector store bằng FAISS
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def main():
    # Tải biến môi trường (chứa API key)
    load_dotenv()
    # Lấy API key từ biến môi trường
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Cấu hình API key cho mô hình của Google
    if api_key:
        GoogleGenerativeAI.configure(api_key=api_key)
    else:
        st.error("Chưa tìm thấy GOOGLE_API_KEY. Vui lòng tạo file .env và thêm key vào.")
        return

    # --- GIAO DIỆN STREAMLIT ---
    st.set_page_config(page_title="Hỏi đáp tài liệu của bạn", page_icon=":books:")

    st.header("Hỏi đáp tài liệu của bạn :books:")
    st.write("Upload tài liệu của bạn (PDF hoặc TXT) và đặt câu hỏi về nội dung bên trong.")

    # Sử dụng session_state để lưu trữ các biến qua mỗi lần chạy lại app
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Vùng sidebar để upload file
    with st.sidebar:
        st.subheader("Tài liệu của bạn")
        uploaded_files = st.file_uploader(
            "Upload file PDF hoặc TXT và nhấn 'Xử lý'", accept_multiple_files=True, type=['pdf', 'txt'])
        
        if st.button("Xử lý"):
            if not uploaded_files:
                st.warning("Vui lòng upload ít nhất một file.")
            else:
                with st.spinner("Đang xử lý tài liệu..."):
                    # 1. Lấy text từ file
                    raw_text = ""
                    pdf_files = [f for f in uploaded_files if f.type == "application/pdf"]
                    text_files = [f for f in uploaded_files if f.type == "text/plain"]
                    
                    if pdf_files:
                        raw_text += get_pdf_text(pdf_files)
                    if text_files:
                        raw_text += get_text_from_files(text_files)
                    
                    # 2. Chia text thành các chunk
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Tạo vector store
                    # Lưu vector store vào session state để không phải tạo lại
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                    st.success("Xử lý tài liệu thành công!")

    # Vùng chính để hỏi đáp
    user_question = st.text_input("Hãy đặt câu hỏi về tài liệu của bạn:")

    if user_question:
        if st.session_state.vectorstore is None:
            st.warning("Vui lòng upload và xử lý tài liệu trước khi đặt câu hỏi.")
        else:
            with st.spinner("Đang tìm câu trả lời..."):
                # Tìm kiếm các chunk liên quan trong vector store
                docs = st.session_state.vectorstore.similarity_search(user_question)
                
                # Tạo chuỗi hỏi đáp (QA chain)
                llm = GoogleGenerativeAI(model="models/gemini-pro", temperature=0.3)
                chain = load_qa_chain(llm, chain_type="stuff")
                
                # Chạy chain để có câu trả lời
                response = chain.run(input_documents=docs, question=user_question)
                
                st.write("### Câu trả lời:")
                st.write(response)

if __name__ == '__main__':
    main()