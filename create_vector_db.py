import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# Folder containing PDF files
PDF_FOLDER = "data"
# Load all PDF documents
documents = []
if not os.path.exists(PDF_FOLDER):
    raise FileNotFoundError(f"Folder '{PDF_FOLDER}' does not exist.")
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(PDF_FOLDER, filename)
        print(f"Loading: {filename}")
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
print(f"\nTotal pages loaded: {len(documents)}")
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")
# Load Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Create FAISS Vector Store
vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)
# Save Vector Database
VECTOR_DB_PATH = "vectorstore"
vector_db.save_local(VECTOR_DB_PATH)
print(f"\nVector database saved successfully to '{VECTOR_DB_PATH}'")
print("Chatbot knowledge base is ready!")
*++
