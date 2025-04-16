from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


loader = Docx2txtLoader("SECRET INFO MANUAL.docx")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)


texts = []
metadatas = []
for i, chunk in enumerate(chunks):
    texts.append(chunk.page_content)
    metadatas.append({"chunk_id": f"doc_chunk_{i+1}"})


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding,
    metadatas=metadatas,
    collection_name="classified_docs",
    persist_directory="./chroma_index"
)

print("✅ Chroma index created successfully.")
