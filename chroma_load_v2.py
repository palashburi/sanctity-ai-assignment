
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

def clean_metadata(metadata: dict) -> dict:
    """Ensure all metadata values are Chroma-compatible (str, int, float, or bool)"""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""  # Convert None to empty string
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)  # Convert other types to string
    return cleaned

# Load  JSON rules
with open("final_enriched_rules.json", "r", encoding="utf-8") as f:
    rules = json.load(f)

# rules to documents
docs = []
for rule in rules:
    content = f"""
    Intent: {rule['intent_label']}
    Rule ID: {rule['rule_id']}
    Keywords: {', '.join(k['keyword'] for k in rule['keywords'])}
    Response: {rule.get('response', 'N/A')}
    Clearance: {rule.get('agent_level', 1)}
    """
    
    # Create and clean metadata
    metadata = {
        "rule_id": rule['rule_id'],
        "intent": rule['intent_label'],
        "clearance": rule.get('agent_level', 1)
    }
    metadata = clean_metadata(metadata)
    
    docs.append(Document(
        page_content=content.strip(),
        metadata=metadata
    ))


splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)
chunks = splitter.split_documents(docs)

# Prepare texts and metadata
texts = []
metadatas = []
for i, chunk in enumerate(chunks):
    texts.append(chunk.page_content)
    chunk_metadata = {
        **chunk.metadata,
        "chunk_id": f"rule_chunk_{i+1}"
    }
    metadatas.append(clean_metadata(chunk_metadata))


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embedding,
    metadatas=metadatas,
    collection_name="json_rules",
    persist_directory="./chroma_index"
)

print("✅ Chroma index created successfully from JSON rules.")