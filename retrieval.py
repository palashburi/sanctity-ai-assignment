

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dfw import IntentGraph
from concept_extractor import extract_seed_concept
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory  # ++


# Global agent clearance
AGENT_CLEARANCE_LEVEL = 1

# Load persisted vectorstore
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    collection_name="classified_docs",
    persist_directory="./chroma_index",
    embedding_function=embedding
)

# Initialize LLM chain
llm = Ollama(model="llama2")
prompt_template = ChatPromptTemplate.from_template(
    """You are a raw agent talk like a agent(strict nonchalant talks) . Use only the following context to answer.
    Keep answers concise and technical. See the documents chunks provided you to compare with agent level always
    and provide the best answer. dont speak more than 7 lines (if providing a list or option dont count lines)
    
    Context: {context}
    
    Question: {question}"""
)
llm_chain = prompt_template | llm | StrOutputParser()

summary_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)

def set_agent_clearance(level):
    global AGENT_CLEARANCE_LEVEL
    AGENT_CLEARANCE_LEVEL = level

class AccessControlledRetriever:
    def __init__(self):
        self.vectorstore = vectorstore
        self.intent_graph = IntentGraph()
        self.summary_memory = summary_memory

    def retrieve(self, query, seed_concept=None, use_llm=False):
        seed_concept = seed_concept or extract_seed_concept(query)
        graph_chunks = self.intent_graph.get_all_related_chunks(seed_concept) if seed_concept else []


        vector_results = self.vectorstore.similarity_search(query, k=8)


        graph_docs = []
        if graph_chunks:
            graph_docs = [d for d in vector_results if d.metadata.get("chunk_id") in graph_chunks]


        fused_docs = self.rrf_fuse(vector_results, graph_docs)

        # Clearance enforcement
        if not fused_docs:
            return ["No matching data found."]
        
        for doc in fused_docs:
            chunk_id = doc.metadata.get("chunk_id")
            for node, data in self.intent_graph.graph.nodes(data=True):
                if data.get("chunk_id") == chunk_id:
                    required_clearance = data.get("clearance", 1)
                    if AGENT_CLEARANCE_LEVEL < required_clearance:
                        return ["Access Denied – Clearance Insufficient."]

        if use_llm:
            context = "\n\n".join([doc.page_content for doc in fused_docs])
            self.summary_memory.save_context(
                {"input": "New operational context"},
                {"output": context}
            )
            summarized_context = self.summary_memory.load_memory_variables({})["history"]
            
            return llm_chain.invoke({
                "context": summarized_context, # for summary storage, did this to maintain some sort of context
                "question": query
            })
        
        return fused_docs

    def rrf_fuse(self, vector_results, graph_docs, k=5):
        doc_scores = {}
        doc_map = {} 

        for rank, doc in enumerate(vector_results):
            doc_scores[doc.page_content] = doc_scores.get(doc.page_content, 0) + 1 / (rank + 1)
            doc_map[doc.page_content] = doc

        for rank, doc in enumerate(graph_docs):
            doc_scores[doc.page_content] = doc_scores.get(doc.page_content, 0) + 1 / (rank + 1)
            doc_map[doc.page_content] = doc  

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in ranked[:k]]