from typing import List, Optional, Dict
from langchain_core.documents import Document 
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from enhanced_intent_graph import EnhancedIntentGraph
from concept_extractor import extract_seed_concept
from langchain_community.llms import Ollama

AGENT_CLEARANCE_LEVEL = 1

AGENT_GREETINGS = {
    1: "Salute, Shadow Cadet.",
    2: "Bonjour, Sentinel.",
    3: "Eyes open, Phantom.",
    4: "In the wind, Commander.",
    5: "The unseen hand moves, Whisper."
}


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

llm = Ollama(model="llama2")
prompt_template = ChatPromptTemplate.from_template(
    """{greeting}
    Use only this context to answer.
    Keep answers concise and technical. Consider agent clearance level.
    Maximum 7 lines (lists excluded).
    
    Context: {context}
    
    Question: {question}"""
)
llm_chain = prompt_template | llm | StrOutputParser()

summary_memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True
)

class JSONAccessControlledRetriever:
    def __init__(self, json_rules_path: str):
        """
        Initialize with path to JSON rules file
        """
        self.vectorstore = vectorstore
        self.intent_graph = EnhancedIntentGraph(json_rules_path)
        self.summary_memory = summary_memory

    def retrieve(self, query: str, seed_concept: Optional[str] = None, 
            use_llm: bool = False) -> List:
        
      
        greeting = AGENT_GREETINGS.get(AGENT_CLEARANCE_LEVEL, "Security Protocol Active:")
        
   
        seed_concept = seed_concept or extract_seed_concept(query)
        
    
        graph_chunks = []
        response_style = "standard"
        if seed_concept:
            for rule in self.intent_graph.get_response_flow(seed_concept):
                if 'chunk_id' in rule:
                    graph_chunks.append(rule['chunk_id'])
                response_style = rule.get('response_style', response_style)


        vector_results = self.vectorstore.similarity_search(query, k=8)

        graph_docs = [
            doc for doc in vector_results 
            if doc.metadata.get("chunk_id") in graph_chunks
        ] if graph_chunks else []

        fused_docs = self.rrf_fuse(vector_results, graph_docs)

        if not fused_docs:
            return [f"{greeting}\nNo matching data found."]
        
        for doc in fused_docs:
            rule_id = doc.metadata.get("rule_id")
            clearance = self.intent_graph.get_clearance_for_rule(rule_id)
            if AGENT_CLEARANCE_LEVEL < clearance:
                return [f"{greeting}\nAccess Denied - Clearance Level Insufficient"]

        if use_llm:
            context = "\n\n".join(doc.page_content for doc in fused_docs)
            self.summary_memory.save_context(
                {"input": "New operational context"},
                {"output": context}
            )
            summarized = self.summary_memory.load_memory_variables({})["history"]
 
            style_guide = {
                "standard": "Provide complete technical information.",
                "tactical": "Focus on actionable steps and procedures.",
                "strategic": "Include multi-layered analysis and alternatives.",
                "cryptic": "Respond with indirect references only."
            }.get(response_style, "")

            llm_response = llm_chain.invoke({
                "greeting": greeting,
                "context": f"{summarized}\nResponse Style: {style_guide}",
                "question": query
            })

            if not llm_response.startswith(greeting.split(",")[0]):
                return f"{greeting}\n{llm_response}"
            return llm_response

        if isinstance(fused_docs, list):
            return [Document(
                page_content=f"{greeting}\n{doc.page_content}",
                metadata=doc.metadata
            ) for doc in fused_docs]
        return fused_docs

    def rrf_fuse(self, vector_results: List, graph_docs: List, k: int = 5) -> List:
        """
        Reciprocal Rank Fusion for combining results
        """
        doc_scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(vector_results):
            doc_scores[doc.page_content] = doc_scores.get(doc.page_content, 0) + 1/(rank + 1)
            doc_map[doc.page_content] = doc
            
        for rank, doc in enumerate(graph_docs):
            doc_scores[doc.page_content] = doc_scores.get(doc.page_content, 0) + 1/(rank + 1)
            doc_map[doc.page_content] = doc
            
        return [
            doc_map[content] 
            for content, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        ]

def set_agent_clearance(level: int):
    global AGENT_CLEARANCE_LEVEL
    AGENT_CLEARANCE_LEVEL = level