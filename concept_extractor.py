# concept_extractor.py
import spacy
import yake
from dfw import IntentGraph  

nlp = spacy.load("en_core_web_sm")
kw_extractor = yake.KeywordExtractor()
graph = IntentGraph()

def extract_seed_concept(query: str) -> str | None:
    doc = nlp(query)
    entities = [ent.text.lower() for ent in doc.ents]
    keywords = [kw.lower() for kw, _ in kw_extractor.extract_keywords(query)]
    candidates = set(entities + keywords)

    all_nodes = set(graph.graph.nodes)
    matches = [node for node in all_nodes if node.lower() in candidates]
    return matches[0] if matches else None
