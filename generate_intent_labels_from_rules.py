import re
import json
import sys
from docx import Document
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
import spacy
from sentence_transformers import SentenceTransformer, util

# Load models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global intent labels
INTENT_LABELS = [
    "Surveillance",
    "Emergency Protocol",
    "Disguise & Identity",
    "Safehouse Operations",
    "Encryption & Signals",
    "Psychological Warfare",
    "Counterintelligence",
    "Mission Intelligence",
    "Disinformation & Media",
    "Tactical Procedures",
    "Classified Asset Handling",
    "Technology & Equipment",
    "Cryptographic Access",
    "General Intelligence"
]
INTENT_EMBEDDINGS = embedder.encode(INTENT_LABELS, convert_to_tensor=True)

# Greeting & Style Map
AGENT_STYLE_MAP = {
    1: {
        "greeting": "Salute, Shadow Cadet.",
        "response_style": "Basic and instructional, like a mentor guiding a trainee."
    },
    2: {
        "greeting": "Bonjour, Sentinel.",
        "response_style": "Tactical and direct, focusing on execution and efficiency."
    },
    3: {
        "greeting": "Eyes open, Phantom.",
        "response_style": "Analytical and multi-layered, providing strategic insights."
    },
    4: {
        "greeting": "In the wind, Commander.",
        "response_style": "Coded language, hints, and only essential confirmations."
    },
    5: {
        "greeting": "The unseen hand moves, Whisper.",
        "response_style": "Vague, layered, sometimes answering with counter-questions."
    }
}

# --- PHRASE EXTRACTION BLOCK ---
def extract_keywords_yake(text, max_keywords=15):
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=max_keywords)
    return [{"keyword": kw[0].lower(), "score": 1 - kw[1]} for kw in kw_extractor.extract_keywords(text)]

def extract_keywords_spacy(text):
    doc = nlp(text)
    return [{"keyword": chunk.text.strip().lower(), "score": 1.0} for chunk in doc.noun_chunks]

def extract_tfidf_keywords(texts, top_n=15):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    result = []
    for row in X.toarray():
        row_keywords = [{"keyword": terms[i], "score": row[i]} for i in row.argsort()[-top_n:][::-1]]
        result.append(row_keywords)
    return result

def merge_and_rank_keywords(yake_kw, spacy_kw, tfidf_kw, top_n=15):
    weights = {"yake": 1.5, "spacy": 0.5, "tfidf": 1.0}
    keyword_map = defaultdict(lambda: {"total_score": 0.0, "count": 0})
    for source, kws in [("yake", yake_kw), ("tfidf", tfidf_kw), ("spacy", spacy_kw)]:
        for kw in kws:
            k = kw["keyword"]
            keyword_map[k]["total_score"] += kw["score"] * weights[source]
            keyword_map[k]["count"] += weights[source]
    ranked = [{"keyword": k, "score": v["total_score"] / v["count"]} for k, v in keyword_map.items()]
    return sorted(ranked, key=lambda x: x["score"], reverse=True)[:top_n]

# --- INTENT & STYLE MAPPERS ---
def classify_intent(text):
    embedding = embedder.encode(text, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embedding, INTENT_EMBEDDINGS)
    best = int(scores.argmax())
    return INTENT_LABELS[best], float(scores[0][best])

def enrich_style(agent_level):
    return AGENT_STYLE_MAP.get(agent_level, {
        "greeting": "Hello, Operative.",
        "response_style": "Standard response style."
    })

# --- RULE EXTRACTION BLOCK ---
def extract_rules_from_doc(doc_path):
    document = Document(doc_path)
    rules = []
    current_category = None
    current_rule = None
    rule_start = re.compile(r'^Rule\s+(\d+):\s*(.*)', re.IGNORECASE)
    level_pattern = re.compile(r'Level[-\s]?(\d+)', re.IGNORECASE)

    category_headers = [
        "Basic Operational Queries", "Advanced Covert Operations Queries",
        "Technology, Encryption & Intelligence Queries", "Counterintelligence & Strategic Planning Queries",
        "Cyber & Intelligence Queries", "Psychological Warfare & Disinformation",
        "High-Level Strategic Operations", "Miscellaneous Security & Verification Queries"
    ]

    for para in document.paragraphs:
        text = para.text.strip()
        if not text: continue
        for header in category_headers:
            if header.lower() in text.lower():
                current_category = header
        match = rule_start.match(text)
        if match:
            if current_rule:
                rules.append(current_rule)
            rule_id = int(match.group(1))
            rule_text = match.group(2).strip()
            current_rule = {
                "rule_id": rule_id,
                "response": rule_text,
                "raw_text": text,
                "agent_level": None,
                "trigger_type": "unknown",
                "keywords": [],
                "category": current_category or "Uncategorized"
            }
            level_match = level_pattern.search(text)
            if level_match:
                current_rule["agent_level"] = int(level_match.group(1))
                current_rule["trigger_type"] = "level_and_keyword"
            if "if the phrase" in text.lower() or "if a query" in text.lower():
                current_rule["trigger_type"] = "phrase_trigger"
        else:
            if current_rule:
                current_rule["response"] += "\n" + text
                current_rule["raw_text"] += "\n" + text

    if current_rule:
        rules.append(current_rule)
    return rules

def parse_and_enrich_rules(doc_path):
    rules = extract_rules_from_doc(doc_path)
    all_texts = [rule["response"] for rule in rules]
    tfidf_keywords_all = extract_tfidf_keywords(all_texts)

    for i, rule in enumerate(rules):
        yake_kw = extract_keywords_yake(rule["response"])
        spacy_kw = extract_keywords_spacy(rule["response"])
        tfidf_kw = tfidf_keywords_all[i]
        rule["keywords"] = merge_and_rank_keywords(yake_kw, spacy_kw, tfidf_kw)
        rule["intent_label"], rule["intent_confidence"] = classify_intent(rule["response"])
        style = enrich_style(rule.get("agent_level"))
        rule["greeting"] = style["greeting"]
        rule["response_style"] = style["response_style"]

    return rules

def main():
    doc_path = "rules.docx"
    try:
        enriched_rules = parse_and_enrich_rules(doc_path)
        output_path = "final_enriched_rules.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched_rules, f, indent=4)
        print(f"✅ Saved {len(enriched_rules)} enriched rules to {output_path}")
    except Exception as e:
        print("❌ Error during rule processing:", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()



