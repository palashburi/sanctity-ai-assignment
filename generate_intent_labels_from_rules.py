# to generate the json from rule docs
import re
import pandas as pd
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def extract_rules(doc_path):
    doc = Document(doc_path)
    rules = []
    for para in doc.paragraphs:
        text = para.text.strip()
        match = re.match(r"Rule\s*(\d+):\s*(.+)", text)
        if match:
            rules.append({
                "rule_id": int(match.group(1)),
                "text": match.group(2)
            })
    return pd.DataFrame(rules)


def vectorize_rules(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
    X = vectorizer.fit_transform(df['text'])
    return vectorizer, X


def cluster_rules(X, num_clusters=12):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans


def get_cluster_keywords(kmeans, vectorizer, num_keywords=10):
    terms = vectorizer.get_feature_names_out()
    clusters = []
    for i in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[i]
        top_indices = center.argsort()[-num_keywords:][::-1]
        top_keywords = [terms[idx] for idx in top_indices]
        clusters.append({
            "cluster_id": i,
            "top_keywords": top_keywords,
            "possible_intent": ", ".join(top_keywords[:3])
        })
    return pd.DataFrame(clusters)

def generate_intent_labels(doc_path, num_clusters=12):
    rules_df = extract_rules(doc_path)
    vectorizer, X = vectorize_rules(rules_df)
    kmeans = cluster_rules(X, num_clusters=num_clusters)
    cluster_df = get_cluster_keywords(kmeans, vectorizer)

    
    rules_df["cluster_id"] = kmeans.labels_
    merged_df = rules_df.merge(cluster_df, on="cluster_id")

    return merged_df[["rule_id", "text", "cluster_id", "possible_intent", "top_keywords"]]


if __name__ == "__main__":
    doc_path = "rules.docx"  
    result_df = generate_intent_labels(doc_path, num_clusters=14)
    result_df.to_csv("rules_with_intent_labels.csv", index=False)
    print("✅ Saved intent-labeled rules to 'rules_with_intent_labels.csv'")
