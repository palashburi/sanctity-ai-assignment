# Classified Retrieval System

## Overview
This project implements a secure, access-controlled document retrieval system designed for classified information management. It integrates vector-based search, intent-driven graph traversal, and clearance-based access control to provide precise and secure responses to queries. The system supports both direct document retrieval and LLM-enhanced responses, with a Streamlit-based frontend for user interaction, including a lockdown mechanism and cipher-protected Secret Chat.

## Features
- **Vector Search**: Utilizes Chroma and HuggingFace embeddings for semantic search over classified documents.
- **Intent Graph**: Employs a directed graph (`IntentGraph` or `EnhancedIntentGraph`) to map queries to relevant document chunks based on intent.
- **Access Control**: Enforces agent clearance levels (1-13) to restrict access to sensitive information.
- **LLM Integration**: Uses Ollama (LLaMA2) for generating concise, context-aware responses when enabled.
- **Conversation Memory**: Maintains context using `ConversationSummaryBufferMemory` for coherent interactions.
- **Frontend**: Streamlit-based UI with normal and encrypted "Secret Chat" modes, protected by a cipher code (`EK7-ΣΔ19-βXQ//4437`).
- **JSON Rules Support**: Processes JSON-based rules for enhanced intent mapping and response styling.
- **Erase Functionality**: Locks down the system upon unauthorized access attempts, with a backend reset mechanism for authorized users.
- **Ollama and llama2**: Used Ollama to locally use the llama2 model , to reduce query times and avoid token limit exceeding and reducing the the query time .

## System Components

### Core Modules
- **`retrieval.py`**: Initial version of the access-controlled retriever using `IntentGraph` for intent mapping and Chroma for vector search.
- **`retrieval_v2.py`**: Enhanced retriever with JSON-based `EnhancedIntentGraph`, supporting response styles (standard, tactical, strategic, cryptic) and improved clearance enforcement.
- **`dfw.py`**: Defines `IntentGraph`, a NetworkX-based directed graph for intent mapping with nodes for protocols, safehouses, and operations.
- **`concept_extractor.py`**: Extracts seed concepts from queries using spaCy and YAKE, mapping to `IntentGraph` nodes.
- **`concept_extractor_v2.py`**: Updated concept extractor for `EnhancedIntentGraph`, using JSON rules.
- **`chroma_load.py`**: Loads and indexes Word documents into Chroma vectorstore with chunking.
- **`chroma_load_v2.py`**: Indexes JSON rules into Chroma, creating documents from intent labels and metadata.
- **`generate_intent_labels_from_rules.py`**: Generates intent labels from rule documents using TF-IDF and K-means clustering.
- **`app.py`**: Streamlit frontend for user interaction, supporting clearance level input, dual chat modes, erase lockdown, and backend reset functionality.
- **`test2.py`**: Test script for `retrieval_v2.py`, running queries with varying clearance levels.

### Key Dependencies
- **LangChain**: For document processing, vectorstores, and LLM chains.
- **HuggingFace Embeddings**: For semantic embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- **Chroma**: Persistent vector database for document retrieval.
- **Ollama**: Local LLM inference with LLaMA2.
- **NetworkX**: For intent graph construction and traversal.
- **spaCy & YAKE**: For concept extraction.
- **Streamlit**: For the web interface.
- **scikit-learn & pandas**: For intent label generation.

## Use of spaCy and YAKE
- **spaCy**: A robust NLP library used in `concept_extractor.py` and `concept_extractor_v2.py` to perform named entity recognition (NER), extracting entities like organizations and locations from queries.
- **YAKE**: A lightweight keyword extraction tool used to identify significant terms in queries, complementing spaCy by capturing contextually relevant phrases.
- **Reason for Use**: spaCy ensures precise entity detection for structured queries, while YAKE's unsupervised approach excels at extracting key terms without training, enabling robust concept mapping to intent graph nodes.
- **Impact**: Their combination enhances query understanding, bridging natural language inputs to the structured intents in `IntentGraph` and `EnhancedIntentGraph`, improving retrieval accuracy.

## Intent Graph Implementations
This section provides a technical analysis of the two intent graph implementations: `IntentGraph` (from `dfw.py`) and `EnhancedIntentGraph` (from provided code). Both classes leverage NetworkX's `DiGraph` for hierarchical intent mapping but differ in their data sources, structure, and functionality.

### IntentGraph (dfw.py)
**Description**:
- A static, predefined directed graph designed to map operational intents (e.g., "communication_verification", "extraction_protocols") to document chunks.
- Hard-coded nodes and edges represent intents, protocols, subprotocols, steps, and operations, each associated with a `chunk_id` linking to Chroma vectorstore documents.
- Nodes include metadata like `type` (e.g., intent, protocol, step), `clearance` (1-5), and `chunk_id`.

**Structure**:
- **Nodes**: Organized hierarchically (e.g., intent → protocol → subprotocol/step). Example: "communication_verification" → "LCC" → "Quantum Hashing".
- **Edges**: Defined with relations (e.g., "uses", "has_layer", "requires") to model operational workflows.
- **Key Methods**:
  - `get_all_related_chunks(seed_node)`: Retrieves all chunk IDs associated with a node and its successors using BFS.
  - `traverse_up(term)`: Returns parent nodes up to the root, useful for context expansion.
  - `get_successors(node)`: Lists immediate child nodes.
  - `visualize()`: Renders the graph using Matplotlib with spring layout.
- **Clearance Enforcement**: Each node has a `clearance` attribute, checked during retrieval to restrict access.

**Use Case**:
- Ideal for static, well-defined operational domains where intents and their relationships are fixed (e.g., predefined protocols like "Project Eclipse").
- Used in `retrieval.py` for initial retrieval logic, mapping queries to document chunks via `concept_extractor.py`.

**Performance**:
- **Time Complexity**: 
  - Graph construction: O(N + E), where N is nodes (fixed at ~30) and E is edges (~50), making it near-constant.
  - Chunk retrieval: O(V) for BFS traversal, where V is reachable nodes (typically small due to shallow hierarchy).
- **Space Complexity**: O(N + E), minimal due to static, small graph size.
- **Limitations**:
  - Static structure requires code changes to update intents or relationships.
  - Limited metadata (only clearance and chunk_id), reducing flexibility for dynamic response styling.
  - No support for weighted edges or confidence scores, limiting intent prioritization.

### EnhancedIntentGraph
**Description**:
- A dynamic, JSON-driven directed graph that constructs intent hierarchies from `final_enriched_rules.json`.
- Nodes represent intents, rules, and keywords, with rich metadata including `response_style`, `confidence`, `agent_level`, and `trigger_type`.
- Designed for flexible, rule-based intent mapping with support for response customization and clearance enforcement.

**Structure**:
- **Nodes**:
  - **Root**: A single "ROOT" node connects all intents.
  - **Intents**: Derived from JSON `intent_label`, with attributes like `category`, `clearance`, `response_style` (e.g., neutral, tactical), and `confidence`.
  - **Rules**: Named `RULE_<rule_id>`, containing `response`, `raw_text`, `trigger_type`, and `agent_level`.
  - **Keywords**: Extracted from JSON `keywords`, with `score` and normalized text, linked to rules and intents.
- **Edges**:
  - "contains": ROOT → intent.
  - "has_rule": intent → rule.
  - "triggered_by": rule → keyword.
  - "maps_to": keyword → intent, weighted by keyword `score`.
- **Key Methods**?!
  - `find_related_intents(keyword, threshold)`: Returns intents linked to a keyword with confidence above the threshold, sorted by confidence.
  - `get_response_flow(intent)`: Retrieves rules for an intent, including response metadata, sorted by confidence.
  - `get_clearance_for_rule(rule_id)`: Returns the clearance level for a rule, defaulting to 1.
  - `get_response_style(intent)`: Retrieves the response style (e.g., neutral, tactical) for an intent.
  - `visualize_subgraph(node, depth)`: Visualizes a subgraph around a node with colored nodes (intent: green, rule: coral, keyword: blue).
  - `get_operational_path(start_intent, max_depth)`: Returns a prioritized node sequence for mission planning using BFS with confidence-based sorting.

**Use Case**:
- Suited for dynamic environments where intents and rules evolve, as it loads from JSON without code changes.
- Used in `retrieval_v2.py` and `concept_extractor_v2.py` for advanced retrieval with response styling and keyword-driven intent matching.
- Supports complex queries requiring nuanced responses (e.g., tactical vs. strategic) and fine-grained clearance checks.

**Performance**:
- **Time Complexity**:
  - Graph construction: O(R * (I + K + E)), where R is rules, I is intents, K is keywords per rule, and E is edges. For typical JSON with ~100 rules and ~10 keywords/rule, this is manageable.
  - Intent retrieval: O(K + E_k) for `find_related_intents`, where K is keywords and E_k is edges to intents (small per keyword).
  - Response flow: O(R_i * log R_i) for sorting rules of an intent, where R_i is rules per intent (~1-10).
- **Space Complexity**: O(N + E), where N includes intents, rules, and keywords (~1000 nodes for large JSON), and E is edges (~2000). Higher than `IntentGraph` but scalable.
- **Advantages**:
  - Dynamic updates via JSON, enabling real-time rule changes.
  - Rich metadata supports response customization and confidence-based prioritization.
  - Weighted edges (`score`) improve intent matching accuracy.
- **Limitations**:
  - Higher memory footprint due to keyword nodes and metadata.
  - JSON parsing and graph construction introduce startup latency.
  - Complexity in rule design may require careful tuning to avoid intent overlap.

### Comparative Analysis
| Feature | IntentGraph | EnhancedIntentGraph |
|---------|-------------|--------------------|
| **Data Source** | Hard-coded | JSON (`final_enriched_rules.json`) |
| **Node Types** | Intent, protocol, subprotocol, step, operation | Root, intent, rule, keyword |
| **Metadata** | Clearance, chunk_id | Clearance, response_style, confidence, category, trigger_type, score |
| **Edge Weights** | None | Keyword → intent (score-based) |
| **Flexibility** | Static, requires code changes | Dynamic, JSON-driven |
| **Use Case** | Fixed operational protocols | Dynamic, rule-based intent mapping |
| **Retrieval** | Chunk-based, clearance-focused | Rule-based, style- and confidence-driven |
| **Visualization** | Full graph, simple | Subgraph, type-colored nodes |
| **Performance** | Faster for small, static graphs | Slower startup, scalable for large rules |
| **Complexity** | Simple, minimal metadata | Complex, rich metadata |

**Recommendations**:
- Use `IntentGraph` for deployments with fixed, well-defined protocols where speed and simplicity are critical (e.g., embedded systems with limited resources).
- Use `EnhancedIntentGraph` for scalable, evolving systems requiring dynamic intent updates and customized responses (e.g., centralized intelligence platforms).
- Optimize `EnhancedIntentGraph` by indexing keywords for faster lookups and pruning low-confidence rules to reduce graph size.

## JSON Rules Generation
This section analyzes the `generate_intent_labels_from_rules.py` script, which processes a Word document (`rules.docx`) to generate intent labels and keywords, ultimately contributing to the creation of `final_enriched_rules.json` used by `EnhancedIntentGraph` and `chroma_load_v2.py`.

### Process Overview
The script extracts rules from a Word document, vectorizes their text, clusters them using K-means, and assigns intent labels based on cluster keywords. The output is a CSV file (`rules_with_intent_labels.csv`), which is further processed (manually or via additional scripts) to create the enriched JSON file with metadata like response styles, clearance levels, and keyword scores.

**Key Steps**:
1. **Rule Extraction (`extract_rules`)**:
   - Reads `rules.docx` using `python-docx`.
   - Parses paragraphs with regex (`r"Rule\s*(\d+):\s*(.+)"`) to extract rule IDs and text.
   - Returns a pandas DataFrame with columns `rule_id` and `text`.
   - **Performance**: O(P), where P is the number of paragraphs (typically hundreds). Regex matching is fast, but large documents may increase I/O time.
   - **Output Example**:
     ```python
     rule_id | text
     1       | "Initiate secure comms with Quantum Hashing"
     2       | "Execute Shadow Step protocol under compromise"
     ```

2. **Text Vectorization (`vectorize_rules`)**:
   - Uses `TfidfVectorizer` with English stop words and a limit of 300 features to convert rule texts into TF-IDF vectors.
   - TF-IDF weights terms based on their frequency in a rule relative to the entire corpus, emphasizing distinctive terms.
   - **Performance**: O(R * T), where R is rules and T is unique terms (capped at 300). Sparse matrix output reduces memory usage.
   - **Output**: A `TfidfVectorizer` object and a sparse matrix `X` (R × 300).

3. **Clustering (`cluster_rules`)**:
   - Applies K-means clustering with `num_clusters=12` (configurable, set to 14 in `__main__`) on the TF-IDF matrix.
   - Uses `random_state=42` for reproducibility.
   - Groups rules into clusters based on semantic similarity.
   - **Performance**: O(R * F * I * K), where F is features (300), I is iterations (typically <100), and K is clusters (14). Computationally intensive for large R.
   - **Output**: A fitted `KMeans` model with cluster assignments.

4. **Keyword Extraction (`get_cluster_keywords`)**:
   - Identifies top 10 keywords per cluster by sorting the K-means cluster centers.
   - Assigns a `possible_intent` label as a comma-separated string of the top 3 keywords.
   - **Performance**: O(K * F * log F) for sorting 300 features per cluster. Minimal overhead.
   - **Output Example**:
     ```python
     cluster_id | top_keywords                              | possible_intent
     0          | ['quantum', 'hashing', 'secure', ...]     | quantum, hashing, secure
     1          | ['shadow', 'protocol', 'compromise', ...] | shadow, protocol, compromise
     ```

5. **Intent Labeling (`generate_intent_labels`)**:
   - Assigns cluster IDs to rules based on K-means labels.
   - Merges rule DataFrame with cluster DataFrame to include `possible_intent` and `top_keywords`.
   - Outputs a DataFrame with columns `rule_id`, `text`, `cluster_id`, `possible_intent`, `top_keywords`.
   - Saves to `rules_with_intent_labels.csv`.
   - **Performance**: O(R) for merging and saving. I/O bound for large outputs.
   - **Output Example** (CSV):
     ```csv
     rule_id,text,cluster_id,possible_intent,top_keywords
     1,"Initiate secure comms with Quantum Hashing",0,"quantum, hashing, secure","['quantum', 'hashing', 'secure', ...]"
     2,"Execute Shadow Step protocol under compromise",1,"shadow, protocol, compromise","['shadow', 'protocol', 'compromise', ...]"
     ```

**JSON Creation**:
- The CSV is not directly converted to `final_enriched_rules.json` by the script. Instead, it serves as an intermediate step.
- Additional processing (likely manual or via a separate script) enriches the CSV data with fields like:
  - `intent_label`: Derived from `possible_intent`, possibly refined manually.
  - `category`: Assigned based on domain (e.g., "communication", "extraction").
  - `response_style`: Added to specify response type (e.g., "tactical", "strategic").
  - `agent_level`: Clearance level (1-13), assigned based on rule sensitivity.
  - `response`: Predefined response text for the rule.
  - `trigger_type`: Type of query trigger (e.g., "keyword", "intent").
  - `intent_confidence`: Confidence score, possibly from clustering or manual tuning.
  - `keywords`: List of `{"keyword": str, "score": float}` from `top_keywords`, with scores likely derived from TF-IDF or clustering.
  - `greeting`: Custom greeting for responses.
- The enriched data is structured as a JSON array of rule objects, e.g.:
  ```json
  [
    {
      "rule_id": 1,
      "intent_label": "secure_communication",
      "category": "communication",
      "response_style": "tactical",
      "agent_level": 3,
      "response": "Initiate Quantum Hashing protocol.",
      "raw_text": "Initiate secure comms with Quantum Hashing",
      "trigger_type": "keyword",
      "intent_confidence": 0.85,
      "keywords": [
        {"keyword": "quantum", "score": 0.9},
        {"keyword": "hashing", "score": 0.8}
      ],
      "greeting": "Salute, Shadow Cadet."
    },
    ...
  ]
  ```
- This JSON is consumed by `EnhancedIntentGraph` to build the graph and by `chroma_load_v2.py` to index rules.

**Performance**:
- **Overall Time Complexity**: Dominated by K-means clustering, O(R * F * I * K). For R=1000 rules, F=300, I=50, K=14, this is computationally feasible but may take seconds on standard hardware.
- **Space Complexity**: O(R * F) for the TF-IDF matrix (sparse) and O(R + K * F) for DataFrames. Memory usage is moderate but scales with rule count.
- **Limitations**:
  - K-means assumes spherical clusters, which may not capture complex semantic relationships.
  - Fixed `num_clusters` requires tuning to avoid over- or under-segmentation.
  - Manual enrichment for JSON adds a non-automated step, prone to errors.
- **Advantages**:
  - TF-IDF ensures robust feature extraction, focusing on distinctive terms.
  - K-means provides interpretable clusters, with top keywords serving as intuitive intent labels.
  - CSV output enables easy inspection and modification before JSON creation.

**Integration**:
- The JSON drives `EnhancedIntentGraph`, enabling dynamic intent mapping and response styling.
- `chroma_load_v2.py` uses the JSON to create Chroma documents, embedding rule content and metadata for vector search.
- The process ensures rules are both graph-navigable (via `EnhancedIntentGraph`) and searchable (via Chroma), enhancing retrieval precision.

**Recommendations**:
- Automate the CSV-to-JSON step with a script to map `possible_intent` to `intent_label` and add metadata.
- Use advanced clustering (e.g., HDBSCAN) for non-spherical clusters.
- Tune `num_clusters` dynamically based on silhouette scores.
- Cache TF-IDF vectors for incremental updates to large rule sets.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure `requirements.txt` includes:
   ```
   langchain
   langchain-community
   langchain-chroma
   langchain-huggingface
   streamlit
   networkx
   spacy
   yake
   ollama
   scikit-learn
   pandas
   python-docx
   matplotlib
   ```
3. **Download spaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **Set Up Chroma Index**:
   - Run `chroma_load.py` to index Word documents:
     ```bash
     python chroma_load.py
     ```
   - Run `chroma_load_v2.py` to index JSON rules:
     ```bash
     python chroma_load_v2.py
     ```
5. **Prepare JSON Rules**:
   - Generate intent labels from rules:
     ```bash
     python generate_intent_labels_from_rules.py
     ```
   - Ensure `final_enriched_rules.json` is in the root directory.
6. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage
- **Access the System**:
  - Open the Streamlit app (default: `http://localhost:8501`).
  - Enter an agent clearance level (1-13).
  - For levels ≥7, choose between "Normal Chat" (uses `retrieval_v2.py`) or "Secret Chat" (uses `retrieval.py`, requires cipher `EK7-ΣΔ19-βXQ//4437`).
- **Querying**:
  - Input queries in the chat interface.
  - Responses are filtered by clearance level and intent, with LLM-enhanced answers recurse
- **Erase Functionality**:
  - Triggered by entering an incorrect cipher code in Secret Chat or attempting access with a clearance level >13.
  - Creates a `.erased.lock` file, locking the app and displaying an "ERASED" message.
  - **Reset**: Authorized users (clearance ≥10) can reset the lockdown via a backend interface in the Streamlit app, which removes the `.erased.lock` file and restarts the application.
- **Secret Chat Cipher**:
  - Access to Secret Chat requires the cipher code `EK7-ΣΔ19-βXQ//4437`.
  - Correct entry grants access; incorrect entry triggers the erase lockdown.
- **Testing**:
  - Run `test2.py` to execute predefined test cases:
    ```bash
    python test2.py
    ```

## Security Notes
- **Clearance Levels**: Higher levels (e.g., 5) access sensitive protocols like "Project Eclipse." Levels >13 trigger a lockdown.
- **Cipher Protection**: Secret Chat is locked behind the cipher `EK7-ΣΔ19-βXQ//4437`. Incorrect attempts create an `.erased.lock` file, rendering the app unusable until reset.
- **Data Sensitivity**: Ensure `SECRET INFO MANUAL.docx` and `final_enriched_rules.json` are secured, as they contain classified data.

## Future Improvements
- Enhance response styles with dynamic templates.
- Implement real-time intent graph updates from new rules.
- Add multi-user support with session isolation.
- Optimize Chroma index for larger document sets.
- Automate CSV-to-JSON conversion for rule enrichment.
- Use advanced clustering algorithms (e.g., HDBSCAN) for better intent segmentation.

## License
This project is classified and not licensed for public distribution. Authorized access only.