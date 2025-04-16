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
- **Ollama and llama2**: Used Ollama to locally use the llama2 model, to reduce query times and avoid token limit exceeding and reducing the query time.

## System Components

### Core Modules

- `retrieval.py`: Initial version of the access-controlled retriever using `IntentGraph` for intent mapping and Chroma for vector search.
- `retrieval_v2.py`: Enhanced retriever with JSON-based `EnhancedIntentGraph`, supporting response styles (standard, tactical, strategic, cryptic) and improved clearance enforcement.
- `dfw.py`: Defines `IntentGraph`, a NetworkX-based directed graph for intent mapping with nodes for protocols, safehouses, and operations.
- `concept_extractor.py`: Extracts seed concepts from queries using spaCy and YAKE, mapping to `IntentGraph` nodes.
- `concept_extractor_v2.py`: Updated concept extractor for `EnhancedIntentGraph`, using JSON rules.
- `chroma_load.py`: Loads and indexes Word documents into Chroma vectorstore with chunking.
- `chroma_load_v2.py`: Indexes JSON rules into Chroma, creating documents from intent labels and metadata.
- `generate_intent_labels_from_rules.py`: Generates intent labels and enriched rule metadata from rule documents, producing `final_enriched_rules.json`.
- `app.py`: Streamlit frontend for user interaction, supporting clearance level input, dual chat modes, erase lockdown, and backend reset functionality.
- `test2.py`: Test script for `retrieval_v2.py`, running queries with varying clearance levels.

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
- **YAKE**: A lightweight keyword extraction tool used to identify significant terms in queries, complementing spaCy byesque capturing contextually relevant phrases.
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
  - Graph construction: O(N + E), where N is nodes (fixed at \~30) and E is edges (\~50), making it near-constant.
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
- **Key Methods**:
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
  - Graph construction: O(R \* (I + K + E)), where R is rules, I is intents, K is keywords per rule, and E is edges. For typical JSON with \~100 rules and \~10 keywords/rule, this is manageable.
  - Intent retrieval: O(K + E_k) for `find_related_intents`, where K is keywords and E_k is edges to intents (small per keyword).
  - Response flow: O(R_i \* log R_i) for sorting rules of an intent, where R_i is rules per intent (\~1-10).
- **Space Complexity**: O(N + E), where N includes intents, rules, and keywords (\~1000 nodes for large JSON), and E is edges (\~2000). Higher than `IntentGraph` but scalable.
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
| --- | --- | --- |
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

This section analyzes the `generate_intent_labels_from_rules.py` script, which processes a Word document (`rules.docx`) to generate enriched rule metadata, including intent labels, keywords, and response styles, outputting `final_enriched_rules.json` for use by `EnhancedIntentGraph` and `chroma_load_v2.py`.

### Json rule - summary

The `generate_intent_labels_from_rules.py` script extracts rules from `rules.docx`, identifying rule IDs, text, categories, and clearance levels. It enriches rules with keywords using YAKE, spaCy, and TF-IDF, merging them with weighted scores. Intent labels are assigned via semantic similarity to predefined labels using sentence embeddings. Style metadata, including greetings and response styles, is added based on agent levels. The enriched rules are saved as `final_enriched_rules.json` for dynamic intent mapping.

### Process Overview

The script extracts rules from a Word document, enriches them with keywords (using YAKE, spaCy, and TF-IDF), assigns intent labels via semantic similarity, and adds style metadata based on agent clearance levels. The output is a JSON file (`final_enriched_rules.json`) containing structured rule objects with metadata for dynamic intent mapping and response customization.

**Key Steps**:

1. **Rule Extraction (**`extract_rules_from_doc`**)**:

   - Reads `rules.docx` using `python-docx`.

   - Parses paragraphs with regex (`r"Rule\s+(\d+):\s*(.*)"`) to extract rule IDs and text.

   - Detects categories (e.g., "Basic Operational Queries") and clearance levels (e.g., "Level-3") using regex and predefined headers.

   - Identifies trigger types (e.g., "phrase_trigger" for conditional phrases, "level_and_keyword" for level-specific rules).

   - Returns a list of rule dictionaries with `rule_id`, `response`, `raw_text`, `agent_level`, `trigger_type`, `category`, and empty `keywords`.

   - **Performance**: O(P), where P is the number of paragraphs (typically hundreds). Regex matching and I/O are fast.

   - **Output Example**:

     ```python
     [
       {
         "rule_id": 1,
         "response": "Initiate secure comms with Quantum Hashing",
         "raw_text": "Rule 1: Initiate secure comms with Quantum Hashing",
         "agent_level": 3,
         "trigger_type": "level_and_keyword",
         "category": "Technology, Encryption & Intelligence Queries",
         "keywords": []
       },
       ...
     ]
     ```

2. **Keyword Extraction (**`parse_and_enrich_rules`**)**:

   - Extracts keywords using three methods:
     - **YAKE**: Extracts up to 15 n-grams (n≤3) with scores (1 - YAKE score).
     - **spaCy**: Identifies noun chunks with a default score of 1.0.
     - **TF-IDF**: Vectorizes all rule texts with `TfidfVectorizer` (300 features, n-grams 1-3) and selects top 15 terms per rule.
   - Merges keywords using `merge_and_rank_keywords`, applying weights (YAKE: 1.5, TF-IDF: 1.0, spaCy: 0.5) and ranking by average weighted score.
   - **Performance**:
     - YAKE: O(T) per rule, where T is text length.
     - spaCy: O(T) per rule for NLP processing.
     - TF-IDF: O(R \* T) for vectorization, where R is rules and T is unique terms (capped at 300).
     - Merging: O(K \* log K), where K is total unique keywords (\~50 per rule).
   - **Output**: List of `{"keyword": str, "score": float}` for each rule.

3. **Intent Classification (**`classify_intent`**)**:

   - Embeds rule text using `sentence-transformers/all-MiniLM-L6-v2`.
   - Computes cosine similarity with pre-embedded `INTENT_LABELS` (e.g., "Surveillance", "Encryption & Signals").
   - Assigns the highest-scoring intent label and confidence score.
   - **Performance**: O(T) for embedding, O(L) for cosine similarity, where L is labels (14). Constant-time for small L.
   - **Output**: `intent_label` (e.g., "Encryption & Signals"), `intent_confidence` (e.g., 0.85).

4. **Style Enrichment (**`enrich_style`**)**:

   - Maps `agent_level` to predefined `AGENT_STYLE_MAP` (levels 1-5) or defaults to a standard style for unspecified levels.
   - Assigns `greeting` (e.g., "Salute, Shadow Cadet.") and `response_style` (e.g., "Basic and instructional").
   - **Performance**: O(1) lookup per rule.
   - **Output**: `greeting` and `response_style` fields added to each rule.

5. **JSON Output (**`main`**)**:

   - Combines enriched rules into a JSON array and writes to `final_enriched_rules.json`.

   - Each rule includes `rule_id`, `intent_label`, `category`, `response_style`, `agent_level`, `response`, `raw_text`, `trigger_type`, `intent_confidence`, `keywords`, and `greeting`.

   - **Performance**: O(R) for serialization, I/O-bound for large rule sets.

   - **Output Example**:

     ```json
     [
       {
         "rule_id": 1,
         "intent_label": "Encryption & Signals",
         "category": "Technology, Encryption & Intelligence Queries",
         "response_style": "Analytical and multi-layered, providing strategic insights.",
         "agent_level": 3,
         "response": "Initiate secure comms with Quantum Hashing",
         "raw_text": "Rule 1: Initiate secure comms with Quantum Hashing",
         "trigger_type": "level_and_keyword",
         "intent_confidence": 0.85,
         "keywords": [
           {"keyword": "quantum hashing", "score": 0.9},
           {"keyword": "secure comms", "score": 0.8}
         ],
         "greeting": "Eyes open, Phantom."
       },
       ...
     ]
     ```

**Performance**:

- **Overall Time Complexity**: Dominated by TF-IDF vectorization, O(R \* T), and keyword merging, O(R \* K \* log K). For R=1000 rules, T=300 terms, K=50 keywords, processing is feasible in seconds on standard hardware.
- **Space Complexity**: O(R \* T) for TF-IDF matrix (sparse) and O(R \* K) for keyword storage. Memory usage is moderate but scales with rule count.
- **Limitations**:
  - Intent classification relies on predefined `INTENT_LABELS`, which may not cover all rule semantics.
  - Keyword merging weights are static, requiring tuning for optimal performance.
  - No clustering step, so intent labels are assigned per rule, potentially leading to redundancy.
- **Advantages**:
  - Combines YAKE, spaCy, and TF-IDF for robust keyword extraction, capturing diverse linguistic patterns.
  - Semantic intent classification ensures accurate mapping to operational contexts.
  - Direct JSON output simplifies integration with `EnhancedIntentGraph` and `chroma_load_v2.py`.

**Integration**:

- The JSON drives `EnhancedIntentGraph`, enabling dynamic intent mapping and response styling.
- `chroma_load_v2.py` uses the JSON to create Chroma documents, embedding rule content and metadata for vector search.
- The process ensures rules are both graph-navigable (via `EnhancedIntentGraph`) and searchable (via Chroma), enhancing retrieval precision.

**Recommendations**:

- Add a clustering step (e.g., K-means) to group similar rules and reduce intent redundancy.
- Dynamically adjust keyword weights based on rule context or performance metrics.
- Expand `INTENT_LABELS` to cover niche operational domains.
- Cache embeddings for incremental rule updates to improve startup time.

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
  - Responses are filtered by clearance level and intent, with LLM-enhanced answers when enabled.

- **Erase Functionality**:

  - Triggered by entering an incorrect cipher code in Secret Chat or attempting access with a clearance level &gt;13.
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

- **Clearance Levels**: Higher levels (e.g., 5) access sensitive protocols like "Project Eclipse." Levels &gt;13 trigger a lockdown.
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
