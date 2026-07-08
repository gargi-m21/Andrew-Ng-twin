# Digital Twin of Professor Andrew Ng 🎓

An intelligent, agentic conversational digital twin of Professor Andrew Ng. This system emulates his technical reasoning, structured pedagogy, and characteristic encouraging demeanor while grounding all academic domain assertions in a localized retrieval database.

### 🎥 [Watch the Demo Video](https://drive.google.com/file/d/1RywIkP4iOcAKx_sq3jbnjBgUSELtTrZJ/view?usp=drive_link)
###  [Documentation](https://docs.google.com/document/d/1vUOP4exaxaAglBljw4db0HVTYRTIwt54sV_xImScU4Q/edit?usp=drive_link)
---

## 🧠 System Architecture & Pillars

The project is built around the three core pillars defined in the assignment specification:

### 1. Agentic Core

Powered by **Gemini 2.5 Flash** running a low-temperature (0.3) inference loop via **LangChain Expression Language (LCEL)**.

The agent naturally adopts:

- Andrew Ng's encouraging teaching style
- Step-by-step explanations
- Data-centric AI philosophy
- Educational reasoning patterns

---

### 2. Throttled RAG Pipeline

The system extends model knowledge through a custom Andrew Ng knowledge base consisting of:

- Coursera Machine Learning Specialization transcripts
- BUILD 2024 Keynote on Agentic AI
- DeepLearning.AI blogs
- Recent Andrew AI essays and manifestos
- Additional public educational content

#### Retrieval Pipeline

- 45+ source documents
- 568 semantic chunks
- Gemini Embedding Model (`gemini-embedding-2-preview`)
- Persistent ChromaDB vector store

#### Rate-Limit Aware Ingestion

To remain compatible with the Google AI Studio Free Tier:

- 20 chunks processed per batch
- 15-second cooldown between batches
- Safe handling of 100 RPM limits
- Persistent local vector database

---

### 3. Dual-Layer Memory

#### Short-Term Memory

Maintains conversational continuity through:

- Rolling chat history
- Multi-turn context awareness
- Session-level memory buffer

#### Long-Term Memory

A lightweight reflection engine runs after every conversation turn and extracts useful student information such as:

- Name
- Academic background
- Current projects
- Learning goals
- Technical interests

The extracted profile is stored in:

```text
user_profiles.json
```

allowing the twin to remember users across sessions.

---

## 📁 Project Directory Structure

```text
andrew-ng-twin/
│
├── data/                         # Grounding Corpus & Long-Term Memory Cache
│   ├── user_profiles.json        # Persistent student profile facts across sessions
│   ├── blog1.txt   
│   ├── stanford1.txt     
│   ├── ted.txt    
│   └── [Remaining 40+ lecture text sources...]
│
├── chromadb_store/               # Local Persistent Vector Database (Chroma)
│
├── src/                          # Production Backend Engine Modules
│   ├── __init__.py
│   ├── agent_engine.py           # Core LangChain + Gemini Orchestration
│   ├── config.py                 # System Prompt Templates & Global Envs
│   ├── ingestion.py              # Throttled Ingestion & Chunking Setup
│   └── memory_manager.py         # Dual-Layer Memory Interface
│
├── app.py                        # Personalized Interactive Streamlit UI
├── requirements.txt              # Standardized Dependencies File
├── .gitignore                    # Local Environmental Safety Guards
├── test_agent.py                 # CLI Local Integration Validation Script
└── README.md                     # Technical Documentation Hub
---

```
## 🛠️ Local Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/gargi-m21/Andrew-Ng-twin.git
cd andrew-ng-twin
```

---

### 2. Install Dependencies

Ensure Python (3.10+) and Anaconda are installed.

```bash
pip install langchain \
langchain-community \
langchain-google-genai \
langchain-text-splitters \
chromadb \
python-dotenv \
streamlit
```

---

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

> `.env` and `chromadb_store/` should be included in `.gitignore`.

---

### 4. Build the Vector Database

Run the ingestion pipeline:

```bash
python -m src.ingestion
```

This process:

1. Loads source documents
2. Splits text into semantic chunks
3. Generates embeddings
4. Stores vectors in ChromaDB

---

### 5. Launch the Application

```bash
streamlit run app.py
```

The application will open locally in your browser.

---

## 🔄 End-to-End Workflow Architecture

```text
       [ User Interaction ]
                │
                ▼
        +───────────────+
        │  User Query   │
        +───────┬───────+
                │
                ├──(1) Send query text to backend
                │
                ▼
+───────────────────────────────────+      +───────────────────────────────+
│   Context Collection Engine       │◄─(2)─┤ Short-Term Session History    │
│  (Aggregates input parameters)    │      │ (Rolling 10-message list cache)│
+───────────────┬───────────────────+      +───────────────────────────────+
                │
         (3) Parallel RAG Lookup
                │
                ▼
+───────────────────────────────────+      +───────────────────────────────+
│ Vector Database Retrieval Module  │◄─(4)─┤ Persistent Local Vector Store │
│  (Computes cosine similarity)    │      │ (ChromaDB / 568 text chunks)  │
+───────────────┬───────────────────+      +───────────────────────────────+
                │
        (5) Inject ground metadata & profile state
                │
                ▼
+──────────────────────────────────────────────────────────────────────────+
│                      LangChain Expression Language (LCEL)                │
│                                                                          │
│  [Dynamic System Prompt Template]                                        │
│   ├── Core Andrew Ng Persona Directives ("AI is the new electricity...") │
│   ├── Long-Term Student Profile Summary (From data/user_profiles.json)   │
│   ├── Vector Grounding Context (Lecture Transcripts & Keynotes)          │
│   └── Multi-turn Dialogue History Context                                │
│                                                                          │
│                                   │                                      │
│                                   ▼                                      │
│                      [ Gemini 2.5 Flash LLM Core ]                       │
│                        (Low-Temperature: 0.3)                            │
+───────────────────────────────────┬──────────────────────────────────────+
                │
         (6) Generate payload
                │
                ├──────────────────────────┐
                ▼                          ▼
    +───────────────────────+   +──────────────────────────────────────+
    │ UI Rendering Engine   │   │ Async Background Reflection Engine   │
    │ (Streamlit Frontend)  │   │   (Evaluates latest dialogue turn)   │
    +───────────────────────+   +──────────────────┬───────────────────+
                │                                  │
         (7) Render chat text               (8) Extract new student traits
                │                                  │
                ▼                                  ▼
    +───────────────────────+   +──────────────────────────────────────+
    │  Persona Response    │   │  data/user_profiles.json Persistence  │
    │   Displayed Live      │   │   (Locks in metrics for next turn)   │
    +───────────────────────+   +──────────────────────────────────────+

```
---

## 🖥️ Demonstrated Features

### ✅ Persona Consistency

The digital twin consistently reproduces:

- Andrew Ng's educational tone
- Data-centric AI mindset
- Encouraging learning philosophy

Example expressions include:

> "AI is the new electricity."

> "Don't worry if you didn't get the math on the first pass."

---

### ✅ Retrieval-Augmented Generation

The agent grounds answers using indexed Andrew Ng content instead of relying solely on model parameters.

Benefits:

- Reduced hallucinations
- Improved factual consistency
- Explainable source-backed responses

---

### ✅ Technical Depth

The system successfully discusses advanced AI topics including:

- Convolutional Neural Networks (CNNs)
- Vision Transformers (ViTs)
- Retrieval-Augmented Generation (RAG)
- Agentic AI Systems
- Grad-CAM Explainability
- Medical AI Applications

Example benchmark discussion:

- CNN vs ViT for chest X-ray classification
- Grad-CAM visualization for pneumonia detection
- Explainability in healthcare AI

---

### ✅ Persistent Memory

The assistant remembers information across sessions such as:

```text
Name: Gargi
Major: Engineering Student
Project: AIMS-DTU Summer Project
Current Focus: Agentic AI + RAG Systems
```

This creates a personalized educational experience.

---

### ⭐ Bonus Feature: Memory Dashboard

The Streamlit sidebar contains a live **Twin Memory Dashboard** showing:

- Stored user profile
- Learned preferences
- Current project context
- Memory updates after each interaction

This makes the memory system transparent and easy to evaluate.

---

## 🚀 Future Improvements

- Source citation display
- Multi-agent reasoning workflow
- Voice cloning and speech synthesis
- Hybrid search (BM25 + Vector Retrieval)
- Knowledge graph augmentation
- Automated dataset refresh pipeline

---

## 📜 Tech Stack

| Layer | Technology |
|---------|-----------|
| LLM | Gemini 2.5 Flash |
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Gemini Embeddings |
| Frontend | Streamlit |
| Memory Store | JSON Persistence |
| Language | Python |

---

## 📄 License

This project was developed as part of the **AIMS DTU Summer Project 2026 Evaluation Sandbox** and is intended for educational and research purposes.
