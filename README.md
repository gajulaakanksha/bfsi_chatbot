# BFSI Call Center AI Assistant 🤖

A local, privacy-focused AI assistant designed for **Banking, Financial Services, and Insurance (BFSI)** queries. It uses a **3-Tier Response Pipeline** to deliver accurate, safe, and context-aware answers using a fine-tuned **TinyLlama-1.1B** model.

![Status](https://img.shields.io/badge/Status-Operational-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![GPU](<https://img.shields.io/badge/GPU-Required%20(RTX%203070%2B)-orange>)

## 🌟 Key Features

The system uses a hybrid pipeline to handle queries:

1.  **Tier 1: Dataset Matcher** ⚡
    - Instantly answers common queries (e.g., "How to check balance?") using similarity search against a curated dataset.
    - **Zero hallucinations.**

2.  **Tier 2: Fine-Tuned SLM** 🧠
    - Handles general banking tasks (e.g., "Draft an email to close my account") using **TinyLlama-1.1B-Chat** fine-tuned on BFSI instructions.
    - Optimized for polite, compliant interaction.

3.  **Tier 3: RAG Augmented** 📚
    - Retrieves specific facts (e.g., "What is the UPI Lite limit?") from a local Knowledge Base (ChromaDB).
    - Grounded generation prevents inventing numbers or policies.

**🛡️ Guardrails:** built-in safety filters block unsafe or out-of-domain queries (e.g., "How to make a bomb?", "Who won the cricket match?").

---

## 🛠️ Installation

### Prerequisites

- Windows 10/11
- NVIDIA GPU with 8GB+ VRAM (Recommended)
- Conda

### Steps

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/GaneshDoosa/bfsi_chatbot.git
    cd bfsi_chatbot
    ```

2.  **Create Conda Environment**

    ```bash
    conda create -n bfsi_chatbot python=3.10
    conda activate bfsi_chatbot
    ```

3.  **Install Dependencies**

    ```bash
    # Install PyTorch with CUDA support first
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    # Install remaining packages
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    Copy `.env.example` to `.env` and add your LangSmith API key (optional, for tracing).
    ```bash
    cp .env.example .env
    ```

---

## 🚀 Usage

### 1. Initialize Data

Build the vector store and basic dataset (only needed once):

```bash
python scripts/build_vectorstore.py
```

### 2. Run the App

Launch the Streamlit UI:

```bash
streamlit run app.py
```

Access the interface at `http://localhost:8501`.

### 3. Training (Optional)

To re-train the model on new data:

```bash
python scripts/train.py
```

---

## 📂 Project Structure

```
bfsi_chatbot/
├── data/
│   ├── alpaca_bfsi_dataset.json   # Fine-tuning dataset
│   └── knowledge_base/            # RAG source documents
├── models/                        # Saved Adapters/Checkpoints
├── src/
│   ├── dataset_matcher.py         # Tier 1 Logic
│   ├── slm_engine.py              # Tier 2 Logic (TinyLlama)
│   ├── rag_engine.py              # Tier 3 Logic (ChromaDB)
│   ├── pipeline.py                # LangGraph Orchestrator
│   └── guardrails.py              # Safety Layer
├── app.py                         # Streamlit UI
└── requirements.txt
```

## 📝 Documentation

- [Design Document](design_document.md) - Architecture & Diagrams.
- [Test Scenarios](test_scenarios.md) - Q&A pairs for verification.
- [Implementation Plan](implementation_plan.md) - Build functionality.

## 📄 License

MIT License.
