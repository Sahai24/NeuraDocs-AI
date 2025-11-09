# 🧠 NeuraDocs AI

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-007FFF?logo=microsoftazure)
![LangChain](https://img.shields.io/badge/Framework-LangChain-orange?logo=openai)
![License](https://img.shields.io/badge/License-MIT-green)

> ⚙️ **NeuraDocs AI** is an intelligent document understanding and conversational AI framework powered by **Azure OpenAI**, **LangChain**, and **RAG (Retrieval-Augmented Generation)**.  
> It enables seamless document querying, embeddings, and chat-driven knowledge retrieval.

---

## 🧩 Key Features

| Feature | Description | Tools |
|----------|--------------|-------|
| 📂 **Document Embedding** | Converts PDFs into searchable vector representations. | PyMuPDF, LangChain |
| 🧠 **Retrieval-Augmented Generation (RAG)** | Retrieves relevant context before generating LLM responses. | Milvus / FAISS |
| 🤖 **Chat API** | Multi-chatbot orchestration for contextual dialogue. | Flask / FastAPI |
| ☁️ **Azure OpenAI Integration** | Uses Azure-hosted GPT endpoints for scalable chat intelligence. | Azure OpenAI |
| 🧮 **Sentiment & NLP** | Extendable to NLTK, SpaCy, and summarization modules. | SpaCy, NLTK |
| 🔐 **Environment Management** | Uses `.env` files for secure key management. | python-dotenv |

---

## 🧠 Project Architecture

```mermaid
flowchart TD
    A[📄 PDF Upload] --> B[🧩 Document Embedding]
    B --> C[🗃️ Vector Database (Milvus/FAISS)]
    C --> D[🔍 Context Retrieval]
    D --> E[💬 LLM (Azure OpenAI)]
    E --> F[🧾 Intelligent Response to User]
```

---

## 🏗️ Project Structure

```
python/
│
├── app.py                      # Main application runner
├── chatapi.py                  # Chat API service
├── azurechatbotapi.py          # Azure OpenAI integration
├── multichat.py                # Multi-conversation handler
├── CleanAPI.py                 # Clean API abstraction layer
├── requirements.txt
│
├── Document_processing_api/    # Document processing module
│   ├── app.py
│   ├── embedding.py
│   ├── process_pdf.py
│   ├── vector_db.py
│   ├── config.py
│   ├── processed/FAQ.pdf
│
└── RAG_processing/             # Retrieval-Augmented Generation module
    ├── app.py
    ├── config.py
    ├── processed/FAQ.pdf
```

---

## 🛠️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/NeuraDocs-AI.git
cd NeuraDocs-AI/python
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment variables
Create a `.env` file with:
```
OPENAI_API_KEY=your_azure_key
MILVUS_HOST=localhost
```

### 4️⃣ Run the app
```bash
python app.py
```

---

## 🧪 Example Workflow

1️⃣ Upload or select a PDF file  
2️⃣ Generate embeddings and store them in the vector database  
3️⃣ Ask a question related to your document  
4️⃣ Receive a contextual, AI-generated response  

---

## 📸 Visual Overview

| Module | Function | Visual Cue |
|---------|-----------|------------|
| 🧠 Chat Engine | Azure-based intelligent chat system | 🤖 |
| 📄 Document API | Embedding + Retrieval pipeline | 📚 |
| 🔍 RAG Processor | Contextual understanding system | 💡 |
| ⚙️ Configuration | Environment & API setup | ⚙️ |

---

## 🧱 Tech Stack

| Layer | Technology |
|-------|-------------|
| Backend | Python 3.11, FastAPI |
| AI Engine | Azure OpenAI, LangChain |
| Vector Store | FAISS / Milvus |
| Data | PDF, Text documents |
| Utility | dotenv, PyMuPDF, pandas |

---

## 🚀 Future Enhancements

- [ ] Streamlit-based user dashboard  
- [ ] Integration with multiple LLM providers (Claude, Gemini)  
- [ ] Real-time document chat  
- [ ] Docker containerization  
- [ ] GitHub Actions CI/CD  

---

## 👨‍💻 Author
**Developed by:** Avanish Sahai  
🎓 *Internship Project — Document AI & Conversational Frameworks*  

---

## 📜 License
This project is licensed under the **MIT License**.  

---

⭐ **NeuraDocs AI** — *Where Knowledge Meets Intelligence.*
