
# 🏡 OBC Project Assistant

The OBC Project Assistant is a conversational AI powered by **LangGraph**, **LangChain**, and **Streamlit** that helps homeowners understand construction regulations, permits, and legal guidelines.

It classifies queries, performs live searches, summarizes legal texts, and generates a detailed 7-phase roadmap for construction projects—all through a chat interface.

---

## ✨ Features

- 💬 Conversational chat interface with Streamlit
- 🧠 Classifies input as legal vs. general
- 📍 Extracts location and project type from queries
- 🔍 Searches legal documents via Tavily API
- 📑 Summarizes search results using LLMs
- 🛠️ Builds a full project roadmap (7 phases)
- ✅ Uses LangGraph for modular agent workflows

---

## 📁 Project Structure

```
project-root/
├── app.py                # 🔹 Streamlit UI & execution entrypoint
├── .env                  # 🔹 API keys (GROQ_API_KEY, TAVILY_API_KEY)
├── checkpoints.db        # 🔹 SQLite checkpoint file (auto-generated)
├── requirements.txt      # 🔹 Dependencies
├── README.md             # 🔹 Project documentation
└── src/                  # 🔸 All core source modules
    ├── __init__.py
    ├── config.py         # Load .env & environment setup
    ├── models.py         # TypedDict and Pydantic models
    ├── agents.py         # LLM agent logic (classify, parse, search, summarize)
    └── workflow.py       # LangGraph graph setup and node wiring
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HannanBinTahir/legal-bot.git
cd legal-bot.git
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Make sure to run this from the root directory where `app.py` is located.

---

## 🔄 Imports & Module Pathing

In `app.py`, use:

```python
from src.models import AgentState
from src.agents import classify_query, handle_general_query, ...
from src.workflow import app as langgraph_app
```

Ensure `src/` has an `__init__.py` file and you're working from the root directory.

---

## 🧠 Technologies Used

| Tool               | Purpose                           |
|--------------------|------------------------------------|
| `Streamlit`        | Web-based chatbot interface        |
| `LangGraph`        | Modular agent workflows            |
| `LangChain`        | Prompt handling + LLM chaining     |
| `ChatGroq`         | Backend LLM (LLaMA 3.1 8B)         |
| `TavilySearch`     | Search engine for legal documents  |
| `Pydantic`         | Data validation and parsing        |
| `dotenv`           | Environment variable loading       |
| `SQLite`           | Session state persistence          |

---

## 📦 Example Use Case

> “Do I need a permit to replace windows in Oakland, California?”

➡️ Assistant will:
1. Classify as a legal query  
2. Extract "window replacement", "Oakland", "California"  
3. Search legal sources using Tavily  
4. Summarize results and cite URLs  
5. Generate a full roadmap (Phase 1–7)  

---

## ⚠️ Disclaimer

> **Owner Builder Concepts (OBC) provides educational and informational content only.**

This app does **not provide legal advice**. Always consult your local municipality, licensed contractors, or legal advisors before beginning construction projects.

Use of this assistant is **at your own risk**.


