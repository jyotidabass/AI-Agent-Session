# 🤖 AI-Agent-Session

A hands-on collection of **5 practical notebooks** demonstrating how to build, orchestrate, and deploy AI agents using modern frameworks — CrewAI, SmolAgents, Gemini, and more. Each notebook is self-contained, beginner-friendly, and runnable directly on Google Colab.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyotidabass/AI-Agent-Session)

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Notebooks at a Glance](#-notebooks-at-a-glance)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Environment Variables](#-environment-variables)
- [Notebook Details](#-notebook-details)
  - [1. CrewAI Financial Analyst](#1-crewai-financial-analyst-agent)
  - [2. SmolAgent + Streamlit News Analyzer](#2-smolagent--streamlit-news-analyzer)
  - [3. Gemini + CrewAI Tech News Generator](#3-gemini--crewai-tech-news-generator)
  - [4. Gemini Vision Pro — UI to Code](#4-gemini-vision-pro--ui-to-code)
  - [5. SmolAgent SQL Agent (Gradio)](#5-smolagent-sql-agent-with-gradio)
- [API Keys Required](#-api-keys-required)
- [Running on Google Colab](#-running-on-google-colab)
- [Running Locally](#-running-locally)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🌟 Project Overview

This repository is a **practical AI agents tutorial series** built for developers, researchers, and AI enthusiasts who want to go beyond theory and build real, working agents. Each notebook focuses on a different framework or use-case:

| # | Notebook | Framework | Interface | Use Case |
|---|----------|-----------|-----------|----------|
| 1 | CrewAI Financial Analyst | CrewAI + OpenAI | Jupyter | Stock market analysis with multi-agent teams |
| 2 | SmolAgent News Analyzer | SmolAgents + Groq | Streamlit | Real-time news search & AI-powered analysis |
| 3 | Gemini + CrewAI Tech Writer | CrewAI + Gemini | Streamlit | Tech article research & generation |
| 4 | Gemini Vision — UI to Code | Gemini 1.5 Flash | Streamlit | Convert UI screenshots into HTML/CSS code |
| 5 | SmolAgent SQL Agent | SmolAgents + HuggingFace | Gradio | Natural language → SQL database queries |

---

## 📁 Repository Structure

```
AI-Agent-Session/
│
├── 1_CrewAI_Notebook.ipynb              # Financial analyst multi-agent system
├── 2_SmoleAgent_Streamlit_Notebook.ipynb  # News analysis agent with Streamlit UI
├── 3_Gemini_crewAI_Agents.ipynb         # Tech news generator (Gemini + CrewAI)
├── 4_GeminiVisionPro_Agent.ipynb        # Vision agent: screenshot → HTML code
├── 5_SmolAgent_sqlAgent.ipynb           # Natural language SQL agent with Gradio
│
├── requirements.txt                     # Core Python dependencies
├── .env.example                         # Template for environment variables
├── .gitignore
└── LICENSE
```

> **Note:** Each notebook is self-contained and generates its own app files (e.g., `app.py`, `sqlagent.py`, `database.py`) at runtime using `%%writefile` magic commands.

---

## 🚀 Notebooks at a Glance

```
Notebook 1 → CrewAI → Two agents (Financial Analyst + QC Reviewer) → Stock analysis report
Notebook 2 → SmolAgents → DuckDuckGo search + Groq LLM → Streamlit news dashboard
Notebook 3 → CrewAI + Gemini → Researcher + Writer agents → Blog article generator
Notebook 4 → Gemini Vision → Upload UI screenshot → Get production HTML/CSS code
Notebook 5 → SmolAgents + SQLAlchemy → Natural language → SQL on a receipts database
```

---

## ✅ Prerequisites

- Python `>= 3.10` and `< 3.13`
- pip or conda
- A Google Colab account (recommended) **or** a local Python environment
- API keys (see [API Keys Required](#-api-keys-required))

---

## 🛠 Installation

### Option A: Google Colab (Recommended)
Click the **"Open in Colab"** badge at the top of any notebook. Each notebook handles its own dependency installation via `!pip install` cells — just run them in order.

### Option B: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/jyotidabass/AI-Agent-Session.git
cd AI-Agent-Session

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# OR with conda:
# conda create -n ai-agents python=3.12
# conda activate ai-agents

# 3. Install dependencies
pip install -r requirements.txt

# 4. For notebook-specific dependencies, also run:
pip install smolagents streamlit groq duckduckgo-search gradio sqlalchemy \
            google-generativeai langchain-google-genai pillow beautifulsoup4 \
            newspaper3k nltk
```

---

## 🔐 Environment Variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY="your_openai_api_key"
SERPER_API_KEY="your_serper_api_key"
GOOGLE_API_KEY="your_google_api_key"
GROQ_API="your_groq_api_key"
```

> For Colab users: set keys directly in notebook cells using `os.environ["KEY"] = "value"` or via Colab Secrets.

---

## 📓 Notebook Details

---

### 1. CrewAI Financial Analyst Agent

**File:** `1_CrewAI_Notebook.ipynb`
**Framework:** CrewAI + OpenAI GPT-3.5-Turbo
**Interface:** Jupyter / Colab

#### What it does
Builds a two-agent financial analysis system that researches a stock (e.g., Tesla) and produces a professional investment report with a Buy/Hold/Sell recommendation.

#### Agents
| Agent | Role | Responsibility |
|-------|------|----------------|
| `financial_analyst` | Senior Financial Analyst | Scrapes NASDAQ, analyzes earnings, stock trends, and risk |
| `investment_qc` | Investment Strategy Reviewer | Reviews and validates the analyst's report for accuracy |

#### Key Components
- **Tool:** `ScrapeWebsiteTool` pointing to `nasdaq.com/market-activity/stocks`
- **Tasks:** `stock_analysis_task` → `investment_review_task` (sequential)
- **Crew Memory:** Enabled (`memory=True`) so agents share context
- **Delegation:** `investment_qc` can delegate back to `financial_analyst`

#### How to Run

```python
# Set your API key
import os
os.environ["OPENAI_API_KEY"] = "your_key"
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

# Run the crew
inputs = {"company_name": "Tesla, Inc."}
result = finance_crew.kickoff(inputs=inputs)
```

#### Output
A detailed markdown report with financial metrics, stock performance trends, and investment recommendation.

---

### 2. SmolAgent + Streamlit News Analyzer

**File:** `2_SmoleAgent_Streamlit_Notebook.ipynb`
**Framework:** SmolAgents + Groq (DeepSeek-R1) + DuckDuckGo
**Interface:** Streamlit (exposed via localtunnel on Colab)

#### What it does
Deploys a Streamlit web app that takes a news topic as input, searches the web via DuckDuckGo, and uses Groq's DeepSeek-R1 LLM to generate a structured, multi-perspective news analysis.

#### Architecture

```
User Input (topic)
      ↓
DuckDuckGoSearch (fetches latest news articles)
      ↓
create_analysis_prompt() (structures the analysis request)
      ↓
GroqLLM — deepseek-r1-distill-llama-70b (generates analysis)
      ↓
Streamlit UI (displays formatted results + activity log)
```

#### Generated Analysis Sections
1. Key Points Summary
2. Stakeholder Analysis
3. Impact Assessment
4. Multiple Perspectives
5. Fact Check & Reliability

#### Key Classes

```python
class DuckDuckGoSearch:
    # Searches DuckDuckGo News API (worldwide, safesearch on)
    # Returns: formatted string with title, date, source, summary, URL

class GroqLLM:
    # Wraps Groq API with model: deepseek-r1-distill-llama-70b
    # Params: temperature=0.7, max_tokens=1024
```

#### How to Run (Colab)

```bash
# Run all cells in order, then:
!streamlit run app.py & npx localtunnel --port 8501
# Visit the tunnel URL and enter the password from:
!curl https://loca.lt/mytunnelpassword
```

#### UI Features
- Search depth slider (3–10 results)
- Analysis type selector (Comprehensive / Quick Summary / Technical / Simplified)
- Expandable agent activity log for full transparency

---

### 3. Gemini + CrewAI Tech News Generator

**File:** `3_Gemini_crewAI_Agents.ipynb`
**Framework:** CrewAI + Google Gemini 1.5 Flash + SerperDev
**Interface:** Streamlit (via localtunnel on Colab)

#### What it does
A Streamlit app where you enter a tech topic and two AI agents — a researcher and a writer — collaborate to produce a polished, publication-ready blog article saved as `new-blog-post.md`.

#### Agents
| Agent | Role | Goal |
|-------|------|------|
| `news_researcher` | Senior Researcher | Uncover groundbreaking technologies in the topic; can delegate |
| `news_writer` | Writer | Write an engaging, accessible 4-paragraph markdown article |

Both agents use **Gemini 1.5 Flash** as the LLM and **SerperDevTool** for live internet search.

#### Workflow

```
User Input: topic (e.g., "AI in healthcare")
      ↓
research_task → 3-paragraph research report on trends, pros/cons, opportunities, risks
      ↓
write_task → 4-paragraph markdown article (async_execution=False)
      ↓
Output saved to: new-blog-post.md
      ↓
Displayed in Streamlit UI
```

#### LLM Configuration

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)
```

#### How to Run (Colab)

```bash
# Install dependencies
!pip install -r requirements.txt

# Set .env with GOOGLE_API_KEY and SERPER_API_KEY
# Then run:
!streamlit run Gemini-crewAI-Agents.py & npx localtunnel --port 8501
```

---

### 4. Gemini Vision Pro — UI to Code

**File:** `4_GeminiVisionPro_Agent.ipynb`
**Model:** Gemini 1.5 Flash Latest (Vision)
**Interface:** Streamlit (via localtunnel on Colab)

#### What it does
Upload any UI screenshot (JPG/PNG) and the agent will:
1. Describe all UI elements with bounding boxes and colors
2. Refine the description by comparing with the image
3. Generate responsive HTML/CSS code matching the original UI
4. Refine the HTML for accuracy and mobile-first design
5. Let you download the final `index.html`

#### Processing Pipeline

```
Uploaded Image
      ↓
Step 1: describe_ui_prompt → detailed element description with bounding boxes
      ↓
Step 2: refine_description_prompt → cross-reference image vs description
      ↓
Step 3: html_generation_prompt → produce HTML + inline CSS (responsive)
      ↓
Step 4: refine_html_prompt → validate and improve HTML fidelity
      ↓
Output: index.html (displayed + downloadable)
```

#### Model Configuration

```python
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 50000,
}
MODEL_NAME = "gemini-1.5-flash-latest"
framework = "Regular CSS use flex grid etc"  # Customizable
```

All safety settings are set to `BLOCK_NONE` to allow full UI code generation.

#### How to Run (Colab)

```bash
!pip install streamlit langchain_google_genai crewai python-dotenv pillow google-generativeai
# Set GOOGLE_API_KEY in .env, then:
!streamlit run app.py & npx localtunnel --port 8501
```

---

### 5. SmolAgent SQL Agent with Gradio

**File:** `5_SmolAgent_sqlAgent.ipynb`
**Framework:** SmolAgents + HuggingFace (Qwen2.5-Coder-32B) + SQLAlchemy + Gradio
**Interface:** Gradio (shareable link or port 7860/7861)

#### What it does
A natural language interface to a SQLite database. Type plain English questions and the agent generates and executes the correct SQL query, returning results instantly.

#### Architecture

```
User's Natural Language Question
      ↓
CodeAgent (Qwen2.5-Coder-32B-Instruct via HfApiModel)
  - Receives schema info + user query
  - Generates SQL SELECT statement
      ↓
sql_engine tool (SQLAlchemy → SQLite)
  - Executes the query on database.db
  - Returns formatted results
      ↓
Gradio UI (displays result + live receipts table)
```

#### Database Schema (`database.db`)

```
Table: receipts
├── receipt_id    INTEGER  (primary key)
├── customer_name VARCHAR(16)
├── price         FLOAT
└── tip           FLOAT
```

Pre-populated with 40 sample customer receipt records.

#### Example Queries

```
"What is the total price of all receipts?"
"Who paid the highest tip?"
"Show all customers who spent more than $50"
"What is the average tip amount?"
```

#### Key Files Generated at Runtime

| File | Purpose |
|------|---------|
| `database.py` | Creates SQLite DB, defines schema, inserts 40 sample rows |
| `sqlagent.py` | Basic Gradio UI with natural language → SQL agent |
| `new_sqlagent.py` | Enhanced Gradio UI with custom styling and better error handling |

#### How to Run

```bash
# Install dependencies
!pip install gradio smolagents python-dotenv sqlalchemy

# Run the agent
!python3 new_sqlagent.py
# OR in Colab:
!python3 new_sqlagent.py  # Launches with share=True (public link)
```

---

## 🔑 API Keys Required

| Key | Used In | How to Get |
|-----|---------|------------|
| `OPENAI_API_KEY` | Notebook 1 | [platform.openai.com](https://platform.openai.com) |
| `SERPER_API_KEY` | Notebooks 1, 3 | [serper.dev](https://serper.dev) |
| `GOOGLE_API_KEY` | Notebooks 3, 4 | [aistudio.google.com](https://aistudio.google.com) |
| `GROQ_API` | Notebook 2 | [console.groq.com](https://console.groq.com) |
| HuggingFace Token | Notebook 5 | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

> ⚠️ Never commit real API keys to GitHub. Always use `.env` files (already in `.gitignore`) or Colab Secrets.

---

## ☁️ Running on Google Colab

Each notebook has a **"Open in Colab"** button at the top. For notebooks with Streamlit/Gradio UIs:

1. Run all cells from top to bottom
2. The last cell starts the app and creates a tunnel URL
3. For localtunnel-based apps, get your tunnel password:
   ```bash
   !curl https://loca.lt/mytunnelpassword
   ```
4. Visit the tunnel URL → enter the password → use the app

---

## 💻 Running Locally

```bash
# Clone and set up (see Installation above)

# Notebook 1 — run in Jupyter
jupyter notebook 1_CrewAI_Notebook.ipynb

# Notebook 2 — Streamlit app
# Extract app.py from the notebook first (run %%writefile cells), then:
streamlit run app.py

# Notebook 3 — Streamlit app
streamlit run Gemini-crewAI-Agents.py

# Notebook 4 — Streamlit app
streamlit run app.py

# Notebook 5 — Gradio app
python3 new_sqlagent.py
# Visit http://localhost:7861
```

---

## 🧰 Technologies Used

| Technology | Purpose |
|------------|---------|
| [CrewAI](https://docs.crewai.com) | Multi-agent orchestration framework |
| [SmolAgents](https://huggingface.co/docs/smolagents) | Lightweight HuggingFace agent framework |
| [LangChain Google GenAI](https://python.langchain.com/docs/integrations/llms/google_ai) | Gemini LLM integration for LangChain/CrewAI |
| [Google Gemini 1.5 Flash](https://ai.google.dev) | Multimodal LLM (text + vision) |
| [Groq + DeepSeek-R1](https://console.groq.com) | Ultra-fast LLM inference |
| [OpenAI GPT-3.5-Turbo](https://platform.openai.com) | LLM for CrewAI agents |
| [Qwen2.5-Coder-32B](https://huggingface.co/Qwen) | Code-specialized LLM for SQL generation |
| [Streamlit](https://streamlit.io) | Rapid web UI framework |
| [Gradio](https://gradio.app) | ML demo UI framework |
| [SQLAlchemy](https://sqlalchemy.org) | SQL toolkit and ORM |
| [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) | Privacy-first web search |
| [SerperDev](https://serper.dev) | Google Search API |
| [localtunnel](https://theboroer.github.io/localtunnel-www/) | Expose local ports publicly from Colab |

---

## 📦 Core Dependencies (`requirements.txt`)

```
requests
ipykernel
python-dotenv
openai
pydantic
langchain-google-genai
crewai
crewai-tools
```

Additional per-notebook installs are handled inside each notebook.

---

## 👩‍💻 Author

**Jyoti Dabass, Ph.D**
Researcher in NLP, Computer Vision & AI

- GitHub: [@jyotidabass](https://github.com/jyotidabass)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a PR or issue.

---

## ⭐ Star this repo if you found it helpful!
