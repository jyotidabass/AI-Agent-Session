# 🤖 AI Agent Session – Building Practical AI Agents with Python

A comprehensive, hands-on repository demonstrating **how to build, orchestrate, and deploy AI agents** from scratch using modern Python frameworks. Structured as a **progressive learning path** — from simple single agents all the way to advanced multi-agent systems — with both runnable Python scripts and interactive Jupyter notebooks.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyotidabass/AI-Agent-Session)

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [What You Will Learn](#-what-you-will-learn)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Environment Variables](#-environment-variables)
- [Module 1 – Introduction to AI Agents](#-module-1--introduction-to-ai-agents)
- [Module 2 – AI Workflow Patterns](#-module-2--ai-workflow-patterns)
- [Module 3 – CrewAI Multi-Agent System](#-module-3--crewai-multi-agent-system)
- [Module 4 – SmolAgents](#-module-4--smolagents)
- [Jupyter Notebooks](#-jupyter-notebooks)
  - [Notebook 1: CrewAI Financial Analyst](#notebook-1-crewai-financial-analyst)
  - [Notebook 2: SmolAgent + Streamlit News Analyzer](#notebook-2-smolagent--streamlit-news-analyzer)
  - [Notebook 3: Gemini + CrewAI Tech News Generator](#notebook-3-gemini--crewai-tech-news-generator)
  - [Notebook 4: Gemini Vision Pro – UI to Code](#notebook-4-gemini-vision-pro--ui-to-code)
  - [Notebook 5: SmolAgent SQL Agent](#notebook-5-smolagent-sql-agent)
- [Key Concepts Demonstrated](#-key-concepts-demonstrated)
- [API Keys Required](#-api-keys-required)
- [Technologies Used](#-technologies-used)
- [Use Cases](#-use-cases)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🌟 Project Overview

This repository is built for developers, researchers, and AI enthusiasts who want to go beyond theory and build **real, working AI agents**. It covers the full spectrum of modern agent patterns:

- **Prompt engineering** — crafting effective inputs for LLM agents
- **Tool usage** — giving agents the ability to call external functions and APIs
- **Retrieval-Augmented Generation (RAG)** — agents that query private knowledge bases
- **Multi-agent workflows** — multiple specialized agents collaborating on tasks
- **Routing and orchestration** — intelligent task delegation and pipeline management
- **CrewAI agents** — role-based collaborative agent teams
- **Data analysis agents** — natural language queries over structured data
- **News agents** — real-time article fetching and AI summarization

The goal is to help developers **understand, build, and experiment with different AI agent architectures step-by-step**, progressing from simple single agents to production-grade multi-agent systems.

---

## 📁 Project Structure

```
AI-Agent-Session
│
├── 1-Introduction_Agents/          # Module 1: Core agent building blocks
│   ├── 1-basic.py                  # Simplest LLM agent
│   ├── 2-structured.py             # Structured/typed output generation
│   ├── 3-tools.py                  # Agent with callable tools
│   ├── 4-retrieval.py              # RAG agent using a knowledge base
│   └── know-db.json                # JSON knowledge base for retrieval
│
├── 2-workflows/                    # Module 2: Agent workflow patterns
│   ├── 1-prompt-chaining.py        # Sequential prompt chaining
│   ├── 2-routing.py                # Intelligent request routing
│   ├── 3-parallization.py          # Parallel task execution
│   └── 4-orchestrator.py           # Full multi-component orchestration
│
├── 3-crewAIAgent/                  # Module 3: Multi-agent with CrewAI
│   └── Financeapp.py               # Role-based Finance AI assistant
│
├── 4-smolagent/                    # Module 4: SmolAgents applications
│   ├── DataanalystAgent/
│   │   ├── app.py                  # Streamlit data analyst UI
│   │   └── streaming.py            # Streaming agent responses
│   └── NewsAgent.py                # AI-powered news agent
│
├── 1_CrewAI_Notebook.ipynb         # Notebook: CrewAI financial analyst (OpenAI)
├── 2_SmoleAgent_Streamlit_Notebook.ipynb  # Notebook: News analyzer (Groq + DuckDuckGo)
├── 3_Gemini_crewAI_Agents.ipynb    # Notebook: Gemini + CrewAI tech article writer
├── 4_GeminiVisionPro_Agent.ipynb   # Notebook: Screenshot → HTML/CSS code
├── 5_SmolAgent_sqlAgent.ipynb      # Notebook: Natural language → SQL (Gradio)
│
├── requirements.txt                # Core Python dependencies
├── .env.example                    # Template for environment variables
├── .gitignore
└── LICENSE
```

---

## 🎓 What You Will Learn

### Basic AI Agent Concepts
- How LLM agents process and respond to prompts
- Structured output generation (JSON, Pydantic models)
- Tool calling — letting agents invoke Python functions
- Knowledge retrieval using RAG with a local JSON knowledge base

### AI Workflow Patterns
- **Prompt Chaining** — breaking complex tasks into sequential LLM calls
- **Intelligent Routing** — the AI decides which agent handles each request
- **Parallel Execution** — multiple agent tasks running simultaneously
- **Orchestration** — coordinating a full multi-component AI pipeline

### Multi-Agent Systems
- CrewAI-based collaborative agent teams
- Role-based agent design (Researcher, Writer, Analyst, Reviewer)
- Task delegation between agents
- Shared memory and context passing

### Real-World Applications
- Finance AI assistant with live data
- Data analysis copilot with natural language queries
- News summarization with real-time web search
- Vision agent: screenshot → production HTML/CSS
- Natural language → SQL on a live SQLite database

---

## ✅ Prerequisites & Installation

**Requirements:** Python `>= 3.10` and `< 3.13`

### Step 1: Clone the Repository

```bash
git clone https://github.com/jyotidabass/AI-Agent-Session.git
cd AI-Agent-Session
```

### Step 2: Create a Virtual Environment

**Using venv:**
```bash
python -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n ai-agents python=3.12
conda activate ai-agents
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core `requirements.txt`:**
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

**Additional dependencies per module (install as needed):**
```bash
# Modules 1–3 and general use
pip install openai langchain crewai streamlit pandas numpy requests

# SmolAgents (Module 4 + Notebooks 2, 5)
pip install smolagents gradio streamlit groq duckduckgo-search

# Gemini notebooks (Notebooks 3, 4)
pip install google-generativeai langchain-google-genai pillow

# SQL agent (Notebook 5)
pip install sqlalchemy

# News agent
pip install newspaper3k nltk beautifulsoup4
```

---

## 🔐 Environment Variables

Copy the example file and fill in your API keys:

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

These keys allow the AI agents to communicate with large language models and search APIs.

> ⚠️ Never commit your `.env` file — it is already listed in `.gitignore`.
> For Colab users: set keys via `os.environ["KEY"] = "value"` in notebook cells, or use Colab Secrets.

---

## 📘 Module 1 – Introduction to AI Agents

**Folder:** `1-Introduction_Agents/`

A step-by-step introduction to the core building blocks of every AI agent. Run each script independently.

```bash
cd 1-Introduction_Agents
```

### 1. Basic Agent — `1-basic.py`

The simplest possible agent: takes a user message, sends it to an LLM, and returns a response. No tools, no memory.

```bash
python 1-basic.py
```

**Key concept:** Direct LLM prompt → response. The foundation that every other pattern in this repo builds on.

---

### 2. Structured Output Agent — `2-structured.py`

Forces the LLM to return data in a predictable, typed format using Pydantic schemas — instead of a raw text blob.

```bash
python 2-structured.py
```

**Key concept:** Structured outputs are essential when agent results feed into downstream systems like APIs, databases, or UIs that require consistent data shapes.

---

### 3. Tool-Using Agent — `3-tools.py`

An agent equipped with callable tools (Python functions). The LLM autonomously decides when to call a tool and what arguments to pass.

```bash
python 3-tools.py
```

**Key concept:** Tool calling is what bridges LLMs with the real world — web search, calculators, file systems, databases, or any API.

---

### 4. Retrieval Agent (RAG) — `4-retrieval.py`

An agent that answers questions by first retrieving relevant context from `know-db.json`, then passing that context to the LLM.

```bash
python 4-retrieval.py
```

**Knowledge base:** `know-db.json` — a local JSON file acting as the agent's private knowledge store.

**Key concept:** Retrieval-Augmented Generation (RAG) lets agents answer questions about custom, private data without fine-tuning the model.

---

## 🔄 Module 2 – AI Workflow Patterns

**Folder:** `2-workflows/`

```bash
cd 2-workflows
```

### 1. Prompt Chaining — `1-prompt-chaining.py`

Splits a complex task into multiple sequential LLM calls where the output of one step becomes the input of the next.

```bash
python 1-prompt-chaining.py
```

**Example flow:**
```
Step 1: Extract key facts from raw input
    ↓
Step 2: Summarize those facts
    ↓
Step 3: Format the summary as a structured report
```

**Key concept:** Chaining lets you tackle tasks that are too complex for a single prompt by decomposing them into simpler, focused steps.

---

### 2. Intelligent Routing — `2-routing.py`

An LLM acts as a **router** — it reads the incoming request and decides which specialized agent or function should handle it.

```bash
python 2-routing.py
```

**Example flow:**
```
User request → Router LLM decides → Finance Agent
                                  → News Agent
                                  → General QA Agent
```

**Key concept:** Routing enables modular, scalable AI systems where each agent is an expert in its own domain.

---

### 3. Parallel Processing — `3-parallization.py`

Runs multiple independent AI tasks simultaneously and combines their results — dramatically reducing total execution time.

```bash
python 3-parallization.py
```

**Key concept:** When tasks don't depend on each other, run them in parallel. Essential for production AI systems handling real workloads efficiently.

---

### 4. Workflow Orchestrator — `4-orchestrator.py`

Coordinates multiple agents and tools into a structured, managed pipeline. Handles sequencing, dependencies, error recovery, and result aggregation.

```bash
python 4-orchestrator.py
```

**Key concept:** Orchestration is the "brain" that manages complex, multi-step AI workflows reliably and at scale.

---

## 🤝 Module 3 – CrewAI Multi-Agent System

**Folder:** `3-crewAIAgent/`

### Finance App — `Financeapp.py`

A multi-agent financial assistant built with CrewAI where specialized agents collaborate to deliver investment insights and analysis.

```bash
cd 3-crewAIAgent
python Financeapp.py
```

**What it does:**
- Role-based agents each own a specific part of the financial analysis workflow
- Agents delegate subtasks to each other as needed
- Final output: a comprehensive, structured financial report

**Key concepts demonstrated:**
- Agent roles, goals, and backstories
- Task dependencies and sequential execution
- Inter-agent communication and delegation
- Financial data analysis with live web scraping

---

## 🧪 Module 4 – SmolAgents

**Folder:** `4-smolagent/`

### Data Analyst Agent — `DataanalystAgent/app.py`

A Streamlit web app for AI-powered data analysis using natural language queries.

```bash
cd 4-smolagent/DataanalystAgent
streamlit run app.py
```

For real-time streaming responses:
```bash
python streaming.py
```

**Features:**
- Ask questions about data in plain English
- Interactive Streamlit interface
- Real-time streaming agent responses

---

### News Agent — `NewsAgent.py`

An autonomous agent that fetches articles from the web, summarizes them, and provides AI-generated insights on any topic.

```bash
cd 4-smolagent
python NewsAgent.py
```

**Capabilities:**
- Fetches latest news articles from multiple sources
- Summarizes content across articles
- Generates AI-powered analysis and insights

---

## 📓 Jupyter Notebooks

Five self-contained notebooks — each runnable on **Google Colab** with one click. Each notebook installs its own dependencies and generates app files at runtime via `%%writefile` cells.

---

### Notebook 1: CrewAI Financial Analyst

**File:** `1_CrewAI_Notebook.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyotidabass/AI-Agent-Session/blob/main/1_CrewAI_Notebook.ipynb)

**Stack:** CrewAI + OpenAI GPT-3.5-Turbo + ScrapeWebsiteTool

A two-agent team that researches a company's stock and produces a professional investment report with a Buy/Hold/Sell recommendation.

**Agents:**

| Agent | Role | Tools | Responsibility |
|-------|------|-------|----------------|
| `financial_analyst` | Senior Financial Analyst | `ScrapeWebsiteTool` (NASDAQ) | Analyze earnings, market trends, and risk factors |
| `investment_qc` | Investment Strategy Reviewer | — | Review and validate the analyst's report |

**CrewAI config:**
- `memory=True` — agents share context across tasks
- `allow_delegation=True` on `investment_qc` — can delegate back to analyst
- `verbose=2` — full execution logging

**Workflow:**
```
Input: { "company_name": "Tesla, Inc." }
      ↓
stock_analysis_task (financial_analyst)
  → scrapes nasdaq.com for latest stock data
  → produces detailed financial analysis
      ↓
investment_review_task (investment_qc)
  → reviews the report for accuracy and completeness
      ↓
Output: Markdown report with Buy / Hold / Sell recommendation
```

**How to run:**
```python
import os
os.environ["OPENAI_API_KEY"] = "your_key"
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

inputs = {"company_name": "Tesla, Inc."}
result = finance_crew.kickoff(inputs=inputs)
```

---

### Notebook 2: SmolAgent + Streamlit News Analyzer

**File:** `2_SmoleAgent_Streamlit_Notebook.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyotidabass/AI-Agent-Session/blob/main/2_SmoleAgent_Streamlit_Notebook.ipynb)

**Stack:** SmolAgents + Groq (DeepSeek-R1-Distill-LLaMA-70B) + DuckDuckGo News + Streamlit

Deploys a live Streamlit web app that searches the web for any topic and generates a structured, multi-perspective news analysis.

**Architecture:**
```
User Input (topic)
      ↓
DuckDuckGoSearch
  → queries DuckDuckGo News API (worldwide, safesearch on)
  → configurable depth: 3–10 results
  → returns: title, date, source, summary, URL per article
      ↓
create_analysis_prompt()
  → builds a structured LLM prompt from search results
      ↓
GroqLLM (deepseek-r1-distill-llama-70b)
  → temperature=0.7, max_tokens=1024
      ↓
Streamlit UI
  → displays formatted analysis
  → expandable Agent Activity Log (shows full prompt + raw output)
```

**Analysis sections generated:**
1. Key Points Summary — main events and developments
2. Stakeholder Analysis — parties involved and their positions
3. Impact Assessment — immediate and long-term effects
4. Multiple Perspectives — different viewpoints and areas of contention
5. Fact Check & Reliability — cross-source verification and credibility

**Key classes:**
```python
class DuckDuckGoSearch:
    def __call__(self, query: str, max_results: int = 5) -> str:
        # Searches DuckDuckGo News API
        # Returns formatted string with title, date, source, summary, URL

class GroqLLM:
    def __init__(self, model_name="deepseek-r1-distill-llama-70b"):
        # Wraps Groq API for fast LLM inference
    def __call__(self, prompt) -> str:
        # Returns analysis as plain text
```

**Run on Colab:**
```bash
!streamlit run app.py & npx localtunnel --port 8501
!curl https://loca.lt/mytunnelpassword   # copy this password to access the app
```

---

### Notebook 3: Gemini + CrewAI Tech News Generator

**File:** `3_Gemini_crewAI_Agents.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyotidabass/AI-Agent-Session/blob/main/3_Gemini_crewAI_Agents.ipynb)

**Stack:** CrewAI + Google Gemini 1.5 Flash + SerperDevTool + Streamlit

Enter any technology topic and two collaborative AI agents research it and write a publication-ready markdown blog article, saved to `new-blog-post.md`.

**Agents:**

| Agent | Role | Goal | Delegation |
|-------|------|------|------------|
| `news_researcher` | Senior Researcher | Uncover latest tech trends, pros/cons, risks | `allow_delegation=True` |
| `news_writer` | Writer | Write an engaging, accessible 4-paragraph article | `allow_delegation=False` |

Both agents use **Gemini 1.5 Flash** as their LLM and **SerperDevTool** for live Google Search.

**Workflow:**
```
User Input: topic (e.g., "AI in healthcare")
      ↓
research_task (news_researcher)
  → SerperDev web search
  → Output: 3-paragraph research report (trends, opportunities, risks)
      ↓
write_task (news_writer)
  → Output: 4-paragraph markdown article
  → async_execution=False
  → Saved to: new-blog-post.md
      ↓
Displayed in Streamlit UI
```

**LLM configuration:**
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)
```

**Required keys in `.env`:**
```
GOOGLE_API_KEY=your_key
SERPER_API_KEY=your_key
```

**Run on Colab:**
```bash
!pip install -r requirements.txt
!streamlit run Gemini-crewAI-Agents.py & npx localtunnel --port 8501
```

---

### Notebook 4: Gemini Vision Pro – UI to Code

**File:** `4_GeminiVisionPro_Agent.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyotidabass/AI-Agent-Session/blob/main/4_GeminiVisionPro_Agent.ipynb)

**Stack:** Google Gemini 1.5 Flash Latest (Vision) + Streamlit + Pillow

Upload any UI screenshot (JPG or PNG) and receive clean, responsive HTML/CSS code that replicates the original design.

**4-Step Processing Pipeline:**
```
Uploaded Image (JPG/PNG)
      ↓
Step 1: Describe all UI elements with bounding boxes + colors
  Prompt: "Describe this UI in accurate details. When you reference a UI element
           put its name and bounding box in the format: [object name (y_min, x_min,
           y_max, x_max)]. Also describe the color of the elements."
      ↓
Step 2: Refine description by cross-referencing image vs. description
  Prompt: "Compare the described UI elements with the image and identify
           any missing elements or inaccuracies..."
      ↓
Step 3: Generate responsive HTML + inline CSS
  Prompt: "Create an HTML file based on the following UI description...
           Make sure the colors used are the same as the original UI.
           The UI needs to be responsive and mobile-first..."
      ↓
Step 4: Validate and refine the generated HTML
  Prompt: "Validate the following HTML code based on the UI description and
           image and provide a refined version..."
      ↓
Output: index.html — shown in UI with syntax highlighting + downloadable
```

**Model configuration:**
```python
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 50000,
}
MODEL_NAME = "gemini-1.5-flash-latest"
framework = "Regular CSS use flex grid etc"  # change to "Bootstrap" etc. as needed
```

All safety filters set to `BLOCK_NONE` to allow unrestricted UI code generation.

**Run on Colab:**
```bash
!pip install streamlit google-generativeai pillow python-dotenv
!streamlit run app.py & npx localtunnel --port 8501
```

---

### Notebook 5: SmolAgent SQL Agent

**File:** `5_SmolAgent_sqlAgent.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jyotidabass/AI-Agent-Session/blob/main/5_SmolAgent_sqlAgent.ipynb)

**Stack:** SmolAgents + Qwen2.5-Coder-32B-Instruct (HuggingFace) + SQLAlchemy + SQLite + Gradio

A natural language interface to a SQLite database. Type plain English questions and get back SQL query results instantly — no SQL knowledge needed.

**Architecture:**
```
User's natural language question
      ↓
CodeAgent (Qwen2.5-Coder-32B-Instruct via HfApiModel)
  Input: table schema info + user question
  Output: valid SQL SELECT statement
      ↓
sql_engine tool
  → SQLAlchemy executes the query on database.db (SQLite)
  → Returns formatted result string
      ↓
query_sql() handler
  → Tries to parse result as float (for numeric answers)
  → Falls back to string for tabular results
      ↓
Gradio UI
  → Left panel: natural language input + result output
  → Right panel: live receipts table (always visible for reference)
```

**Database schema — `receipts` table (40 pre-loaded sample rows):**
```
receipt_id     INTEGER    (primary key)
customer_name  VARCHAR(16)
price          FLOAT
tip            FLOAT
```

**Example natural language queries:**
```
"What is the total price of all receipts?"
"Who paid the highest tip?"
"Show all customers who spent more than $50"
"What is the average tip amount?"
"How many receipts are there in total?"
"List all customers sorted by price descending"
```

**Files generated at runtime via %%writefile:**

| File | Purpose |
|------|---------|
| `database.py` | Creates `database.db`, defines `receipts` table schema, inserts 40 sample rows |
| `sqlagent.py` | Basic Gradio UI — NL input → agent → SQL → result |
| `new_sqlagent.py` | Enhanced Gradio UI with custom CSS styling and improved error handling |

**Schema prompt passed to the agent:**
```python
schema_info = (
    "The database has a table named 'receipts' with the following schema:\n"
    "- receipt_id (INTEGER, primary key)\n"
    "- customer_name (VARCHAR(16))\n"
    "- price (FLOAT)\n"
    "- tip (FLOAT)\n"
    "Generate a valid SQL SELECT query using ONLY these column names.\n"
    "DO NOT explain your reasoning, and DO NOT return anything other than the SQL query itself."
)
```

**Run on Colab:**
```bash
!pip install gradio smolagents sqlalchemy python-dotenv
!python3 new_sqlagent.py   # launches on port 7861 with share=True (public link auto-generated)
```

---

## 💡 Key Concepts Demonstrated

| Concept | Where It Appears |
|---------|-----------------|
| **Prompt Engineering** | All modules and notebooks |
| **Tool Use** | Module 1 (`3-tools.py`), Notebooks 1–3, 5 |
| **Retrieval-Augmented Generation (RAG)** | Module 1 (`4-retrieval.py`) |
| **Agent Routing** | Module 2 (`2-routing.py`) |
| **Parallel Task Execution** | Module 2 (`3-parallization.py`) |
| **Agent Orchestration** | Module 2 (`4-orchestrator.py`) |
| **Multi-Agent Collaboration** | Module 3, Notebooks 1, 3 |
| **Vision + Multimodal Agents** | Notebook 4 |
| **Natural Language → SQL** | Notebook 5 |
| **Streaming Agent Responses** | Module 4 (`streaming.py`) |

Understanding these patterns helps developers build **production-ready AI applications**.

---

## 🔑 API Keys Required

| Key | Used In | How to Get |
|-----|---------|------------|
| `OPENAI_API_KEY` | Modules 1–3, Notebook 1 | [platform.openai.com](https://platform.openai.com) |
| `SERPER_API_KEY` | Notebook 1, Notebook 3 | [serper.dev](https://serper.dev) |
| `GOOGLE_API_KEY` | Notebook 3, Notebook 4 | [aistudio.google.com](https://aistudio.google.com) |
| `GROQ_API` | Notebook 2 | [console.groq.com](https://console.groq.com) |
| HuggingFace Token | Notebook 5 | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## ☁️ Running Notebooks on Google Colab

1. Click the **"Open in Colab"** badge on any notebook
2. Run all cells from top to bottom — each notebook installs its own dependencies
3. For Streamlit/Gradio apps the final cells start the server and print a tunnel URL
4. For localtunnel-based notebooks, get your access password:
   ```bash
   !curl https://loca.lt/mytunnelpassword
   ```
5. Open the tunnel URL in your browser → enter the password → use the app

---

## 🧰 Technologies Used

| Technology | Purpose |
|------------|---------|
| [CrewAI](https://docs.crewai.com) | Multi-agent orchestration framework |
| [SmolAgents](https://huggingface.co/docs/smolagents) | Lightweight HuggingFace agents |
| [LangChain](https://python.langchain.com) | LLM integrations and chaining |
| [OpenAI GPT-3.5-Turbo](https://platform.openai.com) | Core LLM for agents |
| [Google Gemini 1.5 Flash](https://ai.google.dev) | Multimodal LLM (text + vision) |
| [Groq + DeepSeek-R1](https://console.groq.com) | Ultra-fast LLM inference |
| [Qwen2.5-Coder-32B](https://huggingface.co/Qwen) | Code-specialized SQL generation |
| [Streamlit](https://streamlit.io) | Web UI for Python apps |
| [Gradio](https://gradio.app) | ML demo interfaces |
| [SQLAlchemy](https://sqlalchemy.org) | SQL toolkit and ORM |
| [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) | Privacy-first web search |
| [SerperDev](https://serper.dev) | Google Search API for agents |
| [Pydantic](https://docs.pydantic.dev) | Structured data validation |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |
| [localtunnel](https://theboroer.github.io/localtunnel-www/) | Expose Colab apps publicly |

---

## 🎯 Use Cases

The techniques in this repository can be directly applied to:

- **AI Research Assistants** — multi-agent research, summarization, and report generation
- **Data Analysis Copilots** — natural language queries over any structured dataset
- **Financial Advisors** — automated stock analysis and investment reporting
- **News Summarization Systems** — real-time article fetching, synthesis, and insight generation
- **Automated Workflows** — chained, routed, and parallelized AI task pipelines
- **UI / Frontend Automation** — screenshot-to-production-code generation
- **Customer Support Agents** — routed, tool-using conversational agents

---

## 🤝 Contributing

Contributions are welcome! If you would like to improve the project:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE) for educational and research purposes.

---

## 👩‍💻 Author

**Jyoti Dabass, Ph.D**
Researcher in NLP, Computer Vision & AI

GitHub: [github.com/jyotidabass](https://github.com/jyotidabass)

---

⭐ **If you found this repository helpful, please star it to support the project!**
