# 🧠 Agentic RAG Multi-Source - Medications & Health

This project implements an innovative **Agentic Retrieval-Augmented Generation (RAG)** solution that combines multiple trusted medical databases to facilitate understanding of drug-related information. Through an agent-based architecture, we orchestrate intelligent queries across sources to provide coherent and simplified responses to end users.

---

## 🚀 Project Goal

Provide end users (patients, healthcare professionals, developers) with a simple and understandable interface to query the following medical databases:
- **DailyMed**: Provides official drug labeling and pharmaceutical information as published by the U.S. National Library of Medicine.
- **Martindale**: An international drug reference offering comprehensive information on drugs and medicines used worldwide.
- **FDA Drug Database**: Contains regulatory, approval, and safety information on drugs evaluated by the U.S. Food and Drug Administration.

All results are enriched by a **Writer Agent** based on **GPT-4o**, which reformulates and simplifies the output from expert agents for improved comprehension.

---

## 🧩 Technical Architecture

![Architecture Diagram](architecture%20diagram.png)

### 🔗 Core Components

- **4 Specialized Agents:**
- `Martindale Agent` (Azure search + GPT4o1): Queries the Martindale database to extract international drug information and references.
- `DailyMed Agent` (Azure search + GPT4o1): Retrieves official drug labels and descriptions from the DailyMed database.
- `FDA Agent` (Azure search + GPT4o1): Extracts regulatory and safety data from the FDA Drug Database.
- `Writer Agent` (GPT4o): Simplifies and reformulates technical outputs from other agents using GPT-4o for user-friendly understanding.

- **1 Agent Manager**: Orchestrates coordination between agents (logic powered by GPT o1)

- **2 Web Applications deployed on Azure Container Apps:**
  - `API App`: Handles orchestration and business logic
  - `Frontend App`: User interface built with **Streamlit**

- **Azure Services Used:**
  - Azure AI Search (vector-based retrieval)
  - Azure Foundry (model hosting - GPT-4o, o1)
  - Azure Container Registry, Front Door, Firewall
  - Monitoring tools: Application Insights, Log Analytics, Budgets

---

## 💡 Why Is This Innovative?

- **Improving Access to Medical Knowledge**: Makes complex drug-related information from trusted medical databases accessible and understandable to non-specialists.
- **Multi-Source Aggregation**: Combines insights from diverse, authoritative sources (DailyMed, Martindale, FDA) to provide a more complete and reliable picture.
- **Bridging the Expert-Layperson Gap**: Uses AI to translate technical medical content into plain language, empowering patients and healthcare professionals alike.
- **Promoting Transparency in Drug Information**: Ensures users can trace responses back to verified medical sources with contextual clarity.

---

## ⚙️ Key Technologies

- `Python` (multi-agent logic via **AutoGen**)
- `Streamlit` (frontend interface)
- `Azure AI Search` (RAG engine)
- `GPT-4o` (simplification) & `o1` (agent logic and reasoning)
- `Azure Container Apps`, `Container Registry`