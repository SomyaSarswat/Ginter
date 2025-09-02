# LLM-Based Intelligent Document Query System

This project builds a system that leverages **Large Language Models (LLMs)** to process natural language queries and retrieve relevant information from **unstructured documents** like policy files, contracts, and emails.

---

## 🧠 Objective

To develop a system that can:

- Accept user queries in natural language (e.g., **“46-year-old male, knee surgery in Pune, 3-month-old insurance policy”**).
- Parse the query to extract key structured information.
- Semantically search and match relevant clauses in large documents.
- Evaluate and return an automated decision (approval, rejection, payout, etc.).
- Provide transparent reasoning by linking the decision to specific clauses.

---

## 🧩 Key Features

- **Natural Language Query Parsing**  
  Automatically extracts entities like:
  - Age
  - Medical procedure
  - Location
  - Insurance/policy duration

- **Semantic Retrieval**  
  Uses LLMs or embeddings (e.g., via FAISS, Pinecone) to find **semantically** relevant clauses—not just keyword matches.

- **Clause Evaluation & Logic Application**  
  Applies predefined or learned logic to the matched clauses to generate a decision.

- **Structured JSON Output**  
  Returns a clear response with:
  ```json
  {
    "decision": "approved",
    "amount": "₹75,000",
    "justification": "Clause 4.2: Surgery covered for policies older than 90 days"
  }
