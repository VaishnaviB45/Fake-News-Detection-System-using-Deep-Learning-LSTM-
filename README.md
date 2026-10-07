# Agentic VeriFact: Explainable News Verification via Deep Learning & Agentic Retrieval

An end-to-end AI verification system combining sequential stylistic pattern recognition (LSTM) with real-time fact retrieval (Agentic RAG) to verify claims and explain verdicts via a high-performance FastAPI service.

---

## 🎯 Architecture & Workflow

```text
[ Incoming News Claim ]
           │
           ├──► [ Branch 1: Deep Learning Style Scorer ]
           │    └── TensorFlow/Keras LSTM trained on WELFake dataset
           │    └── Evaluates text sensationalism & linguistic patterns
           │
           ├──► [ Branch 2: Agentic Search Retriever ]
           │    └── Queries live reporting & trusted web registries
           │    └── Gathers ground-truth citations
           │
           └──► [ Verification Engine & Synthesis ]
                └── Cross-references linguistic style with factual evidence
                └── Emits structured JSON verdict via FastAPI async endpoints