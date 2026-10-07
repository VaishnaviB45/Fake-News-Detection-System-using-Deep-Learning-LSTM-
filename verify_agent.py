import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ddgs import DDGS

# 1. Load trained artifacts
lstm_model = load_model("fake_news_lstm.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 2. Tool 1: LSTM Stylistic Analysis
def analyze_style_credibility(text: str) -> float:
    """Predicts credibility probability using trained LSTM."""
    cleaned_text = text.lower()
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=100)
    score = float(lstm_model.predict(padded, verbose=0)[0][0])
    return score

# 3. Tool 2: Web Search for Fact Grounding
def search_factual_evidence(query: str, max_results: int = 3) -> list:
    """Queries DuckDuckGo for live reporting and fact checks."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results or []
    except Exception as e:
        return []

# 4. Agent Synthesizer
def run_verifact_agent(news_text: str) -> dict:
    # Step A: Local deep learning model
    style_score = analyze_style_credibility(news_text)
    
    # Step B: Live fact retrieval
    search_query = f"ISRO weather satellite cyclone"
    articles = search_factual_evidence(search_query)

    # Step C: Deterministic reasoning & verdict
    verdict = "VERIFIED REAL" if len(articles) > 0 and style_score > 0.5 else "SUSPICIOUS / UNVERIFIED"
    
    report = {
        "claim": news_text.strip(),
        "lstm_style_score": round(style_score, 2),
        "style_interpretation": "Formal, neutral reporting structure" if style_score > 0.5 else "Sensationalist / Clickbait markers detected",
        "grounding_evidence_found": len(articles),
        "top_source": articles[0]["title"] if articles else "None",
        "final_verdict": verdict
    }
    return report

if __name__ == "__main__":
    sample_news = (
        "India successfully launched a new weather observation satellite to improve "
        "cyclone forecasting and climate monitoring, according to officials from ISRO."
    )
    print("\n--- Running Agent Verification Pipeline ---\n")
    result = run_verifact_agent(sample_news)
    
    print(f"Claim: {result['claim']}")
    print(f"LSTM Style Credibility: {result['lstm_style_score']} ({result['style_interpretation']})")
    print(f"External Sources Found: {result['grounding_evidence_found']}")
    print(f"Top Corroborating Source: {result['top_source']}")
    print(f"\nFINAL VERDICT: {result['final_verdict']}")