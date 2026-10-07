import os
import re
from duckduckgo_search import DDGS

# Try importing TensorFlow if available locally
HAS_TF = False
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import pickle
    HAS_TF = True
except ImportError:
    HAS_TF = False

MODEL_PATH = "fake_news_lstm.keras"
TOKENIZER_PATH = "tokenizer.pkl"

model = None
tokenizer = None

if HAS_TF and os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception:
        model = None
        tokenizer = None

def analyze_linguistic_credibility(text: str) -> tuple[float, str]:
    """
    Evaluates credibility using LSTM if loaded; otherwise applies
    statistical linguistic sensationalism heuristics.
    """
    if model is not None and tokenizer is not None:
        clean = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        seq = tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=300)
        prob = float(model.predict(padded, verbose=0)[0][0])
        score = prob
    else:
        # Cloud/Fallback Linguistic Evaluation
        sensational_triggers = [
            "shocking", "urgent", "you won't believe", "secret", "exposed",
            "conspiracy", "leak", "hidden truth", "miracle", "proof"
        ]
        text_lower = text.lower()
        triggers_count = sum(1 for word in sensational_triggers if word in text_lower)
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        base_score = 0.90
        base_score -= (triggers_count * 0.25)
        if uppercase_ratio > 0.20:
            base_score -= 0.20
        score = max(0.05, min(0.98, base_score))

    if score > 0.65:
        interpretation = "Formal, neutral reporting structure"
    elif score > 0.40:
        interpretation = "Mixed linguistic markers / Moderate tone"
    else:
        interpretation = "Sensationalist / Clickbait linguistic pattern"
        
    return score, interpretation

def retrieve_grounding_evidence(query: str, max_results: int = 3):
    results = []
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "source": r.get("href", "")
                })
    except Exception as e:
        results = []
    return results

def run_verifact_agent(claim: str) -> dict:
    style_score, style_desc = analyze_linguistic_credibility(claim)
    evidence = retrieve_grounding_evidence(claim)
    
    evidence_count = len(evidence)
    top_source = evidence[0]["title"] if evidence else "No direct live source identified"
    
    # Synthesis rule
    if evidence_count >= 2 and style_score >= 0.50:
        final_verdict = "VERIFIED REAL"
    elif evidence_count >= 1:
        final_verdict = "LIKELY REAL (Corroborated)"
    elif style_score < 0.40:
        final_verdict = "SUSPICIOUS / CLICKBAIT"
    else:
        final_verdict = "UNVERIFIED / INSUFFICIENT EVIDENCE"
        
    return {
        "claim": claim,
        "lstm_style_score": round(style_score, 2),
        "style_interpretation": style_desc,
        "grounding_evidence_found": evidence_count,
        "top_source": top_source,
        "final_verdict": final_verdict
    }