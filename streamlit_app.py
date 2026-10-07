import streamlit as st
import requests

st.set_page_config(
    page_title="Agentic VeriFact",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Agentic VeriFact")
st.caption("Explainable News Verification powered by Deep Learning (LSTM) & Live Web Grounding")

API_URL = "http://127.0.0.1:8000/api/verify"

user_input = st.text_area(
    "Enter a news claim or headline to verify:",
    placeholder="e.g., ISRO successfully launched a weather observation satellite.",
    height=120
)

if st.button("Verify Claim", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a claim first.")
    else:
        with st.spinner("Analyzing style with LSTM and searching live corroborating evidence..."):
            try:
                response = requests.post(API_URL, json={"text": user_input})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.divider()
                    
                    # 1. Final Verdict Display
                    verdict = data.get("final_verdict", "UNKNOWN")
                    if verdict == "VERIFIED REAL":
                        st.success(f"### Verdict: {verdict}")
                    else:
                        st.error(f"### Verdict: {verdict}")
                    
                    # 2. Detailed Inspection Columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="LSTM Credibility Score",
                            value=f"{data.get('lstm_style_score', 0.0) * 100:.1f}%"
                        )
                        st.write(f"**Style:** {data.get('style_interpretation', 'N/A')}")
                    
                    with col2:
                        st.metric(
                            label="Evidence Sources Found",
                            value=data.get("grounding_evidence_found", 0)
                        )
                        st.write(f"**Top Source:** {data.get('top_source', 'None')}")
                        
                else:
                    st.error(f"Backend returned error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI backend. Ensure `python app.py` is running on port 8000.")