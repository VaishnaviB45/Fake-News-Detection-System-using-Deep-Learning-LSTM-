import pickle
from tensorflow.keras.models import load_model

# Verify that saved artifacts load correctly
model = load_model("fake_news_lstm.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Step 2 Verification: All libraries working, model and tokenizer loaded perfectly!")