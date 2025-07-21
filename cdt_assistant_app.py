

import streamlit as st
import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CDT knowledge base
cdt_df = pd.read_csv("C:\\Users\\satis\\Downloads\\CDT_AI_Training_100_New_Rows.csv")
cdt_df.fillna("", inplace=True)

# Secure API Key and OpenRouter endpoint
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.base_url = "https://openrouter.ai/api/v1"
MODEL = "mistralai/mistral-7b-instruct"

# Prepare text features for similarity search
descriptions = (
    cdt_df["Short Description"].astype(str)
    + " "
    + cdt_df["When to Use"].astype(str)
    + " "
    + cdt_df["Common Mistakes"].astype(str)
)
vectorizer = TfidfVectorizer().fit(descriptions)
vectors = vectorizer.transform(descriptions).toarray()

def find_similar_descriptions(clinical_note, top_n=3):
    note_vector = vectorizer.transform([clinical_note]).toarray()
    similarity = cosine_similarity(note_vector, vectors)
    top_indices = similarity.argsort()[0][-top_n:][::-1]
    return cdt_df.iloc[top_indices]

def ask_gpt(clinical_note, tooth_number, surface, top_matches):
    prompt = f"""
You are a CDT coding expert assistant. Based on the following clinical note and candidate CDT entries, suggest the most appropriate CDT code.

Respond ONLY in clean text. 
Format:
CDT Code: <code>
Reason: <why this code fits>

Clinical Note: {clinical_note}
Tooth Number: {tooth_number}
Surface: {surface}

Candidate Codes:
{top_matches.to_string(index=False)}
"""

    # Call OpenRouter (OpenAI-compatible)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful CDT coding assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    # Handle raw string response safely
    content = response['choices'][0]['message']['content']
    return content.strip()

# Streamlit interface
st.title("🦷 All in One Assist (AI Powered CDT Code Assistant) ")
st.markdown("Get smart CDT code suggestions from messy clinical notes.")

clinical_note = st.text_area("📝 Enter Clinical Note", height=180)
tooth_number = st.text_input("🦷 Tooth Number (optional)")
surface = st.text_input("🦷 Tooth Surface (optional)")

if st.button("Suggest CDT Code"):
    with st.spinner("Thinking like a CDT expert..."):
        top_matches = find_similar_descriptions(clinical_note)
        try:
            suggestion = ask_gpt(clinical_note, tooth_number, surface, top_matches)
            st.success("✅ Suggested CDT Code & Reason")
            st.write(suggestion)
        except Exception as e:
            st.error(f"Error: {e}")

        st.subheader("📚 Top Matching Codes from Knowledge Base")
        st.dataframe(top_matches.reset_index(drop=True))







