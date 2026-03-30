import streamlit as st
import sys
import os

# Fix import path issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser import extract_text
from src.preprocess import clean_text
from src.feature_engineering import vectorize, extract_keywords
from src.similarity import calculate_similarity, jd_match
from src.ranking import rank_candidates

st.title("AI Resume Screening System")

jd_input = st.text_area("Paste Job Description")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze"):

    if not jd_input or not uploaded_files:
        st.warning("Please provide Job Description and upload resumes.")
    
    else:
        resume_texts = []
        resume_names = []
        matched_all = []
        missing_all = []

        # Clean JD and extract keywords
        jd_cleaned = clean_text(jd_input)
        jd_keywords = extract_keywords(jd_cleaned)

        for file in uploaded_files:
            text = extract_text(file)
            cleaned = clean_text(text)

            resume_texts.append(cleaned)
            resume_names.append(file.name)

            # Match JD keywords with resume
            matched, missing = jd_match(cleaned, jd_keywords)

            matched_all.append(", ".join(matched) if matched else "None")
            missing_all.append(", ".join(missing) if missing else "None")

        # Similarity score
        vectors = vectorize(resume_texts, jd_cleaned)
        scores = calculate_similarity(vectors)

        # Ranking
        result = rank_candidates(resume_names, scores, matched_all, missing_all)

        st.subheader("Ranking Results")
        st.dataframe(result)

        st.subheader("Detailed Skill Analysis")

        
        for i, row in result.iterrows():
            st.markdown(f"### {row['Candidate']} (Score: {row['Score']:.2f})")

            # Matched Skills
            st.markdown("Matched Skills:")
            if row['Matched Skills'] != "None":
                for skill in row['Matched Skills'].split(", "):
                    st.markdown(
                        f"<div style='padding:6px; margin:4px; border:1px solid #2e7d32; display:inline-block; border-radius:6px; background-color:#e8f5e9; color:#1b5e20; font-weight:500;'>{skill}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.write("No matched skills")

            # Missing Skills
            st.markdown("Missing Skills:")
            if row['Missing Skills'] != "None":
                for skill in row['Missing Skills'].split(", "):
                    st.markdown(
                        f"<div style='padding:6px; margin:4px; border:1px solid #c62828; display:inline-block; border-radius:6px; background-color:#ffebee; color:#b71c1c; font-weight:500;'>{skill}</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.write("No missing skills")

            st.markdown("---")