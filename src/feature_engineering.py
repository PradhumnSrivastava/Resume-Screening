from sklearn.feature_extraction.text import TfidfVectorizer
import re


#TF-IDF Vectorization (DON'T REMOVE THIS)
def vectorize(resumes, job_desc):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    vectors = vectorizer.fit_transform(resumes + [job_desc])
    return vectors


#Dynamic Keyword Extraction (Improved)
def extract_keywords(text):
    text = text.lower()

    words = re.findall(r'\b[a-zA-Z]+\b', text)

    stopwords = {
        "the","and","with","for","are","you","this","that",
        "have","has","will","should","can","our","your",
        "job","role","we","looking","candidate",
        "experience","knowledge","required","skills",
        "work","working","build","using","used",
        "strong","good","ability","capable",
        "team","teams","responsibilities",
        "education","degree","bachelor","master",
        "location","india","years","freshers",
        "highly","motivated","ideal"
    }

    clean_words = [w for w in words if w not in stopwords and len(w) > 2]

    # bigrams
    phrases = []
    for i in range(len(clean_words) - 1):
        phrase = clean_words[i] + " " + clean_words[i+1]
        phrases.append(phrase)

    # filter meaningful phrases
    final_keywords = []

    for p in phrases:
        if any(x in p for x in ["data", "learning", "python", "sql", "analysis", "model", "nlp", "deep"]):
            final_keywords.append(p)

    # important single words
    important_words = [
        "python","sql","pandas","numpy","nlp",
        "tensorflow","pytorch","statistics"
    ]

    for w in clean_words:
        if w in important_words:
            final_keywords.append(w)

    return list(set(final_keywords))