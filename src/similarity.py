from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(vectors):
    job_vector = vectors[-1]
    resume_vectors = vectors[:-1]
    scores = cosine_similarity(resume_vectors, job_vector)
    return scores


def jd_match(resume_text, jd_keywords):
    resume_text = resume_text.lower()

    matched = []
    missing = []

    for keyword in jd_keywords:
        if keyword in resume_text:
            matched.append(keyword)
        else:
            missing.append(keyword)

    return matched, missing