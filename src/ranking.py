import pandas as pd

def rank_candidates(names, scores, matched_list, missing_list):
    df = pd.DataFrame({
        "Candidate": names,
        "Score": scores.flatten(),
        "Matched Skills": matched_list,
        "Missing Skills": missing_list
    })
    
    df = df.sort_values(by="Score", ascending=False)
    return df