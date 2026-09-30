
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

MODEL_FILE = Path("movie_recommender.pkl")

def load_model():
    if not MODEL_FILE.exists():
        print("movie_recommender.pkl not found.")
        print("Run the model-saving cell in movielens_recommendation_system.ipynb first.")
        raise SystemExit(1)
    with open(MODEL_FILE,"rb") as f:
        return pickle.load(f)

def content_recommend(model, movie_id, n=10):
    movies=model["movies"]; sim=model["content_similarity"]
    rows=movies.index[movies.MovieID==movie_id].tolist()
    if not rows: return pd.DataFrame()
    i=rows[0]; scores=sim[i].toarray().ravel(); scores[i]=-1
    top=np.argsort(scores)[::-1][:n]
    out=movies.iloc[top][["MovieID","Title","Genres"]].copy()
    out["Similarity"]=scores[top]
    return out.reset_index(drop=True)

def collaborative_recommend(model, movie_id, n=10):
    movies=model["movies"]; sim=model["collab_similarity"]
    ids=model["collab_movie_ids"]; mapping=model["collab_id_to_idx"]
    if movie_id not in mapping: return pd.DataFrame()
    i=mapping[movie_id]; scores=sim[i].toarray().ravel(); scores[i]=-1
    top=np.argsort(scores)[::-1][:n]; top_ids=ids[top]
    out=movies[movies.MovieID.isin(top_ids)][["MovieID","Title","Genres"]].copy()
    score_map=dict(zip(top_ids,scores[top]))
    out["Similarity"]=out.MovieID.map(score_map)
    return out.sort_values("Similarity",ascending=False).reset_index(drop=True)

def print_results(df):
    if df.empty:
        print("No recommendations available.")
        return
    for i,row in df.iterrows():
        s=row.get("Similarity",np.nan)
        extra=f" | similarity={s:.3f}" if pd.notna(s) else ""
        print(f"{i+1:2}. {row.Title} | {row.Genres}{extra}")

def main():
    model=load_model()
    movies=model["movies"]
    print("="*60)
    print("MOVIELENS 1M MOVIE RECOMMENDATION SYSTEM")
    print("="*60)
    print("Type part of a title, or 'exit' to quit.")

    while True:
        query=input("\nMovie title: ").strip()
        if query.lower() in {"exit","quit","q"}:
            break
        if not query:
            continue

        matches=movies[movies.Title.str.contains(query,case=False,regex=False,na=False)]
        if matches.empty:
            print("Movie not found.")
            continue

        matches=matches.head(10)
        print("\nMatches:")
        for i,(_,r) in enumerate(matches.iterrows(),1):
            print(f"{i}. {r.Title}")

        try:
            choice=int(input("Choose movie number: "))
            selected=matches.iloc[choice-1]
        except (ValueError,IndexError):
            print("Invalid selection.")
            continue

        print("\n1. Content-Based")
        print("2. Collaborative")
        print("3. Both")
        method=input("Choose method: ").strip()

        if method in {"1","3"}:
            print(f"\n--- Content-Based: {selected.Title} ---")
            print_results(content_recommend(model,int(selected.MovieID),10))

        if method in {"2","3"}:
            print(f"\n--- Collaborative: {selected.Title} ---")
            print_results(collaborative_recommend(model,int(selected.MovieID),10))

if __name__=="__main__":
    main()
