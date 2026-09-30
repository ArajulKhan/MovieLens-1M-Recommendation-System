# MovieLens 1M Recommendation System

## Files
- `movielens_recommendation_system.ipynb` — complete notebook for the 12 requested requirements.
- `movie_recommender_cli.py` — working CLI prototype.
- `summary.txt` — one-page summary.
- `README.md` — setup and usage.
- `requirements.txt` — dependencies.

**No `.pkl` file is included.** The notebook creates `movie_recommender.pkl` when you run the Model Saving cell.

## Dataset
Download the MovieLens 1M dataset from the Kaggle source:
https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset

Extract it so that the project contains:
```text
ml-1m/
  movies.dat
  ratings.dat
  users.dat
  README.txt
```

## Install
```bash
pip install -r requirements.txt
```

## Create the model
Open `movielens_recommendation_system.ipynb`, place `ml-1m` beside it, and run the cells in order. The final saving cell creates:
```text
movie_recommender.pkl
```

## Run CLI
After creating the pickle:
```bash
python movie_recommender_cli.py
```

Enter a movie title such as `Toy Story`, select the matching movie, and choose Content-Based, Collaborative, or Both.

## Methods
**Popularity:** rating count + average rating with shrinkage.

**Content-Based:** genres → TF-IDF → cosine similarity.

**Collaborative:** movie-user ratings → mean-centering → L2 normalization → cosine similarity.

## Evaluation
Recommendation is a ranking problem, so Precision@K, Recall@K and Hit Rate@K are more appropriate than ordinary classification accuracy. The notebook includes a reproducible leave-one-out style evaluation for the popularity baseline and additional content-similarity analysis.

## Important
Run the notebook on the actual dataset before reporting numerical results. The project does not invent final metrics.

## Source
MovieLens 1M / GroupLens Research, University of Minnesota.
