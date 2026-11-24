# Intelligent Library Book Recommendation System

A machine learning project that recommends books to library readers by comparing four approaches: Popularity, Content-Based Filtering, Item-Item Collaborative Filtering (KNN) and Matrix Factorization (SVD). The project includes a full data analysis pipeline, a model comparison, and a Streamlit web app.

## Student Information

| | |
|---|---|
| **Name** | Prashant Kumar |
| **Department** | CSE (AI & ML) |
| **Institute** | Ghani Khan Choudhury Institute of Engineering and Technology (GKCIET), Malda |
| **Subject** | Introduction to Machine Learning |
| **Subject Code** | PCCAIML 502 |
| **Domain** | Machine Learning |
| **Guided by** | Dr. Showmik Bhowmik |

## 1. Problem Statement

A college library holds thousands of books, and readers often spend a long time looking for something they will enjoy. This project builds an intelligent recommendation system that uses past reader ratings and book details (title, author, tags) to suggest books a reader is likely to like.

### Objectives

1. Clean the rating and book data and explore it through visual analysis.
2. Implement and compare four recommendation approaches: Popularity, Content-Based, Item-Item Collaborative Filtering and SVD.
3. Evaluate the models with RMSE, MAE, Precision@K and Recall@K.
4. Build a working web prototype with Streamlit.

## 2. Dataset

**Goodbooks-10k** (Zajac, 2017): ratings from Goodreads readers for the 10,000 most popular books, together with book metadata and tags.

- GitHub (latest version): https://github.com/zygmuntz/goodbooks-10k
- Kaggle: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k

Files used: `books.csv`, `ratings.csv`, `tags.csv`, `book_tags.csv`.

### Download instructions

The raw dataset is not stored in this repository because of its size.

1. Download the dataset ZIP from the GitHub releases page: https://github.com/zygmuntz/goodbooks-10k/releases
2. Extract the four CSV files into the `data/` folder of this project.

### Dataset after preprocessing

| Item | Value |
|---|---|
| Readers (users) | 10,000 |
| Books | 9,989 |
| Ratings | 493,250 |
| Average rating | 3.85 |
| Sparsity of the rating matrix | 99.51% |
| Train / Test split | 394,600 / 98,650 (80% / 20%) |

## 3. Methodology

### Data preprocessing and EDA (`notebooks/01_data_preprocessing_eda.ipynb`)

1. Removed duplicate ratings and ratings outside the 1 to 5 range.
2. Filled missing values in the book metadata.
3. Kept active readers (at least 20 ratings) and popular books (at least 50 ratings), then took a random sample of 10,000 readers so the project runs on a normal laptop.
4. Created five visualizations: rating distribution, ratings per reader, most rated books, highest rated books and top authors.
5. Saved the cleaned files `ratings_clean.csv` and `books_clean.csv`.

### Algorithms (`notebooks/02_model_building.ipynb`)

| # | Model | How it works | Main settings |
|---|---|---|---|
| 1 | **Popularity (baseline)** | Ranks books by a weighted rating (IMDB formula), so a book with a few perfect ratings does not beat a book loved by many readers. | Minimum votes = 70th percentile |
| 2 | **Content-Based** | Converts each book's title, author and tags into TF-IDF vectors and finds the closest books with cosine similarity. A reader profile is the average of the books they liked. | 20,000 features, 1-2 word n-grams |
| 3 | **Item-Item KNN** | Finds books that were rated similarly by the same readers (cosine distance on the book x reader matrix). | k = 20 neighbours |
| 4 | **SVD (Matrix Factorization)** | Mean-centres ratings per reader and applies Truncated SVD to learn hidden taste factors, then predicts ratings for unread books. | 50 components |

### Evaluation

- **RMSE and MAE** measure how far predicted ratings are from real ratings (lower is better). The Content-Based model does not predict ratings, so it has no RMSE or MAE.
- **Precision@10 and Recall@10** measure the quality of the top 10 recommendations. A held-out test book counts as relevant when the reader rated it 4 or higher. Books the reader already rated in the training data are never recommended. The scores are averaged over 1,000 randomly chosen readers.

## 4. Results

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Popularity (Item Mean) | 0.9549 | 0.7669 | 0.0010 | 0.0021 |
| Content-Based (TF-IDF) | n/a | n/a | 0.0969 | 0.1523 |
| Item-Item KNN | 0.8839 | **0.6218** | **0.1262** | **0.2040** |
| SVD | **0.8503** | 0.6597 | 0.0107 | 0.0120 |

![Model comparison](images/06_model_comparison.png)

### Observations

- **Item-Item KNN is the best model for ranking.** It has the highest Precision@10 (0.126) and Recall@10 (0.204), and the lowest MAE. About one in eight of its top 10 suggestions was a book the reader later rated 4 or higher.
- **SVD predicts ratings most accurately** (lowest RMSE, 0.850) but performs poorly at ranking (Precision@10 of 0.011). Predicting a rating well and choosing the right top 10 books are different tasks. Truncated SVD also treats unrated books as zeros and explained only 5.3% of the variance, which limits its ranking quality. A factorization method trained only on the observed ratings could improve this.
- **Content-Based filtering is a strong second for ranking** (Precision@10 of 0.097) even though it never uses ratings of other readers. It also works for books with no rating history.
- **Popularity is the weakest baseline.** It gives every reader the same list, so almost none of those books appear in a given reader's test set (Precision@10 of 0.001). Its RMSE of 0.955 is also the highest, which shows that personalization helps.
- All Precision values look small in absolute terms because the rating matrix is 99.51% sparse and a reader's test set holds only a few relevant books. The comparison between models is more meaningful than the absolute numbers.

### Limitations

- Tag information could be joined for only 812 of 9,989 books, because the cleaned book file has no Goodreads ID column to link with `book_tags.csv`. The Content-Based model therefore relies mostly on title and author, and the tag join for those 812 books is approximate.
- The data is a 10,000 reader sample of Goodreads ratings, not real borrowing records from a college library.
- Results come from a single random train/test split and were not cross-validated.

### Future work

- Use real borrowing data from the college library.
- Train a matrix factorization model only on observed ratings (for example ALS) and tune the number of factors.
- Build a hybrid model that combines Content-Based and Collaborative scores.
- Fix the tag join so all books have tag features.

## 5. Streamlit Prototype

The web app lets a reader:

- see the most popular books,
- pick a book and get similar books (Content-Based or Collaborative),
- select a reader and get personal recommendations (SVD or KNN),
- view the model comparison results.

![App screenshot](images/app_screenshot.png)

## 6. Project Structure

```
library-book-recommender/
├── app/
│   └── app.py                      # Streamlit web app
├── data/                           # CSV files go here (not tracked by Git)
├── images/                         # EDA charts and model comparison chart
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb
│   └── 02_model_building.ipynb     # models + evaluation
├── .streamlit/
│   └── config.toml                 # app theme
├── requirements.txt
└── README.md
```

## 7. How to Run

**Requirements:** Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/library-book-recommender.git
cd library-book-recommender

# 2. Install dependencies
pip install -r requirements.txt
```

3. Download the dataset and place the four CSV files in `data/` (see Section 2).
4. Run `notebooks/01_data_preprocessing_eda.ipynb` from the first cell to the last. This creates `ratings_clean.csv` and `books_clean.csv` in `data/`.
5. Run `notebooks/02_model_building.ipynb` from the first cell to the last. This trains the models and creates `data/model_comparison.csv` and the comparison chart.
6. Start the app from the project root folder:

```bash
python -m streamlit run app/app.py
```

The app opens at `http://localhost:8501`. The first start takes 1 to 2 minutes while the models are built.

## 8. Technologies

Python, Pandas, NumPy, SciPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Jupyter Notebook.

## 9. Acknowledgements

- Dataset: Zygmunt Zajac, *Goodbooks-10k: a new dataset for book recommendations*, FastML, 2017. http://fastml.com/goodbooks-10k
- The Goodbooks-10k dataset is available on Kaggle under the CC BY-SA 4.0 license.
- Libraries: Scikit-learn, Streamlit, Pandas and the other open-source packages listed above.
- Guidance: Dr. Showmik Bhowmik, GKCIET, Malda.
