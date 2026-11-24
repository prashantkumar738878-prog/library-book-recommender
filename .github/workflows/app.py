import os
import html
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="Library Book Recommender", page_icon="📚", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "Data")
IMG = os.path.join(BASE, "..", "images")

# ============================== STYLE ==============================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&display=swap');
:root{--ink:#16303F;--teal:#1F6F6B;--brass:#B8862B;--bg:#F4F6F8;--card:#FFFFFF;--text:#1B2A33;--muted:#5B6B76;--line:#E1E6EA;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:var(--text);}
.stApp{background:var(--bg);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.4rem;max-width:1200px;}
h1,h2,h3{font-family:'Source Serif 4',serif;color:var(--ink);letter-spacing:-0.01em;}

.hero{background:var(--ink);border-radius:14px;padding:30px 36px;margin-bottom:18px;}
.hero h1{color:#fff;font-size:2.1rem;margin:0 0 6px 0;}
.hero p{color:#C5D3DB;margin:0;font-size:1rem;max-width:720px;line-height:1.55;}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;}
.stat b{display:block;font-size:1.5rem;color:var(--ink);font-family:'Source Serif 4',serif;}
.stat span{color:var(--muted);font-size:.85rem;}

.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{height:44px;padding:0 18px;font-weight:500;color:var(--muted);}
.stTabs [aria-selected="true"]{color:var(--ink)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--brass)!important;}

.section-note{color:var(--muted);font-size:.95rem;margin:6px 0 14px 0;max-width:760px;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;margin-top:10px;}
.card{display:flex;gap:14px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;}
.cover{position:relative;flex:0 0 64px;height:92px;border-radius:4px 8px 8px 4px;display:flex;align-items:center;
  justify-content:center;color:#fff;font-family:'Source Serif 4',serif;font-size:1.7rem;font-weight:700;overflow:hidden;}
.cover img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;}
.meta{min-width:0;flex:1;}
.rank{color:var(--muted);font-size:.78rem;}
.title{font-weight:600;color:var(--ink);line-height:1.3;margin:2px 0;}
.author{color:var(--muted);font-size:.88rem;margin-bottom:8px;}
.chips{display:flex;gap:6px;flex-wrap:wrap;}
.chip{background:#EAF3F2;color:var(--teal);border-radius:20px;padding:2px 10px;font-size:.78rem;font-weight:500;}
.chip.gold{background:#F7EFDD;color:#8A6317;}

.pick{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--brass);border-radius:8px;
  padding:12px 16px;margin:10px 0 4px 0;}
.pick span{color:var(--muted);font-size:.85rem;}
.stButton>button{background:var(--teal);color:#fff;border:0;border-radius:8px;padding:.55rem 1.4rem;font-weight:600;}
.stButton>button:hover{background:var(--ink);color:#fff;}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
@media(max-width:800px){.stats{grid-template-columns:repeat(2,1fr);}.hero h1{font-size:1.6rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SPINES = ["#1F6F6B", "#16303F", "#8A4B38", "#4A5D8A", "#6B5B2E", "#5A3E6B", "#2F5D3A"]


# ============================== DATA & MODELS ==============================
@st.cache_data
def load_data():
    ratings = pd.read_csv(os.path.join(DATA, "ratings_clean.csv"))
    books = pd.read_csv(os.path.join(DATA, "books_clean.csv")).reset_index(drop=True)
    tags = pd.read_csv(os.path.join(DATA, "tags.csv"))
    book_tags = pd.read_csv(os.path.join(DATA, "book_tags.csv"))
    return ratings, books, tags, book_tags


@st.cache_resource(show_spinner="Building recommendation models. This takes 1-2 minutes on the first run.")
def build_models():
    ratings, books, tags, book_tags = load_data()
    books = books.copy()

    user_ids = ratings["user_id"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    book2idx = {b: i for i, b in enumerate(books["book_id"].values)}
    n_users, n_books = len(user_ids), len(books)

    r = ratings[ratings["book_id"].isin(book2idx)].copy()
    r["u"] = r["user_id"].map(user2idx)
    r["b"] = r["book_id"].map(book2idx)

    # Model 1: Popularity (weighted rating)
    stats = r.groupby("b")["rating"].agg(["mean", "count"]).reset_index()
    C = stats["mean"].mean()
    m = stats["count"].quantile(0.70)
    stats["score"] = (stats["count"] / (stats["count"] + m)) * stats["mean"] + \
                     (m / (stats["count"] + m)) * C
    popularity = stats.sort_values("score", ascending=False).reset_index(drop=True)

    # Model 2: Content-Based (TF-IDF)
    junk = {"to-read", "currently-reading", "favorites", "owned", "books-i-own",
            "default", "library", "kindle", "ebook", "audiobook", "wish-list",
            "my-books", "owned-books", "to-buy", "favourites", "e-book", "read"}
    id_col = "goodreads_book_id" if "goodreads_book_id" in books.columns else "book_id"
    books["tag_text"] = ""
    if books[id_col].isin(book_tags["goodreads_book_id"]).sum() > 0:
        bt = book_tags.merge(tags, on="tag_id")
        bt = bt[~bt["tag_name"].isin(junk)]
        bt = bt.sort_values("count", ascending=False).groupby("goodreads_book_id").head(10)
        tag_str = bt.groupby("goodreads_book_id")["tag_name"].apply(lambda x: " ".join(x))
        books["tag_text"] = books[id_col].map(tag_str).fillna("")
    content = books["title"].fillna("") + " " + books["authors"].fillna("") + " " + books["tag_text"]
    tfidf_matrix = TfidfVectorizer(stop_words="english", max_features=20000,
                                   ngram_range=(1, 2)).fit_transform(content)

    # Model 3: Item-Item KNN
    item_user = csr_matrix((r["rating"].values, (r["b"].values, r["u"].values)),
                           shape=(n_books, n_users))
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=21).fit(item_user)

    # Model 4: SVD
    user_mean = r.groupby("u")["rating"].mean()
    r["centered"] = r["rating"] - r["u"].map(user_mean)
    user_item = csr_matrix((r["centered"].values, (r["u"].values, r["b"].values)),
                           shape=(n_users, n_books))
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(user_item)
    item_factors = svd.components_
    user_mean_arr = np.full(n_users, r["rating"].mean())
    user_mean_arr[user_mean.index.values] = user_mean.values

    return dict(books=books, r=r, user_ids=user_ids, user2idx=user2idx,
                n_users=n_users, n_books=n_books, popularity=popularity,
                tfidf_matrix=tfidf_matrix, item_user=item_user, knn=knn,
                user_factors=user_factors, item_factors=item_factors,
                user_mean_arr=user_mean_arr)


# ============================== RECOMMENDERS ==============================
def rec_popular(M, n):
    p = M["popularity"].head(n)
    return p["b"].tolist(), p["score"].values


def rec_content(M, i, n):
    sims = linear_kernel(M["tfidf_matrix"][i], M["tfidf_matrix"]).ravel()
    top = sims.argsort()[::-1][1:n + 1]
    return top.tolist(), sims[top]


def rec_cf_similar(M, i, n):
    dist, ind = M["knn"].kneighbors(M["item_user"][i], n_neighbors=n + 1)
    return ind[0][1:].tolist(), 1 - dist[0][1:]


def rec_cf_user(M, u, n):
    r = M["r"]
    seen = set(r.loc[r["u"] == u, "b"])
    liked = r[(r["u"] == u) & (r["rating"] >= 4)]["b"].tolist()[:20]
    scores = {}
    for b in liked:
        dist, ind = M["knn"].kneighbors(M["item_user"][b], n_neighbors=21)
        for d, j in zip(dist[0][1:], ind[0][1:]):
            if j not in seen:
                scores[j] = scores.get(j, 0) + (1 - d)
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [j for j, _ in top], np.array([s for _, s in top])


def rec_svd_user(M, u, n):
    r = M["r"]
    seen = set(r.loc[r["u"] == u, "b"])
    pred = M["user_mean_arr"][u] + M["user_factors"][u] @ M["item_factors"]
    order = [j for j in pred.argsort()[::-1] if j not in seen][:n]
    return order, np.clip(pred[order], 1, 5)


# ============================== UI HELPERS ==============================
def render_cards(M, idx_list, scores, score_label, covers=False):
    """Show recommendations as a grid of book cards."""
    if len(idx_list) == 0:
        st.info("No recommendations found. Try a different book or reader.")
        return
    books = M["books"]
    cards = []
    for k, i in enumerate(idx_list):
        row = books.iloc[i]
        title = html.escape(str(row["title"]))
        author = html.escape(str(row["authors"]))
        color = SPINES[sum(map(ord, title)) % len(SPINES)]
        initial = title[:1].upper()
        img = ""
        if covers and isinstance(row.get("image_url"), str):
            img = f'<img src="{html.escape(row["image_url"])}" referrerpolicy="no-referrer">'
        chips = [f'<span class="chip gold">{score_label}: {scores[k]:.2f}</span>']
        if "average_rating" in books.columns:
            chips.append(f'<span class="chip">Avg rating {row["average_rating"]:.2f}</span>')
        if "original_publication_year" in books.columns and pd.notna(row["original_publication_year"]):
            chips.append(f'<span class="chip">{int(row["original_publication_year"])}</span>')
        cards.append(
            f'<div class="card"><div class="cover" style="background:{color}">{initial}{img}</div>'
            f'<div class="meta"><div class="rank">#{k + 1}</div><div class="title">{title}</div>'
            f'<div class="author">{author}</div><div class="chips">{"".join(chips)}</div></div></div>')
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


# ============================== PAGE ==============================
M = build_models()
books = M["books"]
titles = books["title"].tolist()

st.markdown(
    '<div class="hero"><h1>Library Book Recommender</h1>'
    '<p>Discover your next read. This system studies thousands of reader ratings and book details '
    'to suggest books that match your taste, using four different machine learning approaches.</p></div>',
    unsafe_allow_html=True)

st.markdown(
    f'<div class="stats">'
    f'<div class="stat"><b>{M["n_books"]:,}</b><span>Books</span></div>'
    f'<div class="stat"><b>{M["n_users"]:,}</b><span>Readers</span></div>'
    f'<div class="stat"><b>{len(M["r"]):,}</b><span>Ratings</span></div>'
    f'<div class="stat"><b>4</b><span>Algorithms</span></div></div>',
    unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    n = st.slider("Number of recommendations", 3, 15, 9)
    covers = st.checkbox("Show book covers", value=False)
    st.caption("Covers load from the dataset image links and may not be available for every book.")
    st.markdown("---")
    st.caption("Dataset: Goodbooks-10k (Zajac, 2017)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Popular Books", "Find Similar Books", "Personal Recommendations", "Model Comparison", "About"])

# ---------- Tab 1: Popular ----------
with tab1:
    note("The highest rated books that many readers agree on. This is the baseline model, "
         "and it gives the same list to everyone.")
    idx, sc = rec_popular(M, n)
    render_cards(M, idx, sc, "Score", covers)

# ---------- Tab 2: Similar books ----------
with tab2:
    note("Pick a book you enjoyed and get a list of similar ones.")
    c1, c2 = st.columns([3, 2])
    with c1:
        i = st.selectbox("Choose a book (type to search)", range(len(titles)),
                         format_func=lambda k: titles[k])
    with c2:
        method = st.radio("Method", ["Content-Based", "Collaborative"], horizontal=True)
    st.caption("Content-Based compares titles, authors and tags. "
               "Collaborative looks at what other readers with similar ratings enjoyed.")
    if st.button("Get recommendations", key="sim"):
        st.markdown(f'<div class="pick"><span>Because you liked</span><br><b>{html.escape(titles[i])}</b></div>',
                    unsafe_allow_html=True)
        if method == "Content-Based":
            idx, sc = rec_content(M, i, n)
            render_cards(M, idx, sc, "Similarity", covers)
        else:
            idx, sc = rec_cf_similar(M, i, n)
            render_cards(M, idx, sc, "Similarity", covers)

# ---------- Tab 3: Personal ----------
with tab3:
    note("Select a reader to see books they have not read yet, chosen from their past ratings.")
    c1, c2 = st.columns([2, 3])
    with c1:
        uid = st.selectbox("Reader ID", M["user_ids"][:500])
    with c2:
        algo = st.radio("Algorithm", ["SVD (Matrix Factorization)", "Item-Item KNN"], horizontal=True)
    u = M["user2idx"][uid]
    r = M["r"]
    mine = r[(r["u"] == u) & (r["rating"] >= 4)].sort_values("rating", ascending=False).head(5)
    liked_titles = books.iloc[mine["b"].values]["title"].tolist()
    shown = html.escape(", ".join(liked_titles)) if liked_titles else "No highly rated books found"
    st.markdown(f'<div class="pick"><span>Books this reader rated 4 or 5 stars</span><br>{shown}</div>',
                unsafe_allow_html=True)
    if st.button("Get recommendations", key="user"):
        if algo.startswith("SVD"):
            idx, sc = rec_svd_user(M, u, n)
            render_cards(M, idx, sc, "Predicted rating", covers)
        else:
            idx, sc = rec_cf_user(M, u, n)
            if len(idx) == 0:
                st.warning("Not enough data to recommend with KNN for this reader. Try SVD or another reader.")
            else:
                render_cards(M, idx, sc, "Score", covers)

# ---------- Tab 4: Comparison ----------
with tab4:
    note("How the four models performed on held-out test data. Lower is better for RMSE and MAE. "
         "Higher is better for Precision and Recall.")
    csv_path = os.path.join(DATA, "model_comparison.csv")
    if os.path.exists(csv_path):
        st.dataframe(pd.read_csv(csv_path), hide_index=True, width="stretch")
    else:
        st.info("Results file not found. Run the evaluation cells in notebooks/02_model_building.ipynb first.")
    img_path = os.path.join(IMG, "06_model_comparison.png")
    if os.path.exists(img_path):
        st.image(img_path, width="stretch")

# ---------- Tab 5: About ----------
with tab5:
    st.markdown("### How it works")
    st.markdown(
        "- **Popularity:** ranks books by a weighted average rating, so a book with a few perfect ratings "
        "does not beat a book loved by many readers.\n"
        "- **Content-Based:** turns each book's title, author and tags into TF-IDF vectors and finds the "
        "closest matches using cosine similarity.\n"
        "- **Item-Item KNN:** finds books that the same readers rated similarly.\n"
        "- **SVD (Matrix Factorization):** learns hidden taste factors for readers and books to predict "
        "ratings for unread books.")
    st.markdown("### Data")
    st.markdown("Goodbooks-10k: ratings from Goodreads users for the 10,000 most popular books. "
                "This app uses a cleaned sample of 10,000 active readers.")