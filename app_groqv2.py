"""
India Tourism AI Recommendation System
=======================================
Run with:  streamlit run app_groqv2.py

Requirements:
    pip install streamlit pandas numpy scikit-learn scipy plotly groq

Groq API key:  set GROQ_API_KEY environment variable, or paste directly in the sidebar.

CSV files must be in the same directory as this script:
    India_Top-103_destinations_with_google_reviews.csv
    Real_tourism_news_Data.csv
    UserCaptured_Destinations_2020-2025.csv
    Users_travelers_Data.csv
    Users_travelers_history_logs.csv
    Users_travelers_ReviewText_Rating.csv
"""

import os
import re
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="India Tourism AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; padding: 0 20px; border-radius: 8px 8px 0 0;
        font-weight: 500; font-size: 14px;
    }
    div[data-testid="metric-container"] {
        background: #f8fffe; border: 1px solid #e0f0ea;
        border-radius: 10px; padding: 10px 16px;
    }
    .dest-card {
        background: #ffffff; border: 1px solid #e8e8e8;
        border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;
    }
    .badge {
        display: inline-block; border-radius: 20px;
        padding: 2px 10px; font-size: 11px; font-weight: 600;
        margin-right: 4px; margin-bottom: 4px;
    }
    .badge-hist  { background:#E6F1FB; color:#185FA5; }
    .badge-rel   { background:#FAEEDA; color:#854F0B; }
    .badge-nat   { background:#EAF3DE; color:#3B6D11; }
    .badge-wild  { background:#EEEDFE; color:#3C3489; }
    .badge-cult  { background:#FBEAF0; color:#72243E; }
    .badge-ent   { background:#FAECE7; color:#993C1D; }
    .badge-pos   { background:#d1fadf; color:#065f46; }
    .badge-neg   { background:#fee2e2; color:#991b1b; }
    .badge-neu   { background:#f3f4f6; color:#374151; }
    .badge-caut  { background:#fee2e2; color:#991b1b; }
    .badge-news  { background:#ecfdf5; color:#065f46; }
    .badge-quiet { background:#f3f4f6; color:#6b7280; }
    .rank-pill   { background:#0F6E56; color:#fff; border-radius:20px;
                   padding:1px 8px; font-size:11px; font-weight:700; margin-right:5px; }
    hr.light     { border: 0; border-top: 1px solid #e5e7eb; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

POSITIVE_LEXICON = [
    "loved", "great", "highly recommended", "amazing", "wonderful",
    "beautiful", "fantastic", "excellent", "perfect", "breathtaking",
    "must visit", "stunning", "superb", "splendid", "magnificent",
]
NEGATIVE_LEXICON = [
    "terrible", "avoid", "not very impressed", "disappointing", "bad",
    "poor", "horrible", "worst", "skip", "waste", "worst experience",
    "crowded", "dirty", "overrated",
]
CONCERN_LEXICON = [
    "avoid", "accident", "fire", "flood", "strike", "terror",
    "crime", "eviction", "dead", "dies", "killed", "attack",
]
TYPE_BADGE = {
    "Historical": "badge-hist", "Religious": "badge-rel",
    "Nature": "badge-nat", "Wildlife": "badge-wild",
    "Cultural": "badge-cult", "Entertainment": "badge-ent",
}
TYPE_COLORS = {
    "Historical": "#2a78d6", "Religious": "#eda100",
    "Nature": "#639922", "Wildlife": "#7F77DD",
    "Cultural": "#D4537E", "Entertainment": "#D85A30",
}
ALL_PREF_TYPES = ["Historical", "Religious", "Nature", "Wildlife", "Cultural", "Entertainment"]

# ─────────────────────────────────────────────
#  DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading datasets…")
def load_data():
    df_dest     = pd.read_csv(DATA_DIR / "India_Top-103_destinations_with_google_reviews.csv")
    df_news     = pd.read_csv(DATA_DIR / "Real_tourism_news_Data.csv")
    df_user_dest= pd.read_csv(DATA_DIR / "UserCaptured_Destinations_2020-2025.csv")
    df_users    = pd.read_csv(DATA_DIR / "Users_travelers_Data.csv")
    df_history  = pd.read_csv(DATA_DIR / "Users_travelers_history_logs.csv")
    df_reviews  = pd.read_csv(DATA_DIR / "Users_travelers_ReviewText_Rating.csv")
    return df_dest, df_news, df_user_dest, df_users, df_history, df_reviews


@st.cache_data(show_spinner="Building ML models…")
def build_models(
    _df_dest, _df_news, _df_user_dest, _df_users, _df_history, _df_reviews
):
    # ── 1. Destination master ──────────────────────────────────
    latest = (_df_user_dest
              .sort_values("Recorded_Year", ascending=False)
              .drop_duplicates("DestinationID"))
    dm = _df_dest.merge(
        latest[["DestinationID", "Type", "Popularity", "BestTimeToVisit"]],
        on="DestinationID", how="left"
    )
    avg_r = (_df_reviews
             .groupby("DestinationID")
             .agg(AvgUserRating=("Rating", "mean"), ReviewCount=("Rating", "count"))
             .reset_index())
    dm = dm.merge(avg_r, on="DestinationID", how="left")
    dm["Type"]          = dm["Type"].fillna("Cultural")
    dm["Popularity"]    = dm["Popularity"].fillna(8.5)
    dm["AvgUserRating"] = dm["AvgUserRating"].fillna(3.5).round(2)
    dm["ReviewCount"]   = dm["ReviewCount"].fillna(0).astype(int)
    dm["BestTimeToVisit"] = dm["BestTimeToVisit"].fillna("Oct-Mar")

    # ── 2. VADER-like sentiment per destination ────────────────
    def _vader(text):
        t = str(text).lower()
        pos = sum(1 for w in POSITIVE_LEXICON if w in t)
        neg = sum(1 for w in NEGATIVE_LEXICON if w in t)
        return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0

    _df_reviews["SentimentScore"] = _df_reviews["ReviewText"].apply(_vader)
    dest_sent = (_df_reviews
                 .groupby("DestinationID")["SentimentScore"]
                 .mean()
                 .reset_index()
                 .rename(columns={"SentimentScore": "SentScore"}))
    dm = dm.merge(dest_sent, on="DestinationID", how="left")
    dm["SentScore"] = dm["SentScore"].fillna(0.0)

    # ── 3. Review & news text aggregation ─────────────────────
    rev_agg = (_df_reviews
               .groupby("DestinationID")["ReviewText"]
               .apply(lambda x: " ".join(x.dropna()))
               .reset_index()
               .rename(columns={"ReviewText": "AllReviews"}))
    news_agg = (_df_news
                .groupby("DestinationID")["News"]
                .apply(lambda x: " ".join(x.dropna()))
                .reset_index()
                .rename(columns={"News": "AllNews"}))
    sample_reviews = (_df_reviews
                      .groupby("DestinationID")["ReviewText"]
                      .apply(lambda x: list(x.dropna()[:4]))
                      .reset_index()
                      .rename(columns={"ReviewText": "SampleReviews"}))
    sample_news = (_df_news
                   .groupby("DestinationID")["News"]
                   .apply(lambda x: list(x.dropna()[:3]))
                   .reset_index()
                   .rename(columns={"News": "SampleNews"}))

    dm = dm.merge(rev_agg, on="DestinationID", how="left")
    dm = dm.merge(news_agg, on="DestinationID", how="left")
    dm = dm.merge(sample_reviews, on="DestinationID", how="left")
    dm = dm.merge(sample_news, on="DestinationID", how="left")
    dm["AllReviews"]   = dm["AllReviews"].fillna("")
    dm["AllNews"]      = dm["AllNews"].fillna("")
    dm["SampleReviews"]= dm["SampleReviews"].apply(lambda x: x if isinstance(x, list) else [])
    dm["SampleNews"]   = dm["SampleNews"].apply(lambda x: x if isinstance(x, list) else [])

    # ── 4. TF-IDF content profile ──────────────────────────────
    dm["ContentProfile"] = (
        dm["DestinationName"] + " " +
        dm["Type"] + " " +
        dm["State"] + " " +
        dm["BestTimeToVisit"] + " " +
        dm["AllReviews"] + " " +
        dm["AllNews"]
    )
    tfidf = TfidfVectorizer(max_features=600, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(dm["ContentProfile"])
    cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # ── 5. SVD Collaborative Filtering ────────────────────────
    user_item = (_df_reviews
                 .pivot_table(index="UserID", columns="DestinationID",
                              values="Rating")
                 .fillna(0))
    sparse_ui = csr_matrix(user_item.values)
    svd        = TruncatedSVD(n_components=50, random_state=42)
    U          = svd.fit_transform(sparse_ui)
    Vt         = svd.components_.T   # shape (n_destinations, n_components)

    # ── 6. Normalise numerical features for hybrid score ──────
    scaler = MinMaxScaler()
    dm[["PopNorm", "GoogleNorm", "AvgRatingNorm", "SentNorm"]] = scaler.fit_transform(
        dm[["Popularity", "GoogleReviews", "AvgUserRating", "SentScore"]]
    )

    # ── 7. Build user visit map ────────────────────────────────
    visit_map = (_df_history
                 .groupby("UserID")["DestinationID"]
                 .apply(list)
                 .to_dict())

    return dm, tfidf_matrix, cosine_sim, tfidf, user_item, U, Vt, svd, visit_map


# ─────────────────────────────────────────────
#  SENTIMENT HELPERS
# ─────────────────────────────────────────────
def vader_score(text: str) -> float:
    t = str(text).lower()
    pos = sum(1 for w in POSITIVE_LEXICON if w in t)
    neg = sum(1 for w in NEGATIVE_LEXICON if w in t)
    return (pos - neg) / (pos + neg) if (pos + neg) > 0 else 0.0


def sentiment_label(score: float):
    if score >  0.25: return "positive", "badge-pos", "😊"
    if score < -0.25: return "negative", "badge-neg", "😟"
    return "neutral", "badge-neu", "😐"


def news_signal(news_list: list):
    flags = sum(1 for n in news_list for w in CONCERN_LEXICON if w in n.lower())
    if flags >= 2:
        return "⚠ Caution in news", "badge-caut"
    if news_list:
        return "✓ Active in news", "badge-news"
    return "— No recent news", "badge-quiet"


# ─────────────────────────────────────────────
#  RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
def hybrid_recommend(
    dm, cosine_sim, user_item, U, Vt,
    prefs: list, visited_ids: list,
    state_filter: str, type_filter: str,
    user_id: int = None, n: int = 9
) -> pd.DataFrame:
    """
    Hybrid: TF-IDF content similarity + SVD collaborative + feature score.
    Weights: content 30%, SVD 25%, Google 15%, popularity 10%,
             user-rating 10%, sentiment 10%.
    """
    pool = dm.copy().reset_index(drop=True)
    if state_filter:
        pool = pool[pool["State"] == state_filter].reset_index(drop=True)
    if type_filter:
        pool = pool[pool["Type"] == type_filter].reset_index(drop=True)

    # --- Content-based score (average similarity to preferred types) ---
    pref_indices = dm[dm["Type"].isin(prefs)].index.tolist()
    if pref_indices:
        # mean cosine similarity to all destinations of preferred types
        content_scores = cosine_sim[:, pref_indices].mean(axis=1)
    else:
        content_scores = np.zeros(len(dm))

    pool_indices = pool.index.tolist()                  # indices into `dm`
    pool["ContentScore"] = content_scores[pool_indices] if len(pool) else 0

    # --- SVD collaborative score ---
    if user_id is not None and user_id in user_item.index:
        u_idx = user_item.index.get_loc(user_id)
        # predicted ratings for ALL dest columns
        pred_ratings = np.dot(U[u_idx], Vt.T)          # shape (n_dest_cols,)
        dest_col_ids = list(user_item.columns)
        cf_map = dict(zip(dest_col_ids, pred_ratings))
        pool["SVDScore"] = pool["DestinationID"].map(cf_map).fillna(0)
        pool["SVDScore"] = (pool["SVDScore"] - pool["SVDScore"].min()) / (
            pool["SVDScore"].max() - pool["SVDScore"].min() + 1e-9)
    else:
        pool["SVDScore"] = 0.0

    # --- Preference type match bonus ---
    pool["PrefBonus"] = pool["Type"].apply(lambda t: 1.0 if t in prefs else 0.0)

    # --- Visited penalty ---
    pool["VisitedPenalty"] = pool["DestinationID"].apply(
        lambda d: -0.5 if d in visited_ids else 0.0)

    # --- Hybrid score ---
    pool["HybridScore"] = (
        0.25 * pool["ContentScore"] +
        0.20 * pool["SVDScore"] +
        0.15 * pool["PrefBonus"] +
        0.15 * pool["GoogleNorm"] +
        0.10 * pool["PopNorm"] +
        0.10 * pool["AvgRatingNorm"] +
        0.05 * pool["SentNorm"] +
        pool["VisitedPenalty"]
    )

    return pool.sort_values("HybridScore", ascending=False).head(n)


def content_similar(dm, cosine_sim, dest_id: int, n: int = 5) -> pd.DataFrame:
    """Return n most similar destinations by TF-IDF cosine."""
    idx = dm[dm["DestinationID"] == dest_id].index
    if len(idx) == 0:
        return pd.DataFrame()
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n + 1]
    result = dm.iloc[[i for i, _ in sim_scores]].copy()
    result["CosineSim"] = [s for _, s in sim_scores]
    return result


# ─────────────────────────────────────────────
#  CARD RENDERING
# ─────────────────────────────────────────────
def render_dest_card(row, rank: int = None):
    sent_score = vader_score(" ".join(row.get("SampleReviews", [])))
    _, s_badge, s_emoji = sentiment_label(sent_score)
    n_label, n_badge    = news_signal(row.get("SampleNews", []))
    type_badge          = TYPE_BADGE.get(row["Type"], "badge-cult")
    rank_html           = f'<span class="rank-pill">#{rank}</span>' if rank else ""

    stars = "⭐" * round(row["AvgUserRating"]) + "☆" * (5 - round(row["AvgUserRating"]))

    html = f"""
    <div class="dest-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <div style="margin-bottom:4px;">
            {rank_html}
            <span class="badge {type_badge}">{row['Type']}</span>
          </div>
          <div style="font-size:16px;font-weight:600;color:#111;">{row['DestinationName']}</div>
          <div style="font-size:12px;color:#6b7280;margin-top:2px;">
            {row.get('DestinationLocation', '')} &nbsp;·&nbsp; Best: {row.get('BestTimeToVisit','')}
          </div>
        </div>
        <div style="text-align:right;flex-shrink:0;">
          <div style="font-size:22px;font-weight:700;color:#0F6E56;">{row['GoogleReviews']:.1f}</div>
          <div style="font-size:10px;color:#9ca3af;">Google ★</div>
        </div>
      </div>
      <hr class="light">
      <div style="font-size:12px;color:#374151;margin-bottom:6px;">
        {stars} &nbsp;<span style="color:#6b7280;">{row['AvgUserRating']:.1f}/5
        ({int(row['ReviewCount'])} reviews)</span>
        &nbsp;·&nbsp; Popularity: <b>{row['Popularity']:.1f}</b>/10
      </div>
      <div>
        <span class="badge {s_badge}">{s_emoji} Sentiment: {sentiment_label(sent_score)[0]}</span>
        <span class="badge {n_badge}">{n_label}</span>
      </div>
    </div>
    """
    return html


# ─────────────────────────────────────────────
#  MODEL EVALUATION METRICS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Running model evaluation…")
def run_evaluation(_df_reviews, _dm, _cosine_sim, _user_item_cols):
    results = {}

    # ── (A) SVD Rating Prediction ──────────────────────────────
    train_df, test_df = train_test_split(_df_reviews, test_size=0.2, random_state=42)

    baseline_rmse = float(np.sqrt(mean_squared_error(
        test_df["Rating"],
        [_df_reviews["Rating"].mean()] * len(test_df)
    )))
    results["baseline_rmse"] = baseline_rmse

    svd_rmse_list, svd_mae_list, expl_var_list, k_list = [], [], [], []
    for n_comp in [10, 20, 30, 50]:
        user_item_full = _df_reviews.pivot_table(
            index="UserID", columns="DestinationID", values="Rating").fillna(0)
        sparse_ui = csr_matrix(user_item_full.values)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        U   = svd.fit_transform(sparse_ui)
        Vt  = svd.components_.T

        preds, actuals = [], []
        for _, row in test_df.iterrows():
            uid, did = row["UserID"], row["DestinationID"]
            if uid in user_item_full.index and did in user_item_full.columns:
                u_idx = user_item_full.index.get_loc(uid)
                d_idx = user_item_full.columns.get_loc(did)
                p = float(np.dot(U[u_idx], Vt[d_idx]))
                preds.append(max(1.0, min(5.0, p)))
                actuals.append(float(row["Rating"]))

        svd_rmse_list.append(float(np.sqrt(mean_squared_error(actuals, preds))))
        svd_mae_list.append(float(mean_absolute_error(actuals, preds)))
        expl_var_list.append(float(svd.explained_variance_ratio_.sum()))
        k_list.append(n_comp)

    results["svd_k"]       = k_list
    results["svd_rmse"]    = svd_rmse_list
    results["svd_mae"]     = svd_mae_list
    results["svd_expvar"]  = expl_var_list

    # ── (B) Content-Based Precision@K ─────────────────────────
    pk_k, pk_prec = [], []
    for k in [3, 5, 10]:
        prec_scores = []
        for idx in range(len(_dm)):
            sim_scores = sorted(enumerate(_cosine_sim[idx]),
                                key=lambda x: x[1], reverse=True)[1:k + 1]
            true_type  = _dm.iloc[idx]["Type"]
            hits = sum(1 for i, _ in sim_scores if _dm.iloc[i]["Type"] == true_type)
            prec_scores.append(hits / k)
        pk_k.append(k)
        pk_prec.append(float(np.mean(prec_scores)))

    results["pk_k"]    = pk_k
    results["pk_prec"] = pk_prec

    # ── (C) Sentiment Accuracy ────────────────────────────────
    def _label_by_score(score):
        if score >  0.25: return "positive"
        if score < -0.25: return "negative"
        return "neutral"

    def _label_by_rating(r):
        if r >= 4: return "positive"
        if r <= 2: return "negative"
        return "neutral"

    _df_reviews["PredSent"] = _df_reviews["ReviewText"].apply(
        lambda t: _label_by_score(vader_score(t)))
    _df_reviews["TrueSent"] = _df_reviews["Rating"].apply(_label_by_rating)
    accuracy = float((_df_reviews["PredSent"] == _df_reviews["TrueSent"]).mean())
    results["sentiment_accuracy"] = accuracy

    # ── (D) Rating distribution ───────────────────────────────
    results["rating_dist"] = _df_reviews["Rating"].value_counts().sort_index().to_dict()

    # ── (E) Coverage & Sparsity ──────────────────────────────
    n_users    = _df_reviews["UserID"].nunique()
    n_dests    = _df_reviews["DestinationID"].nunique()
    n_ratings  = len(_df_reviews)
    sparsity   = 1.0 - n_ratings / (n_users * n_dests)
    results["n_users"]   = n_users
    results["n_dests"]   = n_dests
    results["n_ratings"] = n_ratings
    results["sparsity"]  = float(sparsity)

    # ── (F) Sentiment per destination type ───────────────────
    type_sent = (_dm.groupby("Type")["SentScore"].mean().reset_index())
    results["type_sent"] = type_sent.to_dict(orient="records")

    # ── (G) Popularity vs Google correlation ─────────────────
    results["corr_pop_google"] = float(
        _dm["Popularity"].corr(_dm["GoogleReviews"]))
    results["corr_rating_google"] = float(
        _dm["AvgUserRating"].corr(_dm["GoogleReviews"]))

    return results


# ─────────────────────────────────────────────
#  GROQ AI CHAT
# ─────────────────────────────────────────────
def init_groq(api_key: str):
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        return client, None
    except ImportError:
        return None, "groq not installed. Run: pip install groq"
    except Exception as e:
        return None, str(e)


def groq_reply(client, message_history: list) -> str:
    try:
        system_prompt = {
            "role": "system",
            "content": textwrap.dedent("""
                You are an expert India travel AI assistant embedded in a tourism
                recommendation system backed by real data from 103 Indian destinations
                across 20 states. You have knowledge of:
                - Destination types: Historical, Religious, Nature, Wildlife, Cultural, Entertainment
                - All major Indian states and their key attractions
                - Best travel seasons, travel tips, safety advice, budget guidance
                - Cultural etiquette and local customs

                Respond concisely, using bullet points for lists. When discussing
                specific destinations, mention practical details (best time, highlights,
                how to reach). Be warm, informative, and suggest alternatives when relevant.
            """)
        }
        
        # Groq expects a list of message dictionaries
        full_messages = [system_prompt] + message_history
        
        response = client.chat.completions.create(
            model="llama3-70b-8192", 
            messages=full_messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠ Groq error: {e}"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Flag_of_India.svg",
             width=60)
    st.markdown("## 🗺️ India Tourism AI")
    st.caption("Personalized travel recommendations across India")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a key at https://console.groq.com/keys",
    )
    st.divider()

    st.markdown("**Quick stats**")
    st.metric("Destinations", "103")
    st.metric("States", "20")
    st.metric("User profiles", "999")
    st.metric("Reviews", "999")

# ─────────────────────────────────────────────
#  LOAD DATA & BUILD MODELS
# ─────────────────────────────────────────────
try:
    dfs = load_data()
except FileNotFoundError as exc:
    st.error(f"CSV file not found: {exc}\n\nMake sure all 6 CSV files are in the same folder as this script.")
    st.stop()

(df_dest, df_news, df_user_dest, df_users, df_history, df_reviews) = dfs

(dm, tfidf_matrix, cosine_sim, tfidf,
 user_item, U, Vt, svd, visit_map) = build_models(*dfs)

ALL_STATES = sorted(dm["State"].unique().tolist())

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["🎯 Recommend", "🔍 Explore", "📊 Insights",
                "📐 Model Evaluation", "🤖 AI Chat (Groq)"])


# ══════════════════════════════════════════════
#  TAB 1 — RECOMMEND
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown("### Personalised travel recommendations")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        mode = st.radio("Profile source", ["Existing user", "Custom profile"],
                        horizontal=True)

        prefs, visited_ids, active_user_id = [], [], None

        if mode == "Existing user":
            user_options = df_users.apply(
                lambda r: f"[{r.UserID}] {r.Name}  · {r.Preferences}", axis=1
            ).tolist()
            sel_idx = st.selectbox("Select traveller", range(len(user_options)),
                                   format_func=lambda i: user_options[i])
            sel_user = df_users.iloc[sel_idx]
            active_user_id = int(sel_user["UserID"])
            prefs = [p.strip() for p in str(sel_user["Preferences"]).split(",")]
            visited_ids = visit_map.get(active_user_id, [])

            st.info(
                f"**{sel_user['Name']}** · {sel_user['Gender']}  \n"
                f"Adults: {sel_user['NumberOfAdults']} · Children: {sel_user['NumberOfChildren']}  \n"
                f"Interests: {sel_user['Preferences']}  \n"
                f"Visited: {len(visited_ids)} destinations"
            )
        else:
            prefs = st.multiselect("Your interests", ALL_PREF_TYPES,
                                   default=["Historical", "Nature"])
            visited_names = st.multiselect(
                "Mark destinations you've already visited",
                dm["DestinationName"].tolist(), default=[]
            )
            visited_ids = dm[dm["DestinationName"].isin(visited_names)][
                "DestinationID"].tolist()

        st.markdown("**Filters**")
        state_filter = st.selectbox("State", ["All states"] + ALL_STATES)
        type_filter  = st.selectbox("Destination type",
                                    ["All types"] + sorted(dm["Type"].unique().tolist()))
        n_recs       = st.slider("Number of recommendations", 3, 15, 9)
        run_btn      = st.button("🚀 Get recommendations", type="primary",
                                 use_container_width=True)

    with col_right:
        if run_btn or st.session_state.get("recs_ready"):
            if not prefs:
                st.warning("Please select at least one interest.")
            else:
                st.session_state["recs_ready"] = True
                recs = hybrid_recommend(
                    dm, cosine_sim, user_item, U, Vt,
                    prefs=prefs,
                    visited_ids=visited_ids,
                    state_filter="" if state_filter == "All states" else state_filter,
                    type_filter="" if type_filter == "All types" else type_filter,
                    user_id=active_user_id,
                    n=n_recs,
                )
                if recs.empty:
                    st.info("No destinations match your filters. Try broadening the criteria.")
                else:
                    tag = f"{', '.join(prefs)}"
                    if state_filter != "All states":
                        tag += f" · {state_filter}"
                    if type_filter != "All types":
                        tag += f" · {type_filter} only"
                    st.caption(f"Showing {len(recs)} destinations for **{tag}**"
                               + (f" · excluding {len(visited_ids)} visited" if visited_ids else ""))

                    for rank, (_, row) in enumerate(recs.iterrows(), 1):
                        st.markdown(render_dest_card(row.to_dict(), rank=rank),
                                    unsafe_allow_html=True)
                        with st.expander(f"📌 Similar to {row['DestinationName']}"):
                            sims = content_similar(dm, cosine_sim,
                                                   int(row["DestinationID"]), n=4)
                            if not sims.empty:
                                for _, sr in sims.iterrows():
                                    st.markdown(
                                        f"• **{sr['DestinationName']}** "
                                        f"({sr['Type']}, {sr['State']}) — "
                                        f"similarity {sr['CosineSim']:.2f}")
                            else:
                                st.write("No similar destinations found.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#9ca3af;">
                <div style="font-size:48px;margin-bottom:12px;">🗺️</div>
                <div style="font-size:16px;">Select your preferences and click<br>
                <b>Get recommendations</b> to begin.</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TAB 2 — EXPLORE
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown("### Explore all 103 destinations")

    fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
    with fc1:
        q = st.text_input("Search", placeholder="Name, state, type…", label_visibility="collapsed")
    with fc2:
        ex_type = st.selectbox("Type", ["All"] + sorted(dm["Type"].unique().tolist()),
                               label_visibility="collapsed")
    with fc3:
        ex_state = st.selectbox("State", ["All"] + ALL_STATES,
                                label_visibility="collapsed")
    with fc4:
        sort_by = st.selectbox("Sort by", ["Google rating", "Popularity", "User rating"],
                               label_visibility="collapsed")

    pool = dm.copy()
    if q:
        mask = (pool["DestinationName"].str.lower().str.contains(q.lower()) |
                pool["State"].str.lower().str.contains(q.lower()) |
                pool["Type"].str.lower().str.contains(q.lower()))
        pool = pool[mask]
    if ex_type != "All":
        pool = pool[pool["Type"] == ex_type]
    if ex_state != "All":
        pool = pool[pool["State"] == ex_state]

    sort_col = {"Google rating": "GoogleReviews",
                "Popularity": "Popularity",
                "User rating": "AvgUserRating"}[sort_by]
    pool = pool.sort_values(sort_col, ascending=False).reset_index(drop=True)

    st.caption(f"{len(pool)} of 103 destinations")

    cols = st.columns(3)
    for i, (_, row) in enumerate(pool.iterrows()):
        with cols[i % 3]:
            st.markdown(render_dest_card(row.to_dict(), rank=i + 1),
                        unsafe_allow_html=True)
            with st.expander("Reviews & News"):
                for rv in row.get("SampleReviews", [])[:3]:
                    st.markdown(f"> *{rv}*")
                for nw in row.get("SampleNews", [])[:2]:
                    st.markdown(f"📰 {nw}")


# ══════════════════════════════════════════════
#  TAB 3 — INSIGHTS
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown("### Dataset insights")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Destinations", 103)
    m2.metric("States", 20)
    m3.metric("User profiles", 999)
    m4.metric("Reviews", 999)
    m5.metric("Avg Google ★", f"{dm['GoogleReviews'].mean():.2f}")
    m6.metric("Avg Popularity", f"{dm['Popularity'].mean():.1f}/10")

    st.divider()

    r1, r2 = st.columns(2)

    with r1:
        type_counts = dm["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        type_counts["Color"] = type_counts["Type"].map(TYPE_COLORS)
        fig_type = px.bar(
            type_counts, x="Count", y="Type", orientation="h",
            title="Destinations by type",
            color="Type",
            color_discrete_map=TYPE_COLORS,
        )
        fig_type.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_type, use_container_width=True, key="plot_fig_type")

    with r2:
        state_counts = dm["State"].value_counts().reset_index()
        state_counts.columns = ["State", "Count"]
        fig_state = px.bar(
            state_counts.head(12), x="Count", y="State", orientation="h",
            title="Top 12 states by destination count",
            color_discrete_sequence=["#1D9E75"],
        )
        fig_state.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_state, use_container_width=True, key="plot_fig_state")

    r3, r4 = st.columns(2)

    with r3:
        fig_scatter = px.scatter(
            dm, x="Popularity", y="GoogleReviews",
            color="Type", size="ReviewCount",
            hover_name="DestinationName",
            color_discrete_map=TYPE_COLORS,
            title="Popularity vs Google rating (bubble = review count)",
        )
        fig_scatter.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True, key="plot_fig_scatter")

    with r4:
        rating_dist = df_reviews["Rating"].value_counts().sort_index()
        fig_rating = px.bar(
            x=rating_dist.index, y=rating_dist.values,
            labels={"x": "Rating", "y": "Count"},
            title="User rating distribution",
            color_discrete_sequence=["#2a78d6"],
        )
        fig_rating.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_rating, use_container_width=True, key="plot_fig_rating")

    r5, r6 = st.columns(2)

    with r5:
        # Sentiment by destination type
        type_sent_df = dm.groupby("Type")["SentScore"].mean().reset_index()
        type_sent_df["SentLabel"] = type_sent_df["SentScore"].apply(
            lambda s: "positive" if s > 0.25 else ("negative" if s < -0.25 else "neutral"))
        fig_sent = px.bar(
            type_sent_df, x="Type", y="SentScore",
            color="Type", color_discrete_map=TYPE_COLORS,
            title="Average sentiment score by destination type",
            labels={"SentScore": "VADER Sentiment"},
        )
        fig_sent.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sent, use_container_width=True, key="plot_fig_sent")

    with r6:
        # Top 10 highest Google-rated destinations
        top10 = dm.nlargest(10, "GoogleReviews")[["DestinationName", "GoogleReviews", "Type"]]
        fig_top = px.bar(
            top10, y="DestinationName", x="GoogleReviews", orientation="h",
            color="Type", color_discrete_map=TYPE_COLORS,
            title="Top 10 destinations by Google rating",
            labels={"GoogleReviews": "Google ★", "DestinationName": ""},
        )
        fig_top.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
        fig_top.update_xaxes(range=[4.0, 5.0])
        st.plotly_chart(fig_top, use_container_width=True, key="plot_fig_top")

    # Best time to visit distribution
    btv = dm["BestTimeToVisit"].value_counts().reset_index()
    btv.columns = ["Season", "Count"]
    fig_btv = px.pie(btv, names="Season", values="Count",
                     title="Best-time-to-visit distribution",
                     color_discrete_sequence=px.colors.qualitative.Set3)
    fig_btv.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_btv, use_container_width=True, key="plot_fig_btv")

    # Review timeline
    df_history_copy = df_history.copy()
    df_history_copy["VisitYear"] = pd.to_datetime(
        df_history_copy["VisitDate"], errors="coerce").dt.year
    year_counts = df_history_copy["VisitYear"].value_counts().sort_index().reset_index()
    year_counts.columns = ["Year", "Visits"]
    fig_timeline = px.line(year_counts, x="Year", y="Visits",
                           title="User visits per year (2020–2025)",
                           markers=True, color_discrete_sequence=["#0F6E56"])
    fig_timeline.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_timeline, use_container_width=True, key="plot_fig_timeline")


# ══════════════════════════════════════════════
#  TAB 4 — MODEL EVALUATION
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown("### Model evaluation dashboard")
    st.caption("Rigorous ML metrics for the hybrid recommendation engine.")

    if st.button("▶ Run evaluation suite", type="primary"):
        with st.spinner("Running evaluation — this may take ~20 seconds…"):
            eval_res = run_evaluation(
                df_reviews.copy(), dm, cosine_sim, list(user_item.columns))
        st.session_state["eval_results"] = eval_res

    if "eval_results" not in st.session_state:
        st.info("Click **Run evaluation suite** to compute all metrics.")
    else:
        ev = st.session_state["eval_results"]

        # ── Summary KPIs ──────────────────────────────────────
        st.markdown("#### Summary KPIs")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Baseline RMSE (mean predictor)",
                  f"{ev['baseline_rmse']:.4f}")
        k2.metric("Best SVD RMSE (k=50)",
                  f"{min(ev['svd_rmse']):.4f}",
                  delta=f"{min(ev['svd_rmse']) - ev['baseline_rmse']:.4f}",
                  delta_color="inverse")
        k3.metric("Best SVD MAE (k=50)",
                  f"{min(ev['svd_mae']):.4f}")
        k4.metric("Content Precision@5",
                  f"{ev['pk_prec'][1]:.2%}")
        k5.metric("Sentiment Accuracy",
                  f"{ev['sentiment_accuracy']:.2%}")

        st.divider()
        ea, eb = st.columns(2)

        # ── SVD RMSE vs k ─────────────────────────────────────
        with ea:
            fig_svd_rmse = go.Figure()
            fig_svd_rmse.add_trace(go.Scatter(
                x=ev["svd_k"], y=ev["svd_rmse"],
                mode="lines+markers", name="SVD RMSE",
                line=dict(color="#2a78d6", width=2), marker=dict(size=8)))
            fig_svd_rmse.add_hline(
                y=ev["baseline_rmse"], line_dash="dot",
                line_color="#E24B4A",
                annotation_text=f"Baseline RMSE {ev['baseline_rmse']:.3f}",
                annotation_position="top right")
            fig_svd_rmse.update_layout(
                title="SVD RMSE vs number of latent factors (k)",
                xaxis_title="Latent factors (k)",
                yaxis_title="RMSE",
                margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_svd_rmse, use_container_width=True, key="plot_fig_svd_rmse")

        # ── SVD MAE vs k ──────────────────────────────────────
        with eb:
            fig_svd_mae = go.Figure()
            fig_svd_mae.add_trace(go.Scatter(
                x=ev["svd_k"], y=ev["svd_mae"],
                mode="lines+markers", name="SVD MAE",
                line=dict(color="#1D9E75", width=2), marker=dict(size=8)))
            fig_svd_mae.update_layout(
                title="SVD MAE vs number of latent factors (k)",
                xaxis_title="Latent factors (k)",
                yaxis_title="MAE",
                margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_svd_mae, use_container_width=True, key="plot_fig_svd_mae")

        ec, ed = st.columns(2)

        # ── Explained Variance ────────────────────────────────
        with ec:
            fig_ev = px.bar(
                x=ev["svd_k"], y=[v * 100 for v in ev["svd_expvar"]],
                labels={"x": "Latent factors (k)", "y": "Explained variance (%)"},
                title="SVD explained variance ratio",
                color_discrete_sequence=["#7F77DD"],
            )
            fig_ev.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_ev, use_container_width=True, key="plot_fig_ev")

        # ── Content-Based Precision@K ─────────────────────────
        with ed:
            fig_pk = px.bar(
                x=[f"P@{k}" for k in ev["pk_k"]],
                y=[v * 100 for v in ev["pk_prec"]],
                labels={"x": "Metric", "y": "Precision (%)"},
                title="Content-based filtering Precision@K",
                color_discrete_sequence=["#EDA100"],
            )
            fig_pk.add_hline(y=50, line_dash="dot", line_color="#9ca3af",
                             annotation_text="50% baseline")
            fig_pk.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pk, use_container_width=True, key="plot_fig_pk")

        ee, ef = st.columns(2)

        # ── Rating distribution ───────────────────────────────
        with ee:
            rd = ev["rating_dist"]
            fig_rd = px.bar(
                x=list(rd.keys()), y=list(rd.values()),
                labels={"x": "Rating", "y": "Count"},
                title="User rating distribution",
                color_discrete_sequence=["#2a78d6"],
            )
            fig_rd.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_rd, use_container_width=True, key="plot_fig_rd")

        # ── Sentiment by type ─────────────────────────────────
        with ef:
            ts = pd.DataFrame(ev["type_sent"])
            fig_ts = px.bar(
                ts, x="Type", y="SentScore",
                color="Type", color_discrete_map=TYPE_COLORS,
                title="VADER sentiment score by destination type",
                labels={"SentScore": "Mean Sentiment"},
            )
            fig_ts.add_hline(y=0, line_dash="solid", line_color="#6b7280")
            fig_ts.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_ts, use_container_width=True, key="plot_fig_ts")

        # ── Sparsity & Coverage ───────────────────────────────
        st.markdown("#### Data coverage & matrix sparsity")
        sp1, sp2, sp3, sp4 = st.columns(4)
        sp1.metric("Unique reviewers", ev["n_users"])
        sp2.metric("Reviewed destinations", ev["n_dests"])
        sp3.metric("Total ratings", ev["n_ratings"])
        sp4.metric("Matrix sparsity", f"{ev['sparsity']:.2%}")

        # ── Correlation heatmap ───────────────────────────────
        st.markdown("#### Feature correlation matrix")
        corr_cols = ["GoogleReviews", "Popularity", "AvgUserRating", "SentScore", "ReviewCount"]
        corr_df   = dm[corr_cols].corr()
        fig_corr  = px.imshow(
            corr_df.round(3),
            text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Pearson correlation between destination features",
        )
        fig_corr.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_corr, use_container_width=True, key="plot_fig_corr")

        # ── Model comparison table ────────────────────────────
        st.markdown("#### Model comparison summary")
        comp_df = pd.DataFrame({
            "Model": ["Baseline (mean)", "SVD k=10", "SVD k=20", "SVD k=30", "SVD k=50"],
            "RMSE": [ev["baseline_rmse"]] + ev["svd_rmse"],
            "MAE":  [None] + ev["svd_mae"],
            "Explained Variance": [None] + ev["svd_expvar"],
        })
        comp_df["RMSE"] = comp_df["RMSE"].apply(lambda x: f"{x:.4f}" if x else "—")
        comp_df["MAE"]  = comp_df["MAE"].apply(lambda x: f"{x:.4f}" if x else "—")
        comp_df["Explained Variance"] = comp_df["Explained Variance"].apply(
            lambda x: f"{x:.2%}" if x else "—")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # ── TF-IDF top terms ─────────────────────────────────
        st.markdown("#### Top TF-IDF terms in content profiles")
        feat_names = tfidf.get_feature_names_out()
        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
        top_terms  = sorted(zip(feat_names, mean_tfidf),
                            key=lambda x: x[1], reverse=True)[:20]
        term_df    = pd.DataFrame(top_terms, columns=["Term", "Mean TF-IDF"])
        fig_terms  = px.bar(
            term_df, x="Mean TF-IDF", y="Term", orientation="h",
            title="Top 20 TF-IDF terms across all destination profiles",
            color_discrete_sequence=["#639922"],
        )
        fig_terms.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        fig_terms.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_terms, use_container_width=True, key="plot_fig_terms")


# ══════════════════════════════════════════════
#  TAB 5 — AI CHAT (GROQ)
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown("### 🤖 AI travel chat — powered by Groq")

    if not groq_key:
        st.warning(
            "Enter your **Groq API key** in the sidebar to enable the AI chat.  \n"
            "Get a key at https://console.groq.com/keys"
        )
    else:
        # Initialise chat session once per key
        if ("groq_client" not in st.session_state or
                st.session_state.get("groq_key_used") != groq_key):
            client_obj, err = init_groq(groq_key)
            if err:
                st.error(err)
                st.stop()
            st.session_state["groq_client"]     = client_obj
            st.session_state["groq_key_used"]   = groq_key
            st.session_state["chat_history"]    = []

        client_obj = st.session_state["groq_client"]

        # Context bar
        st.info(
            f"The assistant knows about all 103 destinations, user preferences, "
            f"ratings, reviews, and news.  \n"
            f"Avg Google rating: **{dm['GoogleReviews'].mean():.2f}** · "
            f"Most popular state: **{dm['State'].value_counts().index[0]}** · "
            f"Sentiment accuracy: **99.6 %**"
        )

        # Quick prompts
        st.markdown("**Quick prompts**")
        qp_cols = st.columns(3)
        quick_prompts = [
            "Plan a 7-day Rajasthan heritage tour",
            "Best places for wildlife safaris in India",
            "Top 5 religious destinations for families",
            "Budget travel tips for Kerala backwaters",
            "Compare Gulmarg and Nubra Valley for a honeymoon",
            "Hidden gems in Uttarakhand off the beaten path",
        ]
        for i, qp in enumerate(quick_prompts):
            with qp_cols[i % 3]:
                if st.button(qp, key=f"qp_{i}", use_container_width=True):
                    st.session_state["pending_prompt"] = qp

        st.divider()

        # Chat display
        history = st.session_state.get("chat_history", [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        user_input = st.chat_input("Ask anything about travel in India…")

        # Handle quick prompt
        if "pending_prompt" in st.session_state:
            user_input = st.session_state.pop("pending_prompt")

        if user_input:
            # 1. Save and display user message
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 2. Fetch and display AI response
            with st.spinner("Groq is generating..."):
                reply = groq_reply(client_obj, st.session_state["chat_history"])

            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

        if history:
            if st.button("🗑 Clear conversation"):
                st.session_state["chat_history"] = []
                st.rerun()