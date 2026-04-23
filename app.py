# app.py
import streamlit as st
import joblib, sqlite3, random, pandas as pd, numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title='Personalization Engine', page_icon='🎬', layout='centered')

# ── Load models & data ───────────────────────────────────────────
@st.cache_resource
def load_models():
    best_svd = joblib.load('models/best_svd.pkl')
    movies   = pd.read_csv('movies.csv')
    ratings  = pd.read_csv('ratings.csv')

    movies['genre_list'] = movies['genres'].str.split('|')
    mlb          = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genre_list'])
    genre_df     = pd.DataFrame(genre_matrix, index=movies['movieId'], columns=mlb.classes_)
    movie_sim_df = pd.DataFrame(
        cosine_similarity(genre_df),
        index=genre_df.index,
        columns=genre_df.index
    )
    return best_svd, movies, ratings, movie_sim_df

best_svd, movies, ratings, movie_sim_df = load_models()

ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]

# ── Database setup ───────────────────────────────────────────────
def init_db():
    con = sqlite3.connect('logs.db')
    con.execute('''CREATE TABLE IF NOT EXISTS events (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   TEXT,
        variant   TEXT,
        movie_id  INTEGER,
        event     TEXT,
        ts        DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    con.commit()
    con.close()

init_db()

def log_event(user_id, variant, movie_id, event):
    con = sqlite3.connect('logs.db')
    con.execute(
        'INSERT INTO events (user_id, variant, movie_id, event) VALUES (?,?,?,?)',
        (user_id, variant, movie_id, event)
    )
    con.commit()
    con.close()

def get_seen_movies(user_id):
    con = sqlite3.connect('logs.db')
    try:
        df = pd.read_sql(
            "SELECT movie_id FROM events WHERE user_id=? AND event IN ('like','dislike')",
            con, params=(str(user_id),)
        )
    except:
        df = pd.DataFrame(columns=['movie_id'])
    con.close()
    return df['movie_id'].tolist()

def is_cold_start(user_id):
    con   = sqlite3.connect('logs.db')
    cur   = con.execute(
        "SELECT COUNT(*) FROM events WHERE user_id=? AND event IN ('like','dislike')",
        (str(user_id),)
    )
    count = cur.fetchone()[0]
    con.close()
    return count < 5

# ── Explanation helpers ──────────────────────────────────────────
def explain_svd(user_id, recommended_movie_id):
    uid = int(user_id)

    # Find users who rated this movie highly
    similar_raters = ratings[
        (ratings['movieId'] == recommended_movie_id) &
        (ratings['rating'] >= 4)
    ]['userId'].tolist()

    # Find overlap — users who also rated the same movies as this user highly
    user_liked = set(ratings[(ratings['userId'] == uid) &
                              (ratings['rating'] >= 4)]['movieId'].tolist())

    best_shared = 0
    for other_uid in similar_raters[:100]:
        other_liked = set(ratings[(ratings['userId'] == other_uid) &
                                   (ratings['rating'] >= 4)]['movieId'].tolist())
        shared = len(user_liked & other_liked)
        if shared > best_shared:
            best_shared = shared

    movie_row  = movies[movies['movieId'] == recommended_movie_id].iloc[0]
    avg_rating = ratings[ratings['movieId'] == recommended_movie_id]['rating'].mean()
    n_raters   = len(similar_raters)

    if best_shared > 0:
        return (f"Users with {best_shared} movies in common with you "
                f"rated this {avg_rating:.1f}⭐ ({n_raters:,} similar users liked this)")
    return f"Highly rated by users who share your taste ({avg_rating:.1f}⭐, {n_raters:,} ratings)"

def explain_content(user_id, recommended_movie_id):
    uid   = int(user_id)
    liked = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4)]['movieId'].tolist()
    if not liked:
        return "Matches your genre preferences"
    rec_genres   = set(movies[movies['movieId'] == recommended_movie_id].iloc[0]['genres'].split('|'))
    best_match, best_overlap = None, 0
    for mid in liked:
        row = movies[movies['movieId'] == mid]
        if row.empty:
            continue
        overlap = len(rec_genres & set(row.iloc[0]['genres'].split('|')))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match   = row.iloc[0]['title']
    return f"Because you liked {best_match}" if best_match else "Matches your genre preferences"

# ── Recommendation functions ─────────────────────────────────────
def get_svd_recs(user_id, n=10):
    rated   = ratings[ratings['userId'] == int(user_id)]['movieId'].tolist()
    seen    = get_seen_movies(user_id)
    unrated = [m for m in movies['movieId'].tolist() if m not in rated and m not in seen]
    preds   = [best_svd.predict(int(user_id), mid) for mid in unrated]
    preds.sort(key=lambda x: x.est, reverse=True)
    results = []
    for pred in preds[:n]:
        row = movies[movies['movieId'] == pred.iid].iloc[0]
        results.append({'id': int(pred.iid), 'title': row['title'], 'genres': row['genres']})
    return results

def get_content_recs(user_id, n=10):
    uid   = int(user_id)
    liked = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4)]['movieId'].tolist()
    if not liked:
        liked = ratings[ratings['userId'] == uid]['movieId'].tolist()
    rated      = ratings[ratings['userId'] == uid]['movieId'].tolist()
    seen       = get_seen_movies(user_id)
    sim_scores = movie_sim_df[liked].mean(axis=1)
    sim_scores = sim_scores.drop(index=[m for m in rated + seen if m in sim_scores.index], errors='ignore')
    top_movies = sim_scores.nlargest(n).index.tolist()
    results = []
    for mid in top_movies:
        row = movies[movies['movieId'] == mid].iloc[0]
        results.append({'id': int(mid), 'title': row['title'], 'genres': row['genres']})
    return results

def get_cold_start_recs(preferred_genres, n=10):
    popularity = ratings.groupby('movieId').size().reset_index(name='count')
    popular    = popularity.sort_values('count', ascending=False).merge(movies, on='movieId')
    mask       = popular['genres'].apply(
        lambda g: any(genre in g.split('|') for genre in preferred_genres)
    )
    results = popular[mask].head(n)
    return [
        {
            'id':          int(row['movieId']),
            'title':       row['title'],
            'genres':      row['genres'],
            'explanation': f"Popular in {', '.join(preferred_genres)}"
        }
        for _, row in results.iterrows()
    ]

# ── Session state init ───────────────────────────────────────────
if 'user_id'  not in st.session_state: st.session_state.user_id  = None
if 'variant'  not in st.session_state: st.session_state.variant  = None
if 'user_type' not in st.session_state: st.session_state.user_type = None
if 'preferred_genres' not in st.session_state: st.session_state.preferred_genres = []
if 'page'     not in st.session_state: st.session_state.page     = 'landing'

# ── Pages ────────────────────────────────────────────────────────

def landing_page():
    st.title('🎬 Personalization Engine')
    st.write('How do you want to start?')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('👤 New User\n\nStart from scratch — pick your favourite genres',
                     use_container_width=True):
            st.session_state.user_id   = str(random.randint(10000, 99999))
            st.session_state.variant   = random.choice(['collaborative', 'content'])
            st.session_state.user_type = 'new'
            st.session_state.page      = 'survey'
            st.rerun()
    with col2:
        if st.button('🎬 Existing User\n\nUse a MovieLens user\'s history',
                     use_container_width=True):
            st.session_state.user_id   = str(random.randint(1, 6040))
            st.session_state.variant   = random.choice(['collaborative', 'content'])
            st.session_state.user_type = 'existing'
            st.session_state.page      = 'home'
            st.rerun()


def survey_page():
    st.title('👋 Welcome!')
    st.write('Pick your favourite genres to get started:')
    selected = []
    cols = st.columns(3)
    for i, genre in enumerate(ALL_GENRES):
        if cols[i % 3].checkbox(genre):
            selected.append(genre)
    if st.button('Get Recommendations →', type='primary'):
        if not selected:
            st.error('Please select at least one genre!')
        else:
            st.session_state.preferred_genres = selected
            st.session_state.page             = 'home'
            st.rerun()


def home_page():
    user_id   = st.session_state.user_id
    variant   = st.session_state.variant
    user_type = st.session_state.user_type

    st.title('🎬 Movie Recommendations')
    col1, col2, col3 = st.columns([2, 2, 1])
    col1.metric('User ID', user_id)
    col2.metric('Variant', variant)
    if col3.button('Reset'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Cold start check
    if user_type == 'new' and is_cold_start(user_id):
        preferred = st.session_state.preferred_genres
        if not preferred:
            st.session_state.page = 'survey'
            st.rerun()
        st.info('🌱 Based on your genre preferences — like/dislike movies to get personalised recommendations after 5 interactions!')
        recs = get_cold_start_recs(preferred)
        for rec in recs:
            log_event(user_id, variant, rec['id'], 'impression')
    else:
        if variant == 'collaborative':
            recs = get_svd_recs(user_id)
            for rec in recs:
                rec['explanation'] = explain_svd(user_id, rec['id'])
        else:
            recs = get_content_recs(user_id)
            for rec in recs:
                rec['explanation'] = explain_content(user_id, rec['id'])
        for rec in recs:
            log_event(user_id, variant, rec['id'], 'impression')

    # Render cards
    for rec in recs:
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{rec['title']}**")
                st.caption(rec['genres'])
                if rec.get('explanation'):
                    st.caption(f"💡 {rec['explanation']}")
            with col2:
                col_a, col_b = st.columns(2)
                if col_a.button('👍', key=f"like_{rec['id']}"):
                    log_event(user_id, variant, rec['id'], 'like')
                    st.session_state.variant = 'content' if variant == 'collaborative' else 'collaborative'
                    st.toast(f"Liked! Switching to {st.session_state.variant}")
                    st.rerun()
                if col_b.button('👎', key=f"dislike_{rec['id']}"):
                    log_event(user_id, variant, rec['id'], 'dislike')
                    st.session_state.variant = 'content' if variant == 'collaborative' else 'collaborative'
                    st.toast(f"Disliked! Switching to {st.session_state.variant}")
                    st.rerun()

def results_page():
    st.title('📊 Interleaved Test Results')
    con = sqlite3.connect('logs.db')
    try:
        df = pd.read_sql('SELECT * FROM events', con)
    except:
        df = pd.DataFrame()
    con.close()

    if df.empty:
        st.warning('No data yet — go interact with some recommendations first!')
        return

    summary = {}
    for variant in ['collaborative', 'content']:
        v           = df[df['variant'] == variant]
        impressions = len(v[v['event'] == 'impression'])
        likes       = len(v[v['event'] == 'like'])
        dislikes    = len(v[v['event'] == 'dislike'])
        summary[variant] = {
            'impressions': impressions,
            'likes':       likes,
            'dislikes':    dislikes,
            'like_rate':   round(likes / impressions, 4) if impressions > 0 else 0
        }

    col1, col2 = st.columns(2)
    for col, variant in zip([col1, col2], ['collaborative', 'content']):
        with col:
            s = summary[variant]
            st.subheader(variant.capitalize())
            st.metric('Like Rate',   f"{s['like_rate']:.2%}")
            st.metric('Impressions', s['impressions'])
            st.metric('Likes',       s['likes'])
            st.metric('Dislikes',    s['dislikes'])

    # Bar chart
    chart_df = pd.DataFrame({
        'Variant':   list(summary.keys()),
        'Like Rate': [v['like_rate'] for v in summary.values()]
    })
    st.bar_chart(chart_df.set_index('Variant'))

   # ── Statistical significance test ────────────────────────────
    st.subheader('📈 Model Comparison')
    from scipy import stats

    collab  = summary['collaborative']
    content = summary['content']

    if collab['impressions'] == 0 or content['impressions'] == 0:
        st.info('No data yet for one or both variants.')
    else:
        # Chi-square test
        contingency = [
            [collab['likes'],  collab['impressions']  - collab['likes']],
            [content['likes'], content['impressions'] - content['likes']]
        ]
        chi2, p, _, _ = stats.chi2_contingency(contingency)
        winner = 'Collaborative' if collab['like_rate'] > content['like_rate'] else 'Content'

        # Bayesian win probability using Beta distribution
        # Beta(likes+1, dislikes+impressions+1) models the like rate
        collab_beta  = stats.beta(collab['likes'] + 1,
                                   collab['impressions'] - collab['likes'] + 1)
        content_beta = stats.beta(content['likes'] + 1,
                                   content['impressions'] - content['likes'] + 1)

        # Monte Carlo sample to estimate P(collaborative > content)
        samples      = 100_000
        collab_samp  = collab_beta.rvs(samples)
        content_samp = content_beta.rvs(samples)
        win_prob     = (collab_samp > content_samp).mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Chi-square',   f"{chi2:.3f}")
        col2.metric('p-value',      f"{p:.4f}")
        col3.metric('Winner',       winner)
        col4.metric('Win Probability', f"{win_prob:.1%}")

        if win_prob >= 0.95:
            st.success(f'✅ **{winner}** filtering is better with {win_prob:.1%} probability.')
        elif win_prob >= 0.80:
            st.info(f'📊 **{winner}** filtering is likely better ({win_prob:.1%} probability) — keep collecting data.')
        else:
            st.warning(f'⚠️ Too close to call — {winner} leads with only {win_prob:.1%} probability.')
# ── Navigation ───────────────────────────────────────────────────
with st.sidebar:
    st.title('Navigation')
    if st.button('🏠 Home'):
        st.session_state.page = 'home' if st.session_state.user_id else 'landing'
        st.rerun()
    if st.button('📊 Results'):
        st.session_state.page = 'results'
        st.rerun()
    if st.button('🔄 Reset Session'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Router ───────────────────────────────────────────────────────
page = st.session_state.get('page', 'landing')
if page == 'landing':
    landing_page()
elif page == 'survey':
    survey_page()
elif page == 'home':
    if st.session_state.user_id is None:
        st.session_state.page = 'landing'
        st.rerun()
    home_page()
elif page == 'results':
    results_page()