# app.py
import streamlit as st
import joblib, sqlite3, random, pandas as pd, numpy as np, os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title='Personalization Engine', page_icon='🎬', layout='centered')

# ── PyTorch model classes ────────────────────────────────────────
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    class AttentionScorer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.W_q = nn.Linear(dim, dim, bias=False)
            self.W_k = nn.Linear(dim, dim, bias=False)
            self.scale = dim ** 0.5
        def forward(self, user_emb, movie_emb):
            q = self.W_q(user_emb)
            k = self.W_k(movie_emb)
            return (q * k).sum(dim=-1) / self.scale

    class WideAndDeepNCF(nn.Module):
        def __init__(self, n_users, n_movies, n_genres,
                     embed_dim=64, mlp_layers=(256, 128, 64), dropout=0.3):
            super().__init__()
            self.user_emb = nn.Embedding(n_users, embed_dim)
            self.movie_emb = nn.Embedding(n_movies, embed_dim)
            self.user_bias = nn.Embedding(n_users, 1)
            self.movie_bias = nn.Embedding(n_movies, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))
            layers, in_dim = [], embed_dim * 2
            for out_dim in mlp_layers:
                layers += [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)]
                in_dim = out_dim
            self.deep_mlp = nn.Sequential(*layers)
            self.deep_out = nn.Linear(mlp_layers[-1], 1)
            self.wide_linear = nn.Linear(n_genres, 1, bias=True)
            self.attention = AttentionScorer(embed_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        def forward(self, user, movie, genres, return_attention=False):
            u = self.user_emb(user)
            m = self.movie_emb(movie)
            ub = self.user_bias(user).squeeze(1)
            mb = self.movie_bias(movie).squeeze(1)
            deep_score = self.deep_out(self.deep_mlp(torch.cat([u, m], dim=1))).squeeze(1)
            wide_score = self.wide_linear(genres).squeeze(1)
            attn = self.attention(u, m)
            a = torch.sigmoid(self.alpha)
            raw = a * deep_score + (1 - a) * wide_score + ub + mb + self.global_bias
            rating = torch.sigmoid(raw) * 4 + 1
            cosine_target = (nn.functional.cosine_similarity(u.detach(), m.detach(), dim=1) + 1) / 2
            if return_attention:
                return rating, attn
            return rating, cosine_target, attn

    class BERT4RecMax(nn.Module):
        def __init__(self, vocab_size, max_seq_len, hidden_dim,
                     n_layers, n_heads, ffn_dim, dropout, pad_token=0):
            super().__init__()
            self.pad_token = pad_token
            self.hidden_dim = hidden_dim
            self.item_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token)
            self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
            self.emb_norm = nn.LayerNorm(hidden_dim)
            self.emb_dropout = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=n_heads, dim_feedforward=ffn_dim,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.output_bias = nn.Parameter(torch.zeros(vocab_size))
            self._init_weights()
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.padding_idx is not None:
                        m.weight.data[m.padding_idx].zero_()
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        def forward(self, input_ids):
            B, L = input_ids.shape
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
            x = self.item_emb(input_ids) + self.pos_emb(pos)
            x = self.emb_dropout(self.emb_norm(x))
            pad_mask = (input_ids == self.pad_token)
            x = self.transformer(x, src_key_padding_mask=pad_mask)
            x_norm = self.output_norm(x)
            logits = x_norm @ self.item_emb.weight.T + self.output_bias
            return logits
    TORCH_AVAILABLE = True
except ImportError:
    pass

# ── Load models & data ───────────────────────────────────────────
@st.cache_resource
def load_models():
    best_svd = joblib.load('models/best_svd.pkl')
    content_data = joblib.load('models/content_recommender.joblib')
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    cosine_sim = content_data['cosine_sim']
    cos_sim = content_data['cos_sim']
    movie_idx = content_data['movie_idx']
    movies['genre_list'] = movies['genres'].str.split('|')
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(movies['genre_list'])
    genre_df = pd.DataFrame(genre_matrix, index=movies['movieId'], columns=mlb.classes_)
    return best_svd, content_data, movies, ratings, cosine_sim, cos_sim, movie_idx, genre_df

@st.cache_resource
def load_ncf():
    if not TORCH_AVAILABLE: return None
    checkpoint_path = 'models/ncf_model_checkpoint_v4.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'models/ncf_model_checkpoint_v4.pkt'
    if not os.path.exists(checkpoint_path): return None
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model = WideAndDeepNCF(cfg['n_users'], cfg['n_movies'], cfg['n_genres'],
                               cfg['embed_dim'], cfg['mlp_layers'], cfg['dropout']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt['encoders']['user_enc'], ckpt['encoders']['movie_enc'], \
               ckpt['genre_matrix'], ckpt['pop_array'], ckpt['all_genres'], device
    except Exception:
        return None

@st.cache_resource
def load_bert4rec():
    if not TORCH_AVAILABLE: return None
    checkpoint_path = 'models/bert4rec_max_checkpoint.pt'
    if not os.path.exists(checkpoint_path): return None
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg = ckpt['config']
        model = BERT4RecMax(cfg['vocab_size'], cfg['max_seq_len'], cfg['hidden_dim'],
                            cfg['n_layers'], cfg['n_heads'], cfg['ffn_dim'],
                            cfg['dropout'], cfg['pad_token']).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model, ckpt['encoders']['user_enc'], ckpt['encoders']['movie_enc'], \
               ckpt['sequences']['train_seqs'], cfg['mask_token'], cfg['max_seq_len'], cfg['n_movies'], device
    except Exception:
        return None

best_svd, content_data, movies, ratings, cosine_sim, cos_sim, movie_idx, genre_df = load_models()
ncf_bundle = load_ncf()
bert_bundle = load_bert4rec()
NCF_LOADED = ncf_bundle is not None
BERT_LOADED = bert_bundle is not None

ALL_GENRES = sorted(set(g for genres in movies['genre_list'] for g in genres))
VARIANTS = ['collaborative', 'content']
if NCF_LOADED: VARIANTS.append('ncf')
if BERT_LOADED: VARIANTS.append('bert4rec')
VARIANT_LABELS = {'collaborative': 'SVD Collab', 'content': 'Content', 'ncf': 'NCF Neural', 'bert4rec': 'BERT4Rec'}

# ── Database ─────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect('logs.db')
    con.execute('''CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, variant TEXT, movie_id INTEGER, event TEXT, ts DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    con.commit()
    con.close()
init_db()

def log_event(user_id, variant, movie_id, event):
    con = sqlite3.connect('logs.db')
    con.execute('INSERT INTO events (user_id, variant, movie_id, event) VALUES (?,?,?,?)',
                (user_id, variant, movie_id, event))
    con.commit()
    con.close()

def get_seen_movies(user_id):
    con = sqlite3.connect('logs.db')
    try:
        df = pd.read_sql("SELECT movie_id FROM events WHERE user_id=? AND event IN ('like','dislike')",
                         con, params=(str(user_id),))
    except:
        df = pd.DataFrame(columns=['movie_id'])
    con.close()
    return df['movie_id'].tolist()

def is_cold_start(user_id):
    con = sqlite3.connect('logs.db')
    cur = con.execute("SELECT COUNT(*) FROM events WHERE user_id=? AND event IN ('like','dislike')", (str(user_id),))
    count = cur.fetchone()[0]
    con.close()
    return count < 5

# ── Explanation helpers for warm users ───────────────────────────
def explain_svd(user_id, movie_id):
    uid = int(user_id)
    similar = ratings[(ratings['movieId'] == movie_id) & (ratings['rating'] >= 4)]['userId'].tolist()
    user_liked = set(ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4)]['movieId'].tolist())
    best_shared = 0
    for other in similar[:100]:
        other_liked = set(ratings[(ratings['userId'] == other) & (ratings['rating'] >= 4)]['movieId'].tolist())
        shared = len(user_liked & other_liked)
        if shared > best_shared: best_shared = shared
    avg = ratings[ratings['movieId'] == movie_id]['rating'].mean()
    n = len(similar)
    if best_shared > 0:
        return f"Users with {best_shared} movies in common with you rated this {avg:.1f}⭐ ({n:,} similar users liked this)"
    return f"Highly rated by users who share your taste ({avg:.1f}⭐, {n:,} ratings)"

def explain_content(user_id, movie_id):
    uid = int(user_id)
    liked = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 4)]['movieId'].tolist()
    if not liked:
        return "Matches your genre preferences"
    rec_genres = set(movies[movies['movieId'] == movie_id].iloc[0]['genre_list'])
    best_match, best_overlap = None, 0
    for mid in liked:
        row = movies[movies['movieId'] == mid]
        if row.empty: continue
        overlap = len(rec_genres & set(row.iloc[0]['genre_list']))
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = row.iloc[0]['title']
    return f"Because you liked {best_match}" if best_match else "Matches your genre preferences"

# ── NCF helpers (no model names) ─────────────────────────────────
def _ncf_candidates(user_idx, n_movies, seen):
    mask = np.ones(n_movies, dtype=bool)
    for s in seen:
        if s < n_movies: mask[int(s)] = False
    return np.where(mask)[0]

def get_ncf_recs(user_original_id, n=10):
    if not NCF_LOADED: return []
    model, user_enc, movie_enc, genre_mat, pop_arr, all_genres, device = ncf_bundle
    if user_original_id not in user_enc.classes_:
        pref = st.session_state.get('preferred_genres', ['Drama'])
        return get_ncf_cold_start_recs(pref, n, user_original_id)
    try:
        user_idx = int(user_enc.transform([user_original_id])[0])
        n_movies = len(movie_enc.classes_)
        seen = set()
        for mid in ratings[ratings['userId'] == int(user_original_id)]['movieId']:
            if mid in movie_enc.classes_:
                seen.add(int(movie_enc.transform([mid])[0]))
        for mid in get_seen_movies(user_original_id):
            if mid in movie_enc.classes_:
                seen.add(int(movie_enc.transform([mid])[0]))
        cand = _ncf_candidates(user_idx, n_movies, seen)
        if len(cand) == 0: return []
        c_t = torch.LongTensor(cand).to(device)
        u_t = torch.LongTensor([user_idx] * len(cand)).to(device)
        g_t = torch.FloatTensor(genre_mat[cand]).to(device)
        with torch.no_grad():
            preds, attn = model(u_t, c_t, g_t, return_attention=True)
        preds = preds.cpu().numpy()
        attn = torch.sigmoid(attn).cpu().numpy()
        top = np.argsort(preds)[::-1][:n]
        wide_w = model.wide_linear.weight.squeeze().detach().cpu().numpy()
        results = []
        for i in top:
            midx = cand[i]
            orig_id = movie_enc.classes_[midx]
            row = movies[movies['movieId'] == orig_id]
            if row.empty: continue
            row = row.iloc[0]
            gv = genre_mat[midx]
            top_genre = next((all_genres[j] for j in np.argsort(gv * wide_w)[::-1] if gv[j] > 0), 'various')
            results.append({
                'id': int(orig_id), 'title': row['title'], 'genres': row['genres'],
                'predicted_rating': round(float(preds[i]), 2),
                'attention_score': round(float(attn[i]), 3),
                'explanation': f"Aligns with your taste for {top_genre}"
            })
        return results
    except Exception:
        return []

# ── BERT4Rec helpers ─────────────────────────────────────────────
def _bert_pad(seq, max_len, pad=0):
    return seq[-max_len:] if len(seq) > max_len else seq + [pad] * (max_len - len(seq))
def _bert_candidates(uid, train_seqs, n_movies):
    all_items = np.arange(1, n_movies+1, dtype=np.int64)
    seen = set(train_seqs.get(uid, []))
    return all_items[~np.isin(all_items, list(seen))]

def get_bert4rec_recs(user_original_id, n=10):
    if not BERT_LOADED: return []
    model, user_enc, movie_enc, train_seqs, mask_token, max_len, n_movies, device = bert_bundle
    if user_original_id not in user_enc.classes_:
        pref = st.session_state.get('preferred_genres', ['Drama'])
        return get_bert4rec_cold_start_recs(pref, n, user_original_id)
    try:
        uid = int(user_enc.transform([user_original_id])[0])
        if uid not in train_seqs:
            pref = st.session_state.get('preferred_genres', ['Drama'])
            return get_bert4rec_cold_start_recs(pref, n, user_original_id)
        hist = _bert_pad(train_seqs[uid], max_len-1, 0)
        inp = torch.LongTensor(hist + [mask_token]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)[0, -1, :].cpu().numpy()
        cand = _bert_candidates(uid, train_seqs, n_movies)
        seen_tokens = set()
        for mid in get_seen_movies(user_original_id):
            if mid in movie_enc.classes_:
                seen_tokens.add(int(movie_enc.transform([mid])[0]) + 1)
        if seen_tokens:
            cand = cand[~np.isin(cand, list(seen_tokens))]
        if len(cand) == 0: return []
        scores = logits[cand]
        top = np.argsort(scores)[::-1][:n]
        results = []
        for i in top:
            token = int(cand[i])
            orig_id = movie_enc.classes_[token-1]
            row = movies[movies['movieId'] == orig_id]
            if row.empty: continue
            row = row.iloc[0]
            results.append({
                'id': int(orig_id), 'title': row['title'], 'genres': row['genres'],
                'bert4rec_score': round(float(scores[i]), 4),
                'explanation': "Predicted next based on your watch sequence"
            })
        return results
    except Exception:
        return []

def get_liked_movies(user_id):
    """Return list of movieId that the user has liked (event='like')."""
    con = sqlite3.connect('logs.db')
    try:
        df = pd.read_sql("SELECT movie_id FROM events WHERE user_id=? AND event='like'",
                         con, params=(str(user_id),))
    except:
        df = pd.DataFrame(columns=['movie_id'])
    con.close()
    return df['movie_id'].tolist()

# ── Cold‑start helpers (strictly exclude seen movies, always return n items) ──
def get_cold_start_recs(preferred_genres, n=10, user_id=None):
    """
    SVD collaborative cold start.
    - If user has >=2 liked movies, find similar users based on those movies,
      then restrict candidate movies to those matching preferred genres.
    - Otherwise, fall back to genre-filtered popularity (no collaborative filtering).
    """
    liked_movies = get_liked_movies(user_id) if user_id else []
    seen = set(get_seen_movies(user_id)) if user_id else set()
    pref_set = set(preferred_genres) if preferred_genres else set()

    # Helper: get popularity-based recommendations with genre filter
    def get_popularity_recs(genre_filter=True):
        pop = ratings.groupby('movieId').size().reset_index(name='count')
        pop_movies = pop.merge(movies, on='movieId')
        if genre_filter and pref_set:
            pop_movies = pop_movies[pop_movies['genres'].apply(lambda g: any(p in g.split('|') for p in pref_set))]
        pop_movies = pop_movies[~pop_movies['movieId'].isin(seen)]
        pop_movies = pop_movies.sort_values('count', ascending=False).head(n)
        results = []
        for _, row in pop_movies.iterrows():
            explanation = f"Popular among all users ({int(row['count'])} ratings)"
            if genre_filter and pref_set:
                explanation += f" – matches your genres"
            results.append({
                'id': int(row['movieId']),
                'title': row['title'],
                'genres': row['genres'],
                'explanation': explanation
            })
        return results

    # If not enough liked movies, use genre-filtered popularity
    if len(liked_movies) < 2:
        return get_popularity_recs(genre_filter=True)

    # Step 1: Find users who rated ANY of the liked movies highly (>=4)
    genre_lovers = ratings.merge(movies, on='movieId')
    similar_users = genre_lovers[
        (genre_lovers['movieId'].isin(liked_movies)) &
        (genre_lovers['rating'] >= 4)
    ]['userId'].unique()

    if len(similar_users) == 0:
        return get_popularity_recs(genre_filter=True)

    # Step 2: Candidate movies = highly rated by similar users, not seen, AND match preferred genres (if any)
    candidate_df = genre_lovers[
        genre_lovers['userId'].isin(similar_users[:50]) &
        (genre_lovers['rating'] >= 4) &
        (~genre_lovers['movieId'].isin(seen))
    ]
    if pref_set:
        candidate_df = candidate_df[candidate_df['genres'].apply(lambda g: any(p in g.split('|') for p in pref_set))]
    candidate_movies = candidate_df['movieId'].unique()

    if len(candidate_movies) == 0:
        # No movies after genre filtering – fall back to genre-filtered popularity
        return get_popularity_recs(genre_filter=True)

    # Step 3: Use SVD to predict ratings for candidate movies (average over similar users)
    movie_scores = defaultdict(list)
    for uid in similar_users[:50]:
        for mid in candidate_movies:
            pred = best_svd.predict(uid, mid)
            movie_scores[mid].append(pred.est)
    avg_scores = {mid: np.mean(scores) for mid, scores in movie_scores.items()}
    top_movies = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:n]

    results = []
    for mid, score in top_movies:
        row = movies[movies['movieId'] == mid].iloc[0]
        explanation = f"Users who liked similar movies rated this {score:.2f}⭐"
        # Optionally mention matching genre if desired
        # if pref_set:
        #     match = ', '.join(set(row['genres'].split('|')) & pref_set)
        #     explanation += f" (matches your {match})"
        results.append({
            'id': int(mid),
            'title': row['title'],
            'genres': row['genres'],
            'explanation': explanation
        })
    return results

def get_content_cold_start_recs(preferred_genres, n=10, user_id=None):
    """
    Content cold start:
    - If user has liked movies, use item-item similarity from content model.
    - Otherwise, fall back to genre matching.
    """
    liked_movies = get_liked_movies(user_id) if user_id else []
    seen = set(get_seen_movies(user_id)) if user_id else set()

    if liked_movies:
        # Use content-based similarity from the trained content model
        movie_ids = content_data['movies']['movieId'].tolist()
        id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
        idx_to_id = {i: mid for i, mid in enumerate(movie_ids)}
        # Get indices of liked movies that exist in the content model
        liked_indices = [id_to_idx[m] for m in liked_movies if m in id_to_idx]
        if liked_indices:
            # Average similarity scores from liked movies
            sim_scores = cosine_sim[liked_indices].mean(axis=0)
            # Remove already seen movies
            for idx, mid in idx_to_id.items():
                if mid in seen:
                    sim_scores[idx] = -np.inf
            top_indices = np.argsort(-sim_scores)[:n]
            results = []
            for idx in top_indices:
                mid = idx_to_id[idx]
                row = movies[movies['movieId'] == mid].iloc[0]
                # Find which liked movie is most similar (optional)
                results.append({
                    'id': int(mid), 'title': row['title'], 'genres': row['genres'],
                    'explanation': f"Because you liked similar movies"
                })
            if results:
                return results
        # Fall through to genre-based if content similarity fails

    # Genre-based fallback (original logic, but exclude seen)
    pref_set = set(preferred_genres) if preferred_genres else set()
    if not pref_set:
        # If no genres either, return popular unseen
        pop = ratings.groupby('movieId').size().reset_index(name='count')
        pop_movies = pop.merge(movies, on='movieId')
        pop_movies = pop_movies[~pop_movies['movieId'].isin(seen)]
        pop_movies = pop_movies.sort_values('count', ascending=False).head(n)
        results = []
        for _, row in pop_movies.iterrows():
            results.append({
                'id': int(row['movieId']), 'title': row['title'], 'genres': row['genres'],
                'explanation': f"Popular recommendation"
            })
        return results

    popularity = ratings.groupby('movieId').size().to_dict()
    matched = []
    for _, row in movies.iterrows():
        mid = row['movieId']
        if mid in seen:
            continue
        genres = set(row['genres'].split('|'))
        overlap = genres & pref_set
        if not overlap:
            continue
        score = len(overlap) + 0.001 * popularity.get(mid, 0)
        matched.append((mid, score, overlap))
    if not matched:
        # pure popularity fallback
        pop = ratings.groupby('movieId').size().reset_index(name='count')
        pop_movies = pop.merge(movies, on='movieId')
        pop_movies = pop_movies[~pop_movies['movieId'].isin(seen)]
        pop_movies = pop_movies.sort_values('count', ascending=False).head(n)
        results = []
        for _, row in pop_movies.iterrows():
            results.append({
                'id': int(row['movieId']), 'title': row['title'], 'genres': row['genres'],
                'explanation': f"Popular recommendation (no genre match for {', '.join(preferred_genres)})"
            })
        return results

    matched.sort(key=lambda x: x[1], reverse=True)
    results = []
    for mid, score, overlap in matched[:n]:
        overlap_str = ', '.join(overlap)
        results.append({
            'id': int(mid), 'title': movies[movies['movieId']==mid].iloc[0]['title'],
            'genres': movies[movies['movieId']==mid].iloc[0]['genres'],
            'explanation': f"Because you selected {overlap_str}"
        })
    return results

def get_ncf_cold_start_recs(preferred_genres, n=10, user_id=None):
    """
    NCF cold start: purely genre-based cosine similarity + popularity.
    Shows similarity percentage instead of genre name.
    """
    if not NCF_LOADED:
        return []
    seen = set(get_seen_movies(user_id)) if user_id else set()
    _, user_enc, movie_enc, genre_mat, pop_arr, all_genres, _ = ncf_bundle
    pref_set = set(preferred_genres) if preferred_genres else set()

    # If no genres selected, fall back to SVD cold start
    if not pref_set:
        return get_cold_start_recs(preferred_genres, n, user_id)

    pref_vec = np.array([1.0 if g in pref_set else 0.0 for g in all_genres], dtype=np.float32)
    pref_norm = np.linalg.norm(pref_vec)
    if pref_norm == 0:
        return get_cold_start_recs(preferred_genres, n, user_id)

    scores = []
    for midx in range(len(movie_enc.classes_)):
        orig_id = movie_enc.classes_[midx]
        if orig_id in seen:
            continue
        gv = genre_mat[midx]
        gv_norm = np.linalg.norm(gv)
        if gv_norm == 0:
            continue
        sim = float(np.dot(pref_vec, gv) / (pref_norm * gv_norm))
        if sim < 0.02:
            continue
        final = sim + 0.15 * float(pop_arr[midx])
        scores.append((midx, sim, final))

    if not scores:
        return get_cold_start_recs(preferred_genres, n, user_id)

    scores.sort(key=lambda x: x[2], reverse=True)
    results = []
    for midx, sim, _ in scores[:n]:
        orig_id = movie_enc.classes_[midx]
        row = movies[movies['movieId'] == orig_id]
        if row.empty:
            continue
        row = row.iloc[0]
        results.append({
            'id': int(orig_id),
            'title': row['title'],
            'genres': row['genres'],
            'explanation': f"Genre match: {sim:.2%} + popularity boost"   # numeric similarity
        })
    return results

def get_bert4rec_cold_start_recs(preferred_genres, n=10, user_id=None):
    """
    BERT4Rec cold start:
    - If user has liked movies, build a sequence from those liked movies (in timestamp order) and predict next.
    - Otherwise, fall back to content-based genre matching.
    """
    if not BERT_LOADED:
        return []
    liked_movies = get_liked_movies(user_id) if user_id else []
    seen = set(get_seen_movies(user_id)) if user_id else set()
    model, user_enc, movie_enc, train_seqs, mask_token, max_len, n_movies, device = bert_bundle

    # If user has liked movies, try to use them as a sequence
    if liked_movies:
        # Map liked movies to token IDs (1-based)
        seq_tokens = []
        for mid in liked_movies:
            if mid in movie_enc.classes_:
                token = int(movie_enc.transform([mid])[0]) + 1
                seq_tokens.append(token)
        if seq_tokens:
            # Build input: sequence (truncated) + [MASK]
            seq = seq_tokens[-max_len+1:] if len(seq_tokens) > max_len-1 else seq_tokens
            seq = seq + [mask_token]  # append mask at end
            # Pad to max_len
            if len(seq) < max_len:
                seq = [0] * (max_len - len(seq)) + seq
            inp = torch.LongTensor([seq]).to(device)
            with torch.no_grad():
                logits = model(inp)[0, -1, :].cpu().numpy()
            # Candidates: all movies except those seen (including liked)
            all_tokens = np.arange(1, n_movies+1, dtype=np.int64)
            # Exclude already seen (liked/disliked) movies
            seen_tokens = set()
            for mid in seen:
                if mid in movie_enc.classes_:
                    seen_tokens.add(int(movie_enc.transform([mid])[0]) + 1)
            if seen_tokens:
                cand = all_tokens[~np.isin(all_tokens, list(seen_tokens))]
            else:
                cand = all_tokens
            if len(cand) > 0:
                scores = logits[cand]
                top = np.argsort(scores)[::-1][:n]
                results = []
                for token in cand[top]:
                    orig_id = movie_enc.classes_[token-1]
                    row = movies[movies['movieId'] == orig_id]
                    if row.empty: continue
                    row = row.iloc[0]
                    results.append({
                        'id': int(orig_id), 'title': row['title'], 'genres': row['genres'],
                        'bert4rec_score': round(float(logits[token]), 4),
                        'explanation': "Predicted next based on your watch sequence"
                    })
                if results:
                    return results

    # Fallback to content cold start (genre-based)
    recs = get_content_cold_start_recs(preferred_genres, n, user_id)
    for r in recs:
        r['explanation'] = r['explanation'].replace("Because you selected", "Based on your interest in")
        if not r['explanation'].startswith("Based on"):
            r['explanation'] = f"Predicted from your genre preferences: {', '.join(preferred_genres)}"
    return recs

# ── Main recommendation functions ────────────────────────────────
def get_svd_recs(user_id, n=10):
    uid = int(user_id)
    seen = get_seen_movies(user_id)
    rated = ratings[ratings['userId'] == uid]['movieId'].tolist()
    if len(rated) == 0:
        pref = st.session_state.get('preferred_genres', ['Drama'])
        return get_cold_start_recs(pref, n, user_id)
    unrated = [m for m in movies['movieId'].tolist() if m not in rated and m not in seen]
    preds = [best_svd.predict(uid, mid) for mid in unrated]
    preds.sort(key=lambda x: x.est, reverse=True)
    results = []
    for pred in preds[:n]:
        row = movies[movies['movieId'] == pred.iid].iloc[0]
        results.append({
            'id': int(pred.iid),
            'title': row['title'],
            'genres': row['genres'],
            'explanation': explain_svd(user_id, pred.iid)
        })
    return results

def get_content_recs(user_id, n=10):
    try:
        uid = int(user_id)
        seen = get_seen_movies(user_id)
        seen_movies = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= 3)]['movieId'].tolist()
        seen_movies += seen
        if not seen_movies:
            pref = st.session_state.get('preferred_genres', ['Drama'])
            return get_content_cold_start_recs(pref, n, user_id)
        movie_ids = content_data['movies']['movieId'].tolist()
        id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
        idx_to_id = {i: mid for i, mid in enumerate(movie_ids)}
        train_indices = [id_to_idx[m] for m in seen_movies if m in id_to_idx]
        if not train_indices:
            pref = st.session_state.get('preferred_genres', ['Drama'])
            return get_content_cold_start_recs(pref, n, user_id)
        user_scores = cosine_sim[train_indices].mean(axis=0)
        user_scores[train_indices] = -np.inf
        top_indices = np.argsort(-user_scores)[:n]
        results = []
        for idx in top_indices:
            mid = idx_to_id[idx]
            row = movies[movies['movieId'] == mid]
            if row.empty: continue
            row = row.iloc[0]
            results.append({
                'id': int(mid),
                'title': row['title'],
                'genres': row['genres'],
                'explanation': explain_content(user_id, mid)
            })
        return results
    except Exception:
        pref = st.session_state.get('preferred_genres', ['Drama'])
        return get_content_cold_start_recs(pref, n, user_id)

# ── Session state ────────────────────────────────────────────────
if 'user_id' not in st.session_state: st.session_state.user_id = None
if 'variant' not in st.session_state: st.session_state.variant = None
if 'user_type' not in st.session_state: st.session_state.user_type = None
if 'preferred_genres' not in st.session_state: st.session_state.preferred_genres = []
if 'page' not in st.session_state: st.session_state.page = 'landing'
if 'liked_movies' not in st.session_state: st.session_state.liked_movies = []

# ── Pages ────────────────────────────────────────────────────────
def landing_page():
    st.title('🎬 Personalization Engine')
    active = []
    if NCF_LOADED: active.append('Neural CF')
    if BERT_LOADED: active.append('Sequence model')
    st.write('How do you want to start?')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('👤 New User\n\nPick your favourite genres', use_container_width=True):
            st.session_state.user_id = str(random.randint(10000, 99999))
            st.session_state.variant = random.choice(VARIANTS)
            st.session_state.user_type = 'new'
            st.session_state.page = 'survey'
            st.rerun()
    with col2:
        if st.button('🎬 Existing User\n\nUse MovieLens user history', use_container_width=True):
            st.session_state.user_id = str(random.randint(1, 6040))
            st.session_state.variant = random.choice(VARIANTS)
            st.session_state.user_type = 'existing'
            st.session_state.page = 'home'
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
            st.session_state.page = 'home'
            st.rerun()

def _next_variant(current):
    idx = VARIANTS.index(current) if current in VARIANTS else 0
    return VARIANTS[(idx + 1) % len(VARIANTS)]

def home_page():
    user_id = st.session_state.user_id
    variant = st.session_state.variant
    user_type = st.session_state.user_type

    st.title('🎬 Movie Recommendations')
    col1, col3 = st.columns([4, 1])
    col1.metric('User ID', user_id)
    if col3.button('Reset'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    recs = []
    if user_type == 'new' and is_cold_start(user_id):
        pref = st.session_state.preferred_genres
        if not pref:
            st.session_state.page = 'survey'
            st.rerun()
        st.info('🌱 Based on your genre preferences – like/dislike to personalise after 5 interactions!')
        if variant == 'bert4rec' and BERT_LOADED:
            recs = get_bert4rec_cold_start_recs(pref, user_id=user_id)
        elif variant == 'ncf' and NCF_LOADED:
            recs = get_ncf_cold_start_recs(pref, user_id=user_id)
        elif variant == 'content':
            recs = get_content_cold_start_recs(pref, user_id=user_id)
        else:
            recs = get_cold_start_recs(pref, user_id=user_id)
    else:
        if variant == 'collaborative':
            recs = get_svd_recs(user_id)
        elif variant == 'content':
            recs = get_content_recs(user_id)
        elif variant == 'ncf':
            recs = get_ncf_recs(int(user_id) if user_id.isdigit() else user_id)
        elif variant == 'bert4rec':
            recs = get_bert4rec_recs(int(user_id) if user_id.isdigit() else user_id)

    if not recs:
        st.warning('No recommendations available. Try resetting your session.')
        return

    for rec in recs:
        log_event(user_id, variant, rec['id'], 'impression')
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{rec['title']}**")
                st.caption(rec['genres'])
                if rec.get('explanation'):
                    st.caption(f"💡 {rec['explanation']}")
                if variant == 'ncf' and rec.get('predicted_rating'):
                    st.caption(f"🧠 Predicted rating: {rec['predicted_rating']:.2f} · Attention: {rec.get('attention_score',0):.3f}")
                if variant == 'bert4rec' and rec.get('bert4rec_score') is not None:
                    st.caption(f"🤖 Score: {rec['bert4rec_score']:.4f}")
            with col2:
                col_a, col_b = st.columns(2)
                if col_a.button('👍', key=f"like_{rec['id']}"):
                    log_event(user_id, variant, rec['id'], 'like')
                    st.session_state.liked_movies.append({'title': rec['title'], 'genres': rec['genres'], 'variant': variant})
                    st.session_state.variant = _next_variant(variant)
                    st.rerun()
                if col_b.button('👎', key=f"dislike_{rec['id']}"):
                    log_event(user_id, variant, rec['id'], 'dislike')
                    st.session_state.variant = _next_variant(variant)
                    st.rerun()

def results_page():
    import plotly.graph_objects as go
    import numpy as np
    from scipy import stats

    st.title('📊 Interleaved Test Results')
    con = sqlite3.connect('logs.db')
    try:
        df = pd.read_sql('SELECT * FROM events', con)
    except:
        df = pd.DataFrame()
    con.close()

    if df.empty:
        st.warning('No data yet – go interact with recommendations first!')
        return

    # Compute per‑variant metrics
    summary = {}
    for v in VARIANTS:
        sub = df[df['variant'] == v]
        imp = len(sub[sub['event'] == 'impression'])
        like = len(sub[sub['event'] == 'like'])
        dislike = len(sub[sub['event'] == 'dislike'])
        like_rate = like / imp if imp > 0 else 0
        summary[v] = {
            'impressions': imp,
            'likes': like,
            'dislikes': dislike,
            'like_rate': like_rate,
            'total_interactions': like + dislike,
        }

    # Only show models that have impressions
    active = [v for v in VARIANTS if summary[v]['impressions'] > 0]
    if len(active) == 0:
        st.info('No interaction data yet. Start using the app!')
        return

    # Display metrics in columns
    st.subheader('📈 Model Performance')
    cols = st.columns(len(active))
    for col, v in zip(cols, active):
        s = summary[v]
        with col:
            st.metric(f"{VARIANT_LABELS.get(v, v)}", f"{s['like_rate']:.2%}", 
                      help=f"Like rate = {s['likes']} / {s['impressions']} impressions")
            st.caption(f"👍 {s['likes']} likes · 👎 {s['dislikes']} dislikes · 👁️ {s['impressions']} views")
            if s['total_interactions'] > 0:
                st.caption(f"✍️ {s['total_interactions']} total reactions")

    # Bar chart of like rates with 95% credible intervals
    like_rates = [summary[v]['like_rate'] for v in active]
    likes = [summary[v]['likes'] for v in active]
    impressions = [summary[v]['impressions'] for v in active]
    labels = [VARIANT_LABELS.get(v, v) for v in active]

    lower_bounds = []
    upper_bounds = []
    for l, n in zip(likes, impressions):
        if n > 0:
            alpha = l + 1
            beta = n - l + 1
            lower_bounds.append(stats.beta.ppf(0.025, alpha, beta))
            upper_bounds.append(stats.beta.ppf(0.975, alpha, beta))
        else:
            lower_bounds.append(0)
            upper_bounds.append(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=like_rates,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[u - l for u, l in zip(upper_bounds, like_rates)],
            arrayminus=[l - lr for l, lr in zip(lower_bounds, like_rates)],
            color='rgba(0,0,0,0.5)'
        ),
        marker_color='skyblue',
        marker_line_color='navy',
        marker_line_width=1,
        opacity=0.8,
        text=[f"{r:.2%}" for r in like_rates],
        textposition='outside'
    ))
    fig.update_layout(
        title='Like Rate with 95% Credible Intervals',
        xaxis_title='Recommendation Model',
        yaxis_title='Like Rate',
        yaxis_tickformat='.0%',
        yaxis=dict(range=[0, max(upper_bounds + [0.1]) * 1.1]),
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name='Likes', x=labels, y=[summary[v]['likes'] for v in active], marker_color='green'))
    fig2.add_trace(go.Bar(name='Dislikes', x=labels, y=[summary[v]['dislikes'] for v in active], marker_color='red'))
    fig2.update_layout(barmode='group', title='Likes vs Dislikes per Model', template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

    # --- Identify leader ---
    best_idx = np.argmax(like_rates)
    best_variant = active[best_idx]
    best_name = VARIANT_LABELS.get(best_variant, best_variant.capitalize())
    best_rate = like_rates[best_idx]

    st.success(f"🏆 **Current leader: {best_name}** ({best_rate:.2%} like rate)")

    # --- Leader description (model explanation) ---
    leader_description = ""
    if best_variant == 'collaborative':
        leader_description = "Uses matrix factorization (SVD) and popularity‑sorted predictions."
    elif best_variant == 'content':
        leader_description = "Recommends movies similar to those the user has liked, based on genres."
    elif best_variant == 'ncf':
        leader_description = "Neural collaborative filtering with wide & deep architecture and attention."
    elif best_variant == 'bert4rec':
        leader_description = "Transformer‑based sequence model that predicts the next movie in the user’s watch history."

    st.info(f"📘 {leader_description}")

# ── Sidebar ─────────────────────────────────────────────────────
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

    if st.session_state.liked_movies:
        st.divider()
        st.subheader(f'❤️ Liked Movies ({len(st.session_state.liked_movies)})')
        search = st.text_input('🔍 Search', placeholder='Type to search...')
        filtered = [m for m in st.session_state.liked_movies if search.lower() in m['title'].lower()] if search else st.session_state.liked_movies
        with st.container(height=300):
            if not filtered: st.caption('No matches.')
            for movie in filtered[-20:]:
                st.markdown(f"**{movie['title']}**")
                st.caption(f"{movie['genres']}")
                st.divider()

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