# Personalisation Engine
A machine learning based personalisation engine that recommends items (movies) to users by learning from past interactions. The system evaluates multiple recommendation approaches and compares their effectiveness using engagement and ranking metrics.

This project addresses the problem of content overload by building a recommendation system that predicts which items a user is most likely to engage with.
We implement and compare three primary models:
- A Singular Value Decomposition (SVD) Model
- A Content Filtering (by Genre) Model
- A Neural Collaborative Filtering (NCF) Model

The system is evaluated using offline A/B testing to determine the most effective approach.

---
## Features
### Three Engineered Models:
**A Singular Value Decomposition (SVD) Model**
Finds patterns in user behaviours to match users with similar interests. It looks at how users interact with items and learns underlying preferences, then uses those patterns to recommend new items a user might like.

**A Content Filtering (by Genre) Model**
Recommends items similar to what the user already likes/ If a user likes certain genres, the system suggests other items with similar characteristics, without needing other users' data.

**A Neural Collaborative Filtering (NCF) Model**
Uses a neural network to learn complex relationships between users and items. Instead of simple patterns, this model learns deeper and more nuanced preferences, allowing it to capture more subtle user behaviours and interactions; the most complex of the three primary models.

**Popularity Baseline**
Recommends the most popular items to everyone. A simple approach where all users are shown the items that are most frequently highly rated, regardless of personal preference.

**Cold-Start Onboarding**
Handles new users by asking for initial preferences. Since new users have no interaction history, the system uses content-based filtering (by genre) to recommend items until enough data is collected for personalisation.

---
## Streamlit App
**View Demo:** [Watch Me!](https://www.youtube.com/watch?v=GKddulwY9lY)

## Key Results
Every recommendation is accompanies by a natural language explanation drawn from the model that produced it.
Examples include:
- "Users with similar tastes to yours also loves {item}."
- "Because you liked {genre}."

All models achieved comparable performance. Differences across CTR, Precision, Recall and NDCG@K were marginal. No statistically significant improvement (p > 0.05). Increasing model complexity did not lead to meaningful gains on this dataset.

---
## Future Work
- The app already includes the prototype for a BERT4Rec model which would extend our system towards a more sequence-aware recommendations, where the model learns from user behaviour.
- Expanding within model comparison to catch limitations and make improvements to the system.

## Important Links
[Report](link)
