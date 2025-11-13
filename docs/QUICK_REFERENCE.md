# Quick Reference Guide
## Project 1: Classic Content-Based Movie Recommender

This is a condensed reference for quick lookups while working. For detailed explanations, see the full documents.

---

## 🎯 Project at a Glance

- **Goal**: Build content-based movie recommender
- **Input**: Movie title → **Output**: 10 similar movies
- **Dataset**: MovieLens 25M (62k movies)
- **Key Tech**: pandas, scikit-learn, TF-IDF, cosine similarity
- **Success Metric**: Precision@10 ≥ 0.7
- **Time**: 20-30 hours (2-3 weeks)

---

## 📋 7 Phases Checklist

- [ ] **Phase 1**: Environment Setup (1-2h)
- [ ] **Phase 2**: Data Loading & EDA (4-6h)
- [ ] **Phase 3**: Data Cleaning (3-4h)
- [ ] **Phase 4**: Feature Engineering (3-4h)
- [ ] **Phase 5**: Vectorization & Similarity (2-3h)
- [ ] **Phase 6**: Recommendation Function (2-3h)
- [ ] **Phase 7**: Evaluation & Documentation (4-6h)

---

## 💻 Essential Code Snippets

### Load Data
```python
import pandas as pd
movies = pd.read_csv('data/ml-25m/movies.csv')
tags = pd.read_csv('data/ml-25m/tags.csv')
```

### Aggregate Tags
```python
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = movies.merge(movie_tags, on='movieId', how='left')
```

### Create Feature Soup
```python
movies_with_tags['soup'] = (
    movies_with_tags['genres'].fillna('') + ' ' + 
    movies_with_tags['tag'].fillna('')
).str.lower().str.replace('|', ' ')
```

### TF-IDF Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['soup'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

### Recommendation Function
```python
def get_recommendations(title, n=10):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()
```

### Precision@K
```python
def calculate_precision_at_k(input_movie, recommendations, movies_df, k=10):
    input_genres = set(movies_df[movies_df['title'] == input_movie]['genres'].iloc[0].split('|'))
    relevant_count = 0
    for rec_title in recommendations[:k]:
        rec_genres = set(movies_df[movies_df['title'] == rec_title]['genres'].iloc[0].split('|'))
        if len(input_genres.intersection(rec_genres)) >= 1:
            relevant_count += 1
    return relevant_count / k
```

---

## 📊 Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `max_features` | 5000 | Balance between richness and computation |
| `stop_words` | 'english' | Remove common words (the, is, a) |
| `n_recommendations` | 10 | Standard for recommender evaluation |
| `test_size` | 0.2 | 80/20 train/test split (optional) |
| `random_state` | 42 | Reproducibility |

---

## 🔍 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Memory error | Similarity matrix too large | Reduce dataset to 10k movies |
| Low precision | Sparse features | Check soup quality, more tags needed |
| Slow computation | Large dataset | Normal for first run; save matrix |
| Movie not found | Typo in title | Use exact title with year |
| Import errors | Missing libraries | Check requirements.txt installed |

---

## 📈 Evaluation Targets

- **Precision@10**: ≥ 0.7 (70% relevant)
- **Mean similarity**: ≥ 0.3 (decent matches)
- **Genre coverage**: All major genres ≥ 0.6 precision
- **Qualitative**: Recommendations make intuitive sense

---

## 🗂️ File Structure

```
movie-recommender/
├── data/ml-25m/          # MovieLens dataset
├── notebooks/
│   └── movie_recommender.ipynb
├── models/               # Saved models (optional)
├── README.md
└── requirements.txt
```

---

## 🎨 Notebook Sections

1. Setup & Imports
2. Data Loading
3. EDA
4. Data Cleaning
5. Feature Engineering
6. Model Building
7. Recommendation Function
8. Evaluation
9. Conclusions
10. Future Work

---

## 📚 When to Reference Which Document

| Working on... | Reference... |
|--------------|--------------|
| Understanding project | 01_project_brief.md |
| Algorithm details | 02_technical_specification.md |
| Phase planning | 03_implementation_plan.md |
| Data questions | 04_data_guide.md |
| Code organization | 05_code_structure.md |
| Task tracking | 06_development_checklist.md |
| Evaluation | 07_evaluation_framework.md |

---

## ⚡ Pandas Cheat Sheet (for this project)

```python
# Load CSV
df = pd.read_csv('file.csv')

# Explore
df.head()                    # First 5 rows
df.info()                    # Column types, non-null counts
df.describe()                # Statistics
df['column'].value_counts()  # Count unique values
df.isnull().sum()           # Count missing values

# Clean
df['col'].fillna('')         # Fill missing values
df['col'].str.lower()        # Lowercase
df['col'].str.replace('|', ' ')  # Replace characters
df.dropna()                  # Remove rows with missing values

# Transform
df['new'] = df['a'] + ' ' + df['b']  # Combine columns
df.groupby('col').size()     # Count per group
df.groupby('col')['text'].apply(lambda x: ' '.join(x))  # Aggregate text

# Merge
df_merged = df1.merge(df2, on='id', how='left')

# Filter
df[df['col'] > 5]           # Rows where col > 5
df[df['col'].str.contains('word')]  # Rows containing 'word'
```

---

## 🚨 Before Moving to Next Phase

Ask yourself:
1. Did I complete all tasks in the checklist?
2. Can I answer the checkpoint questions?
3. Did I commit my progress to Git?
4. Do my intermediate results make sense?
5. Is my code documented with comments?

---

## 🎓 Key Concepts to Understand

- **TF-IDF**: Converts text to numbers, weighting rare words higher
- **Cosine Similarity**: Measures angle between vectors (0-1 scale)
- **Content-Based Filtering**: Recommends based on item features, not user behavior
- **Precision@K**: Of K recommendations, % that are relevant
- **Feature Engineering**: Creating useful representations from raw data

---

## 📞 Stuck? Debugging Steps

1. Print the DataFrame: `print(df.head())`
2. Check shape: `print(df.shape)`
3. Check data types: `print(df.dtypes)`
4. Check for nulls: `print(df.isnull().sum())`
5. Print intermediate results at each step
6. Try on a small sample first (10 movies)
7. Read error messages carefully
8. Google the specific error
9. Check the relevant full document

---

## ✅ Project Complete When...

- [ ] Notebook runs end-to-end without errors
- [ ] Precision@10 ≥ 0.7
- [ ] All visualizations present and labeled
- [ ] README.md is professional
- [ ] Code on GitHub
- [ ] Can explain how it works

---

## 🎯 Next Steps After Project 1

1. **Project 2**: Enhanced recommender with multiple metrics
2. **Project 3**: Film noir specialist
3. **FastAPI Integration**: Turn into web service
4. **Add to Resume**: "Built content-based recommender with 0.7+ precision"
5. **LinkedIn Post**: Share your learning journey

---

**Remember**: It's okay to refer back to the full documents. This quick reference is just for fast lookups, not for learning. When in doubt, read the detailed guide!
