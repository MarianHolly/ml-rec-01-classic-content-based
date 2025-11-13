# PROJECT 1: CLASSIC CONTENT-BASED MOVIE RECOMMENDER
## DOCUMENT 2: TECHNICAL SPECIFICATION

### 2.1 System Architecture Overview

**High-Level Flow**:
```
Raw Data (CSV files) 
  → Data Loading & Cleaning (pandas)
  → Feature Engineering (combine text features)
  → Vectorization (TF-IDF converts text → numbers)
  → Similarity Calculation (cosine similarity between all movies)
  → Recommendation Function (finds most similar movies)
  → Evaluation (measures quality)
```

**Key Concept**: We're building an **offline, content-based recommender**. "Offline" means we compute all similarities once, then look them up quickly when needed.

### 2.2 Data Requirements

**Primary Dataset**: MovieLens 25M Dataset
- **Why this dataset?**: Industry-standard, clean, well-documented, free
- **What it contains**: 
  - 62,000+ movies
  - Genres, tags, ratings
  - Movie metadata (titles, years)
- **Download**: https://grouplens.org/datasets/movielens/25m/

**Files We'll Use**:
1. `movies.csv` - Movie IDs, titles, genres
2. `tags.csv` - User-generated tags for movies (keywords)
3. `ratings.csv` - (Optional, for evaluation only)

**Data We Won't Use** (yet):
- `links.csv` - External IDs (IMDB, TMDB)
- `genome-scores.csv` and `genome-tags.csv` - Advanced tag relevance scores

### 2.3 Technologies and Libraries

**Core Stack** (add to `requirements.txt`):
```
numpy==1.24.3          # Numerical operations, arrays
pandas==2.0.3          # Data manipulation
scikit-learn==1.3.0    # TF-IDF, cosine similarity, metrics
matplotlib==3.7.2      # Basic plotting
seaborn==0.12.2        # Statistical visualizations
jupyter==1.0.0         # Notebook environment
```

**Why These Versions?**: Recent stable releases with good compatibility.

**Installation**:
```bash
pip install -r requirements.txt
```

### 2.4 Key Algorithms and Methodologies

#### 2.4.1 TF-IDF Vectorization

**What it does**: Converts text (like "Action Comedy Sci-Fi") into numerical vectors that capture importance.

**Why TF-IDF, not simple word counts?**
- TF (Term Frequency): Rewards words that appear often in a document
- IDF (Inverse Document Frequency): Penalizes words that appear in many documents (like "the", "movie")
- Result: Rare, distinctive words get higher weight

**Example**:
- Movie A: "Action Adventure Fantasy" 
- Movie B: "Action Drama"
- TF-IDF will recognize "Fantasy" as more distinctive than "Action" (which appears in both)

**Implementation**: `sklearn.feature_extraction.text.TfidfVectorizer`

#### 2.4.2 Cosine Similarity

**What it does**: Measures how similar two vectors are, ignoring magnitude (only caring about direction).

**Why Cosine?**: 
- Range: 0 (completely different) to 1 (identical)
- Works well with TF-IDF vectors
- Fast to compute for many comparisons

**Geometric Intuition**: Imagine two arrows in space. Cosine similarity measures the angle between them. Small angle = similar direction = high similarity.

**Implementation**: `sklearn.metrics.pairwise.cosine_similarity`

#### 2.4.3 Content-Based Filtering Logic

**Core Idea**: 
1. Represent each movie as a vector of features
2. Calculate similarity between all movie pairs
3. When user likes Movie X, recommend movies with highest similarity to X

**Mathematical Representation**:
```
Movie Vector = [genre features, tag features, ...]
Similarity(Movie A, Movie B) = cosine_similarity(Vector_A, Vector_B)
Recommendations = Top N movies with highest similarity to input movie
```

### 2.5 Feature Engineering Strategy

**Features We'll Use**:
1. **Genres** (from `movies.csv`): "Action|Adventure|Sci-Fi"
2. **Tags** (from `tags.csv`): User-generated keywords aggregated per movie
3. **Combined Feature String**: Merge genres + tags into single text field

**Feature Preprocessing**:
- Lowercase all text
- Remove special characters if needed
- Handle missing values (movies without tags)
- No stemming/lemmatization needed (for simplicity in v1)

**Why This Approach?**:
- Genres are categorical but treated as text (works well with TF-IDF)
- Tags provide rich, descriptive information
- Combining both captures different aspects of movie content

### 2.6 Input/Output Specifications

**Input**:
```python
movie_title = "Toy Story (1995)"  # Exact title as appears in dataset
n_recommendations = 10             # Number of recommendations to return
```

**Output**:
```python
[
  "Toy Story 2 (1999)",
  "Monsters, Inc. (2001)", 
  "Finding Nemo (2003)",
  ...  # 10 total recommendations
]
```

**Edge Cases to Handle**:
- Movie title not found → return helpful error message
- Movie has no features → skip or return popular movies
- n_recommendations > available similar movies → return all available

### 2.7 Performance Requirements

**Computation Time**:
- Initial similarity matrix computation: 1-5 minutes (acceptable for offline computation)
- Single recommendation query: < 1 second

**Memory**:
- Similarity matrix for 62k movies: ~30 GB if stored as dense matrix
- **Solution**: Use sparse matrices or only store top K similarities per movie

**Optimization Strategy** (for this project):
- Accept the computation time (it's one-time)
- Store full similarity matrix in memory (if RAM allows)
- If memory issues: reduce dataset to 10k most popular movies

### 2.8 Evaluation Approach

**Metrics We'll Use**:

1. **Precision@K**: Of K recommendations, how many share at least one genre with input?
   - Formula: (# relevant recommendations) / K
   - Target: ≥ 0.7 (70% should share genre)

2. **Qualitative Assessment**: Manually review 10 sample recommendations
   - Do they "make sense" intuitively?
   - Are they too obvious (sequels) or too diverse?

3. **Similarity Score Distribution**: Visualize distribution of similarity scores
   - Are we finding truly similar movies, or are all similarities low?

**Why Not More Metrics?**:
- No user ratings to predict (RMSE, MAE not applicable)
- No user-item interaction matrix (recall harder to define)
- We're focusing on content quality, not prediction accuracy

### 2.9 Assumptions and Limitations

**Assumptions**:
- Movie titles are unique in the dataset
- Users will input exact titles (no fuzzy matching)
- Static dataset (no new movies added during runtime)
- All movies have at least a genre

**Known Limitations**:
- Can only recommend movies in the dataset
- No personalization (everyone gets same recommendations for a given movie)
- Quality depends entirely on feature richness (movies with sparse features get poor recommendations)
- Overspecialization: might only recommend very similar movies, missing diverse options
- Cold start: cannot recommend brand new movies with no features

**These are acceptable** for a learning project. You'll address them in future iterations.
