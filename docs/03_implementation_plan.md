# PROJECT 1: CLASSIC CONTENT-BASED MOVIE RECOMMENDER
## DOCUMENT 3: IMPLEMENTATION PLAN / DEVELOPMENT ROADMAP

### 3.1 Overview

This plan breaks the project into 7 phases over 2-3 weeks. Each phase has clear objectives, tasks, and checkpoints.

**Estimated Total Time**: 20-30 hours

### 3.2 Phase Breakdown

---

#### **PHASE 1: Environment Setup & Data Acquisition** (1-2 hours)

**Objective**: Get your development environment ready and download data.

**Tasks**:
1. Create project directory structure
2. Set up virtual environment (optional but recommended)
3. Install dependencies
4. Download MovieLens 25M dataset
5. Create initial Jupyter notebook
6. Test imports (verify libraries work)

**Deliverables**:
- Working Jupyter environment
- Data files in project directory
- Empty notebook with test cell

**Checkpoint**: Can you import pandas, numpy, sklearn and load a CSV without errors?

**Time Estimate**: 1-2 hours

---

#### **PHASE 2: Data Loading & Initial Exploration** (4-6 hours)

**Objective**: Load data, understand its structure, and perform exploratory data analysis (EDA).

**Tasks**:
1. Load `movies.csv` into pandas DataFrame
2. Examine structure: shape, columns, data types, missing values
3. Load `tags.csv` and explore
4. Basic visualizations:
   - Distribution of genres
   - Number of tags per movie
   - Movie release year distribution
5. Identify data quality issues (missing values, duplicates, inconsistencies)
6. Document findings in markdown cells

**Key Learning**:
- pandas: `.read_csv()`, `.head()`, `.info()`, `.describe()`
- pandas: `.value_counts()`, `.isnull().sum()`, `.groupby()`
- matplotlib/seaborn: basic bar charts, histograms

**Deliverables**:
- Loaded DataFrames with documented structure
- 3-5 visualizations with interpretations
- Summary of data quality issues

**Checkpoint**: Can you answer these questions?
- How many unique movies?
- What are the top 5 genres?
- What % of movies have tags?
- Are there any duplicated movie titles?

**Time Estimate**: 4-6 hours

---

#### **PHASE 3: Data Cleaning & Preprocessing** (3-4 hours)

**Objective**: Prepare data for feature engineering by handling inconsistencies.

**Tasks**:
1. Handle missing values:
   - Movies without genres → decide strategy (drop or assign "Unknown")
   - Movies without tags → keep (will handle in feature engineering)
2. Clean genre strings:
   - Split pipe-separated genres: "Action|Comedy" → ["Action", "Comedy"]
   - Standardize genre names (if needed)
3. Aggregate tags:
   - Group tags by movie
   - Combine multiple user tags into single string per movie
   - Consider tag frequency (optional: weight popular tags)
4. Create master DataFrame:
   - Merge movies + aggregated tags
   - One row per movie with all features

**Key Learning**:
- pandas: `.dropna()`, `.fillna()`, `.str.split()`
- pandas: `.groupby()` with `.agg()` for combining tags
- pandas: `.merge()` for joining DataFrames

**Deliverables**:
- Cleaned `movies` DataFrame
- Aggregated `tags` DataFrame  
- Single `movies_with_features` DataFrame ready for modeling

**Checkpoint**: 
- Does every movie have at least one feature (genre or tag)?
- Can you print a sample row showing all features?

**Time Estimate**: 3-4 hours

---

#### **PHASE 4: Feature Engineering** (3-4 hours)

**Objective**: Create the text feature that will be vectorized.

**Tasks**:
1. Create "soup" column:
   - Combine genres + tags into single text string
   - Example: "Action Adventure Sci-Fi dystopian future robots"
2. Text preprocessing:
   - Lowercase all text
   - Remove special characters (keep only letters and spaces)
   - Handle NaN values (replace with empty string)
3. Explore feature richness:
   - Distribution of "soup" length (word count per movie)
   - Identify movies with sparse features
4. Create train/test split (optional, for evaluation):
   - Randomly select 80% movies for training, 20% for testing

**Key Learning**:
- pandas string operations: `.str.lower()`, `.str.replace()`
- Creating derived columns
- Understanding why text preprocessing matters

**Deliverable**:
- DataFrame with "soup" column containing combined features
- Documentation of preprocessing decisions
- (Optional) train/test split indices

**Checkpoint**: 
- Print the "soup" for 5 sample movies - does it look reasonable?
- Are there any movies with empty or very short soups?

**Time Estimate**: 3-4 hours

---

#### **PHASE 5: Vectorization & Similarity Calculation** (2-3 hours)

**Objective**: Convert text features to numerical vectors and compute similarity matrix.

**Tasks**:
1. Initialize TF-IDF Vectorizer:
   - Set parameters: `max_features=5000`, `stop_words='english'`
   - Understand what each parameter does
2. Fit and transform "soup" column:
   - Result: sparse matrix of TF-IDF vectors
3. Examine TF-IDF output:
   - Shape of matrix
   - Most important terms per movie (top TF-IDF scores)
4. Compute cosine similarity matrix:
   - Use `cosine_similarity()` on TF-IDF matrix
   - Result: square matrix (n_movies × n_movies)
5. Inspect similarity matrix:
   - Check diagonal (should be 1.0 - movie is identical to itself)
   - Sample some similarities to verify they make sense

**Key Learning**:
- How TF-IDF works conceptually
- sklearn: `TfidfVectorizer` class
- sklearn: `cosine_similarity()` function
- Understanding sparse vs dense matrices
- NumPy array indexing

**Code Example**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize vectorizer
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')

# Create TF-IDF matrix: each row is a movie, columns are words
tfidf_matrix = tfidf.fit_transform(movies_df['soup'])

# Calculate cosine similarity between all pairs of movies
# Result shape: (n_movies, n_movies)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# cosine_sim[i][j] = similarity between movie i and movie j
```

**Deliverables**:
- TF-IDF matrix (sparse)
- Cosine similarity matrix
- Documentation of parameter choices

**Checkpoint**:
- Is the similarity matrix symmetric? (sim[i][j] should equal sim[j][i])
- Pick a movie you know - do its top 5 similar movies make sense?

**Time Estimate**: 2-3 hours

---

#### **PHASE 6: Build Recommendation Function** (2-3 hours)

**Objective**: Create a function that takes a movie title and returns recommendations.

**Tasks**:
1. Create helper function to get movie index from title
2. Create main recommendation function:
   - Input: movie title, number of recommendations
   - Process: lookup similarity scores, sort, return top N
   - Output: list of movie titles
3. Handle edge cases:
   - Movie not found
   - Requesting more recommendations than available
4. Test function on various inputs:
   - Popular movies (Toy Story, The Dark Knight)
   - Obscure movies
   - Different genres (action, romance, documentary)
5. Add optional: similarity scores in output

**Key Learning**:
- Python function design
- pandas indexing and filtering
- NumPy array sorting and indexing
- Error handling with try/except

**Code Example**:
```python
def get_recommendations(title, n=10):
    """
    Get movie recommendations based on content similarity.
    
    Args:
        title (str): Exact movie title (e.g., "Toy Story (1995)")
        n (int): Number of recommendations to return
    
    Returns:
        list: Movie titles of recommended movies
    """
    # Step 1: Find the index of the movie in our dataframe
    idx = movies_df[movies_df['title'] == title].index[0]
    
    # Step 2: Get similarity scores for this movie with all others
    # cosine_sim[idx] is a 1D array of similarities
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Step 3: Sort movies by similarity score (descending)
    # Skip the first one (the movie itself with similarity 1.0)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    # Step 4: Get movie indices from sorted scores
    movie_indices = [i[0] for i in sim_scores]
    
    # Step 5: Return the movie titles
    return movies_df['title'].iloc[movie_indices].tolist()
```

**Deliverables**:
- Working `get_recommendations()` function
- Test cases showing varied inputs and outputs
- Documentation explaining how it works

**Checkpoint**:
- Can you get recommendations for any movie in the dataset?
- Do the recommendations intuitively make sense?

**Time Estimate**: 2-3 hours

---

#### **PHASE 7: Evaluation & Documentation** (4-6 hours)

**Objective**: Measure recommendation quality and document your work.

**Tasks**:

**Evaluation**:
1. Implement Precision@K metric:
   - For a sample of movies, check if recommendations share genres
   - Calculate average precision across sample
2. Analyze similarity score distribution:
   - Histogram of all similarity scores
   - Identify threshold for "good" similarity
3. Qualitative review:
   - Manually review 10 diverse movie recommendations
   - Document what works and what doesn't
4. Create visualizations:
   - Precision by genre (do some genres recommend better?)
   - Similarity score distributions

**Documentation**:
5. Add markdown throughout notebook:
   - Section headers
   - Explanations of why you made decisions
   - Interpretations of results
6. Create README.md:
   - Project overview
   - Setup instructions
   - Results summary
   - Limitations and future work
7. Clean notebook:
   - Remove debug cells
   - Organize imports at top
   - Add table of contents
8. Create requirements.txt

**Key Learning**:
- How to evaluate recommendations without user feedback
- Writing professional documentation
- Communicating technical work

**Deliverables**:
- Precision@K scores documented
- 3-5 evaluation visualizations
- Complete, well-documented notebook
- Professional README
- requirements.txt

**Checkpoint**:
- Is your precision@10 above 0.7?
- Can someone else understand your notebook without your explanation?
- Does your README clearly explain what you built?

**Time Estimate**: 4-6 hours

---

### 3.3 Dependencies Between Phases

```
Phase 1 (Setup)
    ↓
Phase 2 (EDA) 
    ↓
Phase 3 (Cleaning) ← Must complete before Phase 4
    ↓
Phase 4 (Feature Engineering) ← Must complete before Phase 5
    ↓
Phase 5 (Vectorization) ← Must complete before Phase 6
    ↓
Phase 6 (Recommendation Function) ← Must complete before Phase 7
    ↓
Phase 7 (Evaluation)
```

**Critical Path**: You must complete phases in order. Each phase builds on previous work.

### 3.4 Milestones and Checkpoints

**Week 1 Milestone**: Phases 1-3 complete
- You have clean, explored data ready for modeling
- Checkpoint: Can you print a DataFrame row showing all features?

**Week 2 Milestone**: Phases 4-6 complete
- You have a working recommendation system
- Checkpoint: Can you get recommendations for any movie?

**Week 3 Milestone**: Phase 7 complete
- You have evaluated and documented your work
- Checkpoint: Is your GitHub repo professional and complete?

### 3.5 Risk Management

**Potential Issues & Solutions**:

1. **Memory errors when computing similarity matrix**
   - Solution: Reduce dataset to 10k movies initially
   - Solution: Use `similarity_matrix = cosine_similarity(tfidf_matrix)` which returns sparse matrix

2. **Poor recommendation quality**
   - Solution: Inspect feature quality - are soups too sparse?
   - Solution: Try different TF-IDF parameters (`max_features`, `ngram_range`)

3. **Slow computation**
   - Solution: This is expected for first run - computation is offline
   - Solution: Save similarity matrix with `np.save()` to avoid recomputing

4. **Difficulty with pandas operations**
   - Solution: Break complex operations into smaller steps
   - Solution: Print intermediate results to understand transformations
   - Solution: Use `.head()` frequently to inspect DataFrames

5. **Getting stuck on a phase**
   - Solution: Move to next phase and return later
   - Solution: Simplify - get something working first, optimize later

### 3.6 Definition of Done

**This project is complete when**:
- ✅ All 7 phases are finished
- ✅ Notebook runs end-to-end without errors
- ✅ Precision@10 ≥ 0.7
- ✅ README is professional and complete
- ✅ Code is on GitHub with good commit history
- ✅ You can explain to someone how it works
