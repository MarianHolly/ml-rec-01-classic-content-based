# PROJECT 1: CLASSIC CONTENT-BASED MOVIE RECOMMENDER
## DOCUMENT 6: DEVELOPMENT CHECKLIST

This checklist provides concrete, actionable tasks for each phase. Check off items as you complete them.

### PHASE 1: Environment Setup (1-2 hours)

**Setup Tasks**:
- [ ] Create project directory: `movie-recommender/`
- [ ] Create subdirectories: `data/`, `notebooks/`, `models/`
- [ ] (Optional) Create virtual environment: `python -m venv venv`
- [ ] (Optional) Activate virtual environment
- [ ] Create `requirements.txt` with dependencies
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Download MovieLens 25M dataset from GroupLens
- [ ] Extract dataset to `data/ml-25m/`
- [ ] Verify files exist: `movies.csv`, `tags.csv`

**Notebook Setup**:
- [ ] Create `notebooks/movie_recommender.ipynb`
- [ ] Add title and introduction markdown cell
- [ ] Add table of contents
- [ ] Create imports cell
- [ ] Test imports: run cell, verify no errors
- [ ] Add configuration cell with file paths
- [ ] Test loading a CSV: `pd.read_csv(MOVIES_FILE).head()`

**Git Setup**:
- [ ] Initialize git: `git init`
- [ ] Create `.gitignore` file
- [ ] Add data/ and models/ to `.gitignore`
- [ ] Initial commit: `git add .` and `git commit -m "Initial setup"`

**Checkpoint Questions**:
- [ ] Can you import all required libraries without errors?
- [ ] Can you load `movies.csv` and see its contents?
- [ ] Do you have at least 10 GB free disk space?

---

### PHASE 2: Data Loading & EDA (4-6 hours)

**Data Loading**:
- [ ] Load `movies.csv` into DataFrame
- [ ] Verify shape: should be (62423, 3)
- [ ] Load `tags.csv` into DataFrame
- [ ] Verify shape: should be (1093360, 4)
- [ ] Display first 10 rows of each DataFrame
- [ ] Check data types: `movies.dtypes`, `tags.dtypes`
- [ ] Check for missing values: `.isnull().sum()`

**Movies Dataset Exploration**:
- [ ] Count unique movies: `movies['movieId'].nunique()`
- [ ] Check for duplicate titles
- [ ] Examine title format (verify year in parentheses)
- [ ] List all unique genres: `movies['genres'].str.split('|').explode().unique()`
- [ ] Count movies per genre
- [ ] Identify movies with "(no genres listed)"
- [ ] Find movies from different decades (1950s, 1980s, 2010s)

**Tags Dataset Exploration**:
- [ ] Count unique users: `tags['userId'].nunique()`
- [ ] Count unique movies with tags: `tags['movieId'].nunique()`
- [ ] Calculate % of movies that have tags
- [ ] Count tags per movie: `tags.groupby('movieId').size().describe()`
- [ ] Find most tagged movies
- [ ] Find most common tags: `tags['tag'].value_counts().head(20)`
- [ ] Examine sample tags for a popular movie (e.g., Toy Story)

**Visualizations** (create at least 3):
- [ ] Bar chart: Top 15 genres by count
- [ ] Histogram: Distribution of release years
- [ ] Histogram: Number of tags per movie
- [ ] Bar chart: Top 20 most common tags

**Documentation**:
- [ ] Add markdown cell summarizing dataset size
- [ ] Document data quality observations
- [ ] Note any interesting patterns discovered
- [ ] List potential issues to handle in cleaning phase

**Checkpoint Questions**:
- [ ] How many movies in total? (Should be ~62k)
- [ ] What % of movies have at least one tag? (Should be ~40%)
- [ ] What are the top 3 genres? (Likely Drama, Comedy, Action)
- [ ] Can you explain what each column represents?

---

### PHASE 3: Data Cleaning & Preprocessing (3-4 hours)

**Movies Cleaning**:
- [ ] Replace "(no genres listed)" with empty string or "Unknown"
- [ ] Verify no null values in critical columns (movieId, title)
- [ ] Split genres into list for analysis: `.str.split('|')`
- [ ] Check for any unusual characters in titles
- [ ] Verify all titles have year in parentheses
- [ ] Create genre_list column for later use

**Tags Cleaning**:
- [ ] Convert all tags to lowercase: `tags['tag'] = tags['tag'].str.lower()`
- [ ] Remove leading/trailing whitespace: `.str.strip()`
- [ ] Check for null tags: `tags['tag'].isnull().sum()`
- [ ] (Optional) Remove very rare tags (appear only once)
- [ ] Examine tag quality (look for typos, gibberish)

**Tags Aggregation**:
- [ ] Group tags by movieId
- [ ] Concatenate all tags per movie: `groupby('movieId')['tag'].apply(lambda x: ' '.join(x))`
- [ ] Result: DataFrame with movieId and combined tags
- [ ] Verify: number of rows = number of unique movies with tags
- [ ] Check sample: verify tags are properly combined

**Data Merging**:
- [ ] Merge movies with aggregated tags: `movies.merge(tags_agg, on='movieId', how='left')`
- [ ] Use `how='left'` to keep all movies
- [ ] Verify result shape: should equal number of movies
- [ ] Check for movies without tags (NaN in tag column)
- [ ] Fill NaN tags with empty string: `.fillna('')`

**Master DataFrame Creation**:
- [ ] Create `movies_features` DataFrame with columns: movieId, title, genres, tags
- [ ] Verify no null values in movieId or title
- [ ] Print summary statistics: `.info()`, `.describe()`
- [ ] Sample 10 random rows and inspect

**Checkpoint Questions**:
- [ ] Does every movie have at least genres OR tags?
- [ ] How many movies have both genres and tags?
- [ ] Can you print a full record showing all features?
- [ ] Are there any unexpected data types?

---

### PHASE 4: Feature Engineering (3-4 hours)

**Create "Soup" Column**:
- [ ] Combine genres and tags: `genres + ' ' + tags`
- [ ] Handle missing values: `.fillna('')`
- [ ] Remove pipe separators: `.str.replace('|', ' ')`
- [ ] Convert to lowercase: `.str.lower()`
- [ ] Remove special characters (keep only letters and spaces)
- [ ] Remove extra whitespace: `.str.strip()`
- [ ] Add "soup" column to movies_features

**Feature Quality Analysis**:
- [ ] Calculate soup length (character count): `movies_features['soup'].str.len()`
- [ ] Calculate soup word count: `movies_features['soup'].str.split().str.len()`
- [ ] Plot distribution of soup lengths
- [ ] Identify movies with very short soups (< 5 words)
- [ ] Identify movies with very long soups (> 100 words)
- [ ] Sample and inspect: 5 short soups, 5 medium, 5 long

**Feature Validation**:
- [ ] Check for empty soups: `(movies_features['soup'] == '').sum()`
- [ ] Verify soup combines genres and tags correctly
- [ ] Print soups for well-known movies (Toy Story, Matrix, Titanic)
- [ ] Verify they look reasonable

**Vocabulary Analysis**:
- [ ] Extract all unique words from soups: `' '.join(soups).split()`
- [ ] Count unique words: vocabulary size
- [ ] Find most common words across all soups
- [ ] Visualize: word frequency distribution

**Optional: Train/Test Split**:
- [ ] Decide: split or use full dataset for first version
- [ ] If splitting: `train_test_split(movies_features, test_size=0.2, random_state=42)`
- [ ] Verify split sizes
- [ ] Save indices for later use

**Checkpoint Questions**:
- [ ] What's the average soup length (words)?
- [ ] How many movies have empty soups?
- [ ] Does the soup for "Toy Story" include both genres and tags?
- [ ] What are the 10 most common words in soups?

---

### PHASE 5: Vectorization & Similarity (2-3 hours)

**TF-IDF Setup**:
- [ ] Import: `from sklearn.feature_extraction.text import TfidfVectorizer`
- [ ] Initialize vectorizer: `TfidfVectorizer(max_features=5000, stop_words='english')`
- [ ] Understand parameters:
  - [ ] What does `max_features` do?
  - [ ] Why `stop_words='english'`?
- [ ] Document parameter choices in markdown

**Vectorization**:
- [ ] Fit and transform soups: `tfidf_matrix = tfidf.fit_transform(movies_features['soup'])`
- [ ] Check matrix shape: `tfidf_matrix.shape`
  - Should be (n_movies, max_features)
- [ ] Check matrix type: Should be sparse matrix
- [ ] Calculate matrix density: `tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])`
- [ ] Get feature names: `tfidf.get_feature_names_out()`

**TF-IDF Analysis**:
- [ ] For a sample movie, get its TF-IDF vector
- [ ] Find words with highest TF-IDF scores in that vector
- [ ] Verify: do high-scoring words make sense for that movie?
- [ ] Repeat for 3-5 different movies
- [ ] Document observations

**Cosine Similarity Calculation**:
- [ ] Import: `from sklearn.metrics.pairwise import cosine_similarity`
- [ ] Compute similarity: `cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)`
- [ ] Check shape: Should be (n_movies, n_movies)
- [ ] Verify symmetry: `cosine_sim[i][j]` should equal `cosine_sim[j][i]`
- [ ] Check diagonal: Should all be 1.0
- [ ] Measure computation time

**Similarity Matrix Analysis**:
- [ ] Get similarity distribution: `cosine_sim.flatten()`
- [ ] Calculate statistics: mean, median, min, max
- [ ] Plot histogram of all similarity scores
- [ ] Find pairs with highest similarity (likely sequels)
- [ ] Find pairs with lowest similarity
- [ ] Sample: Get top 10 similar movies for a well-known title

**Optional: Save Artifacts**:
- [ ] Save similarity matrix: `np.save('models/cosine_sim.npy', cosine_sim)`
- [ ] Save vectorizer: `joblib.dump(tfidf, 'models/tfidf.pkl')`
- [ ] Test loading: Verify you can reload and use them

**Checkpoint Questions**:
- [ ] What's the shape of your TF-IDF matrix?
- [ ] What's the average similarity score across all pairs?
- [ ] Can you name the top 5 similar movies to "Toy Story"?
- [ ] Do the similarities make intuitive sense?

---

### PHASE 6: Build Recommendation Function (2-3 hours)

**Helper Function: Get Movie Index**:
- [ ] Define function to find movie index from title
- [ ] Handle case: movie not found (raise helpful error)
- [ ] Test with various titles (existing and non-existing)
- [ ] Add docstring explaining usage

**Main Recommendation Function**:
- [ ] Define `get_recommendations(title, n=10)` function
- [ ] Step 1: Get movie index from title
- [ ] Step 2: Extract similarity scores for that movie
- [ ] Step 3: Sort movies by similarity (descending)
- [ ] Step 4: Exclude the input movie itself (similarity = 1.0)
- [ ] Step 5: Get top N movie indices
- [ ] Step 6: Return movie titles
- [ ] Add comprehensive docstring

**Enhanced Version (optional)**:
- [ ] Return similarity scores along with titles
- [ ] Add option to filter by minimum similarity threshold
- [ ] Add option to exclude sequels/prequels
- [ ] Format output as DataFrame for better readability

**Error Handling**:
- [ ] Test with non-existent movie title
- [ ] Test with n > available movies
- [ ] Test with n = 0 or negative n
- [ ] Add try/except blocks where appropriate
- [ ] Provide helpful error messages

**Function Testing**:
- [ ] Test with popular movie: "Toy Story (1995)"
- [ ] Test with action movie: "The Dark Knight (2008)"
- [ ] Test with romance: "Titanic (1997)"
- [ ] Test with documentary or niche genre
- [ ] Test with old movie (pre-1950)
- [ ] Test with obscure movie (few tags)

**Qualitative Review**:
- [ ] For each test, review recommended titles
- [ ] Do they share genres with input?
- [ ] Do they share themes/keywords?
- [ ] Are there obvious sequels? (expected)
- [ ] Are there surprising recommendations?
- [ ] Document which recommendations seem good vs. poor

**Benchmark Performance**:
- [ ] Time how long a single recommendation takes
- [ ] Should be < 1 second
- [ ] If slow, investigate bottlenecks

**Checkpoint Questions**:
- [ ] Can your function handle any movie in the dataset?
- [ ] What happens when you request 100 recommendations?
- [ ] Do recommendations for action movies return action movies?
- [ ] Can you explain why a specific movie was recommended?

---

### PHASE 7: Evaluation & Documentation (4-6 hours)

**Precision@K Metric**:
- [ ] Define what "relevant" means (shared genre? similar theme?)
- [ ] Write function: `calculate_precision_at_k(input_movie, recommendations, k=10)`
- [ ] For sample of movies, get recommendations
- [ ] Calculate precision: # relevant / k
- [ ] Calculate average precision across sample
- [ ] Target: ≥ 0.7 average precision

**Evaluation Dataset**:
- [ ] Select 50-100 diverse movies for evaluation
- [ ] Include different genres
- [ ] Include popular and obscure movies
- [ ] Include old and new movies

**Run Evaluation**:
- [ ] For each movie in eval set, get recommendations
- [ ] Calculate precision@10
- [ ] Store results in DataFrame
- [ ] Calculate overall statistics (mean, median, std)
- [ ] Identify movies with best/worst precision

**Similarity Score Analysis**:
- [ ] For evaluation set, examine similarity scores
- [ ] Plot distribution of top similarity scores
- [ ] What's the typical similarity of recommended movies?
- [ ] Are high-precision movies also high-similarity?

**Precision by Genre**:
- [ ] Group evaluation movies by primary genre
- [ ] Calculate average precision per genre
- [ ] Visualize: bar chart of precision by genre
- [ ] Identify which genres recommend well vs. poorly
- [ ] Hypothesize why some genres perform better

**Qualitative Analysis**:
- [ ] Manually review 10 sets of recommendations
- [ ] Document: what works well?
- [ ] Document: what doesn't work?
- [ ] Identify patterns in failures
- [ ] Note surprising/interesting recommendations

**Create Visualizations**:
- [ ] Histogram: Precision@10 distribution
- [ ] Bar chart: Precision by genre
- [ ] Scatter plot: Similarity score vs. Precision
- [ ] Box plot: Similarity score distribution for good vs. bad recommendations

**Notebook Documentation**:
- [ ] Add clear markdown headers for all sections
- [ ] Explain methodology before each major section
- [ ] Interpret results after each analysis
- [ ] Add "Key Findings" subsections
- [ ] Ensure code cells have comments
- [ ] Verify all visualizations have titles, labels, legends

**README Creation**:
- [ ] Project title and description
- [ ] Overview of approach (content-based filtering)
- [ ] Setup instructions (requirements, data download)
- [ ] Usage example (how to run notebook)
- [ ] Results summary (precision achieved, key findings)
- [ ] Limitations section
- [ ] Future improvements section
- [ ] References/acknowledgments

**requirements.txt**:
- [ ] List all dependencies with versions
- [ ] Test: create new environment and install from requirements.txt
- [ ] Verify notebook runs in fresh environment

**Notebook Cleanup**:
- [ ] Remove debug/test cells
- [ ] Remove unused imports
- [ ] Organize imports at top
- [ ] Run "Restart & Run All" - verify no errors
- [ ] Check for TODOs or placeholder text
- [ ] Add final conclusions section

**Git Finalization**:
- [ ] Commit all changes
- [ ] Write descriptive commit message
- [ ] Create GitHub repository (if not already)
- [ ] Push to GitHub
- [ ] Verify README displays correctly on GitHub
- [ ] Add topics/tags to repo (machine-learning, recommender-system, etc.)

**Final Checks**:
- [ ] Notebook runs end-to-end without errors
- [ ] All visualizations display correctly
- [ ] README is professional and complete
- [ ] Precision@10 ≥ 0.7 achieved (or documented why not)
- [ ] Can explain methodology to someone else
- [ ] Ready to share with potential employers

**Checkpoint Questions**:
- [ ] What's your overall precision@10?
- [ ] Which genre recommends best/worst?
- [ ] What are the main limitations of your system?
- [ ] What would you improve next?
