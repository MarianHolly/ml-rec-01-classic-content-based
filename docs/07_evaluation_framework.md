# PROJECT 1: CLASSIC CONTENT-BASED MOVIE RECOMMENDER
## DOCUMENT 7: EVALUATION FRAMEWORK

### 7.1 Overview

Evaluation is critical to understand if your recommender works and where it succeeds/fails. Unlike supervised ML (where you have labels), recommender systems are tricky to evaluate because "good" recommendations are subjective.

**Our Approach**: Multi-faceted evaluation combining quantitative metrics and qualitative analysis.

### 7.2 Evaluation Philosophy

**Key Questions We'll Answer**:
1. Do recommendations share content features with input? (Precision)
2. How similar are recommended movies? (Similarity scores)
3. Do recommendations "make sense" intuitively? (Qualitative review)
4. Are some types of movies recommended better than others? (Genre analysis)

**What We Won't Measure** (yet):
- User satisfaction (no user feedback data)
- Click-through rates (not a deployed system)
- Diversity vs. similarity tradeoff (v2 improvement)
- Coverage (what % of catalog gets recommended)

### 7.3 Primary Metric: Precision@K

**Definition**: Of the K recommendations, what fraction are relevant?

**Formula**:
```
Precision@K = (Number of relevant recommendations in top K) / K
```

**Relevance Definition** (you choose):
- **Option 1** (Stricter): Recommendation shares ≥ 2 genres with input movie
- **Option 2** (Looser): Recommendation shares ≥ 1 genre with input movie
- **Option 3** (Most lenient): Recommendation has similarity score ≥ 0.3

**Why Precision@K?**
- Easy to understand and interpret
- Directly measures content similarity
- Standard metric in information retrieval
- Can be computed without user feedback

**Implementation Strategy**:
```python
def calculate_precision_at_k(input_movie, recommendations, movies_df, k=10):
    """
    Calculate precision@k for recommendations.
    
    Args:
        input_movie (str): Input movie title
        recommendations (list): List of recommended movie titles
        movies_df (DataFrame): DataFrame with movie metadata
        k (int): Number of top recommendations to consider
    
    Returns:
        float: Precision score (0 to 1)
    """
    # Get genres of input movie
    input_genres = set(
        movies_df[movies_df['title'] == input_movie]['genres']
        .iloc[0].split('|')
    )
    
    relevant_count = 0
    
    # Check each recommendation
    for rec_title in recommendations[:k]:
        rec_genres = set(
            movies_df[movies_df['title'] == rec_title]['genres']
            .iloc[0].split('|')
        )
        
        # Check for genre overlap (at least 1 shared genre)
        if len(input_genres.intersection(rec_genres)) >= 1:
            relevant_count += 1
    
    return relevant_count / k
```

**Target Performance**:
- **Good**: Precision@10 ≥ 0.7 (70% relevant)
- **Acceptable**: Precision@10 ≥ 0.5 (50% relevant)
- **Poor**: Precision@10 < 0.5

### 7.4 Evaluation Dataset Selection

**Approach**: Create diverse test set to measure performance across different movie types.

**Sampling Strategy**:
```python
# Select 100 movies for evaluation, stratified by genre
eval_movies = []

# Top 5 genres
top_genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Romance']

for genre in top_genres:
    # Get movies with this genre
    genre_movies = movies_df[movies_df['genres'].str.contains(genre)]
    
    # Sample 20 movies
    sample = genre_movies.sample(20, random_state=42)
    eval_movies.extend(sample['title'].tolist())

print(f"Evaluation set: {len(eval_movies)} movies")
```

**What to Include**:
- Popular movies (have many tags, rich features)
- Obscure movies (few tags, sparse features)
- Old movies (pre-1990)
- Recent movies (2010+)
- Different genres (action, drama, comedy, horror, documentary, etc.)
- Movies from franchises (to test if sequels dominate)

**What to Exclude**:
- Movies without genres (can't calculate genre-based precision)
- Movies with only "(no genres listed)"

### 7.5 Comprehensive Evaluation Process

**Step 1: Calculate Precision Across Evaluation Set**

```python
# Evaluate all movies in evaluation set
results = []

for movie_title in eval_movies:
    # Get recommendations
    recs = get_recommendations(movie_title, n=10)
    
    # Calculate precision
    precision = calculate_precision_at_k(movie_title, recs, movies_df, k=10)
    
    # Get genres for analysis
    genres = movies_df[movies_df['title'] == movie_title]['genres'].iloc[0]
    
    # Store result
    results.append({
        'movie': movie_title,
        'precision': precision,
        'genres': genres
    })

results_df = pd.DataFrame(results)
```

**Step 2: Overall Statistics**

```python
# Summary statistics
print("=== Overall Precision@10 ===")
print(f"Mean: {results_df['precision'].mean():.3f}")
print(f"Median: {results_df['precision'].median():.3f}")
print(f"Std Dev: {results_df['precision'].std():.3f}")
print(f"Min: {results_df['precision'].min():.3f}")
print(f"Max: {results_df['precision'].max():.3f}")

# Visualize distribution
plt.figure(figsize=(10, 6))
plt.hist(results_df['precision'], bins=20, edgecolor='black')
plt.xlabel('Precision@10')
plt.ylabel('Number of Movies')
plt.title('Distribution of Precision@10 Scores')
plt.axvline(results_df['precision'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()
```

**Step 3: Precision by Genre**

```python
# Extract primary genre (first one listed)
results_df['primary_genre'] = results_df['genres'].str.split('|').str[0]

# Group by genre
genre_precision = results_df.groupby('primary_genre')['precision'].agg(['mean', 'count'])
genre_precision = genre_precision.sort_values('mean', ascending=False)

# Visualize
plt.figure(figsize=(12, 6))
genre_precision['mean'].plot(kind='bar')
plt.xlabel('Genre')
plt.ylabel('Average Precision@10')
plt.title('Recommendation Quality by Genre')
plt.xticks(rotation=45)
plt.axhline(0.7, color='red', linestyle='--', label='Target (0.7)')
plt.legend()
plt.tight_layout()
plt.show()

# Interpret
print("\n=== Precision by Genre ===")
print(genre_precision)
```

**Step 4: Best and Worst Cases**

```python
# Movies with best recommendations
best = results_df.nlargest(10, 'precision')
print("=== Top 10 Movies (Best Recommendations) ===")
print(best[['movie', 'precision', 'genres']])

# Movies with worst recommendations
worst = results_df.nsmallest(10, 'precision')
print("\n=== Bottom 10 Movies (Worst Recommendations) ===")
print(worst[['movie', 'precision', 'genres']])

# Analyze patterns
print("\nPattern Analysis:")
print(f"Best performers - Average # of genres: {best['genres'].str.count('\|').mean()}")
print(f"Worst performers - Average # of genres: {worst['genres'].str.count('\|').mean()}")
```

### 7.6 Similarity Score Analysis

**Purpose**: Understand the relationship between similarity scores and recommendation quality.

```python
# For evaluation set, get similarity scores of recommendations
similarity_data = []

for movie_title in eval_movies:
    # Get movie index
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Get recommendations with scores
    recs = get_recommendations_with_scores(movie_title, n=10)  # Enhanced function
    
    # Average similarity of top 10
    avg_similarity = np.mean([score for _, score in recs])
    
    # Get precision
    precision = results_df[results_df['movie'] == movie_title]['precision'].iloc[0]
    
    similarity_data.append({
        'movie': movie_title,
        'avg_similarity': avg_similarity,
        'precision': precision
    })

sim_df = pd.DataFrame(similarity_data)

# Visualize relationship
plt.figure(figsize=(10, 6))
plt.scatter(sim_df['avg_similarity'], sim_df['precision'], alpha=0.6)
plt.xlabel('Average Similarity Score')
plt.ylabel('Precision@10')
plt.title('Similarity vs. Precision')
plt.show()

# Correlation
correlation = sim_df['avg_similarity'].corr(sim_df['precision'])
print(f"Correlation between similarity and precision: {correlation:.3f}")
```

**Key Insights to Look For**:
- Are high-similarity recommendations always high-precision?
- Is there a minimum similarity threshold for relevant recommendations?
- Are there outliers (high similarity but low precision, or vice versa)?

### 7.7 Qualitative Review Process

**Why Qualitative?**: Numbers don't tell the full story. Manual review reveals nuances.

**Process**:
1. Select 10 diverse movies
2. For each, examine top 10 recommendations
3. Answer these questions:
   - Do recommendations make intuitive sense?
   - Are they too similar (all sequels)?
   - Do they capture the "spirit" of the input movie?
   - Would a real user find these helpful?

**Review Template**:

```markdown
### Movie: [Title]
**Input Genres**: [Genre list]
**Top 10 Recommendations**:
1. [Title 1] - [Genres] - ✓ / ✗ (relevant?)
2. [Title 2] - [Genres] - ✓ / ✗
...

**Observations**:
- [Note interesting patterns]
- [Note any obviously bad recommendations]
- [Note if sequels dominate]

**Overall Quality**: Good / Acceptable / Poor
```

**Example Review**:

```
### Movie: Toy Story (1995)
**Input Genres**: Adventure, Animation, Children, Comedy, Fantasy

**Top 10 Recommendations**:
1. Toy Story 2 (1999) - Adventure, Animation, Children, Comedy, Fantasy - ✓ (Obvious sequel)
2. Monsters, Inc. (2001) - Adventure, Animation, Children, Comedy, Fantasy - ✓ (Same studio, similar themes)
3. Finding Nemo (2003) - Adventure, Animation, Children, Comedy - ✓ (Pixar, family)
4. Toy Story 3 (2010) - Adventure, Animation, Children, Comedy, Fantasy - ✓ (Obvious sequel)
5. A Bug's Life (1998) - Adventure, Animation, Children, Comedy - ✓ (Pixar, family)
6. Shrek (2001) - Adventure, Animation, Children, Comedy, Fantasy, Romance - ✓ (Similar vibe)
7. The Incredibles (2004) - Action, Adventure, Animation, Children, Comedy - ✓ (Pixar)
8. Aladdin (1992) - Adventure, Animation, Children, Comedy, Fantasy, Musical, Romance - ✓ (Disney, family)
9. Up (2009) - Adventure, Animation, Children, Drama - ✓ (Pixar)
10. WALL·E (2008) - Adventure, Animation, Children, Romance, Sci-Fi - ✓ (Pixar)

**Observations**:
- Strong Pixar bias (7 out of 10) - makes sense, shared production quality and storytelling style
- All sequels present (expected, they share more than just genre)
- Good mix of animation studios (Disney, DreamWorks)
- All appropriate for children - captures "family film" aspect well

**Overall Quality**: Excellent
**Precision@10**: 1.0 (all share at least Adventure, Animation, Children)
```

### 7.8 Error Analysis

**Goal**: Understand *why* some recommendations fail.

**Categories of Failures**:

1. **Sparse Features**: Movie has only 1-2 genres, no tags
2. **Generic Features**: Movie has common genres (Drama), common tags (boring)
3. **Niche Content**: Very specific movie with no similar items
4. **Data Quality**: Incorrect or missing metadata

**Analysis Process**:

```python
# Identify low-precision movies
low_precision = results_df[results_df['precision'] < 0.5]

# For each, examine its features
for _, row in low_precision.iterrows():
    movie_title = row['movie']
    
    # Get its soup
    soup = movies_df[movies_df['title'] == movie_title]['soup'].iloc[0]
    
    # Analyze
    word_count = len(soup.split())
    unique_words = len(set(soup.split()))
    
    print(f"\n=== {movie_title} ===")
    print(f"Precision: {row['precision']:.2f}")
    print(f"Genres: {row['genres']}")
    print(f"Soup length: {word_count} words ({unique_words} unique)")
    print(f"Soup: {soup[:100]}...")  # First 100 chars
    
    # Hypothesis
    if word_count < 5:
        print("→ Likely failure cause: Sparse features")
    elif unique_words < 5:
        print("→ Likely failure cause: Generic/repetitive features")
    else:
        print("→ Likely failure cause: Niche content or data quality")
```

### 7.9 Baseline Comparison

**Why?**: To know if content-based filtering is better than simple baselines.

**Baseline 1: Random Recommendations**
```python
# Recommend 10 random movies
def random_recommendations(n=10):
    return movies_df.sample(n)['title'].tolist()

# Calculate precision for random
random_precisions = []
for movie in eval_movies:
    recs = random_recommendations()
    precision = calculate_precision_at_k(movie, recs, movies_df)
    random_precisions.append(precision)

print(f"Random baseline precision: {np.mean(random_precisions):.3f}")
```

**Baseline 2: Popular Recommendations**
```python
# Always recommend the 10 most popular movies (by rating count)
popular_movies = ratings.groupby('movieId').size().nlargest(10).index
popular_titles = movies_df[movies_df['movieId'].isin(popular_movies)]['title'].tolist()

def popular_recommendations():
    return popular_titles

# Calculate precision
popular_precisions = []
for movie in eval_movies:
    recs = popular_recommendations()
    precision = calculate_precision_at_k(movie, recs, movies_df)
    popular_precisions.append(precision)

print(f"Popular baseline precision: {np.mean(popular_precisions):.3f}")
```

**Comparison**:
```python
print("=== Baseline Comparison ===")
print(f"Random recommendations: {np.mean(random_precisions):.3f}")
print(f"Popular recommendations: {np.mean(popular_precisions):.3f}")
print(f"Content-based (our model): {results_df['precision'].mean():.3f}")
print(f"\nImprovement over random: {(results_df['precision'].mean() / np.mean(random_precisions) - 1) * 100:.1f}%")
print(f"Improvement over popular: {(results_df['precision'].mean() / np.mean(popular_precisions) - 1) * 100:.1f}%")
```

### 7.10 Results Interpretation Guidelines

**What Good Results Look Like**:
- Mean precision@10 ≥ 0.7
- Most genres perform reasonably well (≥ 0.6)
- Qualitative review shows sensible recommendations
- Clear improvement over random baseline

**Red Flags**:
- Very low precision (< 0.5) on average
- Huge variance in precision (some genres 0.9, others 0.2)
- Qualitative review shows nonsensical recommendations
- Barely better than random baseline

**How to Interpret**:

```
If precision is 0.7:
- "7 out of 10 recommendations share at least one genre with the input"
- "This means users will likely find most recommendations relevant"
- "There's still room for improvement (capturing themes beyond genres)"

If Action movies have precision 0.85 but Romance has 0.55:
- "Action movies recommend well because they have distinctive features (explosions, fighting tags)"
- "Romance recommendations struggle - 'romantic' is common across many genres"
- "Future: Add more specific features (cinematography style, mood) to help Romance"
```

### 7.11 Documentation Requirements

**In Your Notebook**:
1. **Evaluation Methodology Section**: Explain what you're measuring and why
2. **Results Section**: Show all metrics, visualizations, statistics
3. **Analysis Section**: Interpret results, identify patterns
4. **Limitations Section**: What your evaluation doesn't capture
5. **Conclusions**: Summarize findings

**In Your README**:
- Brief results summary (1-2 paragraphs)
- Key metric: "Achieved precision@10 of 0.73"
- Notable findings: "Animation and Sci-Fi recommend best"
- Link to notebook for detailed evaluation

### 7.12 Future Evaluation Enhancements

**For Project 2 (Enhanced Recommender)**:
- Diversity metric: How different are top 10 recommendations?
- Novelty: Are we recommending obscure movies or always popular ones?
- Coverage: What % of catalog ever gets recommended?
- Temporal analysis: Do old movies recommend as well as new ones?

**For Project 3 (Niche Specialist)**:
- Domain-specific precision: Do film noir recommendations capture noir essence?
- Expert review: Have a film noir enthusiast review recommendations
- Comparative: How does specialist compare to general recommender?
