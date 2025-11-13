# PROJECT 1: CLASSIC CONTENT-BASED MOVIE RECOMMENDER
## DOCUMENT 4: DATA GUIDE

### 4.1 Dataset Overview

**Name**: MovieLens 25M Dataset
**Provider**: GroupLens Research (University of Minnesota)
**License**: Free for research and education
**Last Updated**: 2019
**Size**: ~250 MB compressed

**Why This Dataset?**
- Industry standard for recommender system research
- Clean, well-structured data (minimal preprocessing needed)
- Rich metadata (genres, tags)
- Large enough to be realistic, small enough to process locally
- Excellent documentation

### 4.2 Download Instructions

**Step 1**: Visit official page
```
https://grouplens.org/datasets/movielens/25m/
```

**Step 2**: Download `ml-25m.zip` (265 MB)

**Step 3**: Extract to your project directory
```
your-project/
  ├── data/
  │   ├── ml-25m/
  │   │   ├── movies.csv
  │   │   ├── tags.csv
  │   │   ├── ratings.csv
  │   │   ├── links.csv
  │   │   ├── genome-scores.csv
  │   │   ├── genome-tags.csv
  │   │   └── README.txt
  ├── notebooks/
  │   └── movie_recommender.ipynb
  └── README.md
```

**Alternative**: Use direct download link (in notebook):
```python
import urllib.request
import zipfile

url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
urllib.request.urlretrieve(url, "ml-25m.zip")

with zipfile.ZipFile("ml-25m.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")
```

### 4.3 Data Schema

#### 4.3.1 movies.csv

**Purpose**: Core movie metadata

**Structure**:
```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
...
```

**Columns**:
- `movieId` (int): Unique identifier for each movie
- `title` (string): Movie title with release year in parentheses
- `genres` (string): Pipe-separated list of genres

**Size**: 62,423 movies

**Key Characteristics**:
- All movies have at least one genre (or "(no genres listed)")
- Titles include year, which helps disambiguate remakes
- Genre values: 20 unique genres (Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IUPUI, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western)

**Data Quality**:
- Very clean, minimal missing values
- Some movies have "(no genres listed)" - you'll need to handle these
- Titles are unique

#### 4.3.2 tags.csv

**Purpose**: User-generated tags (keywords) for movies

**Structure**:
```
userId,movieId,tag,timestamp
336,1,pixar,1139045763
474,1,pixar,1164885435
567,1,fun,1525286180
...
```

**Columns**:
- `userId` (int): User who created the tag
- `movieId` (int): Movie being tagged
- `tag` (string): Free-form text tag
- `timestamp` (int): Unix timestamp when tag was created

**Size**: 1,093,360 tags from 17,045 users

**Key Characteristics**:
- Multiple users can tag the same movie with the same tag
- Tags are free-form text (typos, variations exist)
- Not all movies have tags (~40% have at least one tag)
- Tag frequency varies widely

**Data Quality**:
- Some tags are single words ("funny"), others are phrases ("thought-provoking")
- Case inconsistent ("Pixar" vs "pixar")
- Some tags are very specific, others very general
- You'll need to aggregate multiple tags per movie

**Example Tags for Toy Story**:
- "pixar", "fun", "animation", "childhood", "toys", "adventure"

#### 4.3.3 ratings.csv (Optional)

**Purpose**: User ratings for movies (1-5 stars)

**Structure**:
```
userId,movieId,rating,timestamp
1,296,5.0,1147880044
1,306,3.5,1147868817
1,307,5.0,1147868828
...
```

**Columns**:
- `userId` (int): User who rated
- `movieId` (int): Movie being rated
- `rating` (float): Rating from 0.5 to 5.0 (half-star increments)
- `timestamp` (int): Unix timestamp of rating

**Size**: 25,000,095 ratings from 162,541 users

**Note**: We won't use this for content-based filtering, but you can use it for evaluation (e.g., test if high-rated movies get good recommendations).

### 4.4 Data Preprocessing Strategy

#### Step 1: Load Data
```python
import pandas as pd

# Load movies
movies = pd.read_csv('data/ml-25m/movies.csv')

# Load tags
tags = pd.read_csv('data/ml-25m/tags.csv')

# Optional: Load ratings for evaluation
ratings = pd.read_csv('data/ml-25m/ratings.csv')
```

#### Step 2: Explore Structure
```python
# Check shape and columns
print(movies.shape)  # (62423, 3)
print(movies.columns)  # ['movieId', 'title', 'genres']

# Check for missing values
print(movies.isnull().sum())

# Look at sample rows
print(movies.head(10))
```

#### Step 3: Clean Genres
```python
# Handle "(no genres listed)" - replace with empty string or 'Unknown'
movies['genres'] = movies['genres'].replace('(no genres listed)', '')

# Split genres for analysis (but keep original for feature engineering)
movies['genre_list'] = movies['genres'].str.split('|')
```

#### Step 4: Aggregate Tags
```python
# Goal: One row per movie with all tags combined

# Group by movieId and concatenate all tags into single string
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# This creates a DataFrame:
# movieId | tag
# 1       | "pixar fun animation childhood toys adventure"
# 2       | "adventure game jungle"
```

**Why Join Tags?**
- Multiple users tag the same movie
- We want all descriptive tags combined
- More tags = richer features = better recommendations

**Handling Tag Frequency** (optional enhancement):
```python
# Some tags appear multiple times (multiple users used same tag)
# You can weight by frequency or just include each unique tag once

# Option 1: Keep duplicates (natural weighting)
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))

# Option 2: Unique tags only
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x.unique()))
```

#### Step 5: Merge Data
```python
# Merge movies with aggregated tags
movies_with_tags = movies.merge(movie_tags, on='movieId', how='left')

# 'how=left' ensures we keep all movies, even those without tags
# Movies without tags will have NaN in 'tag' column
```

#### Step 6: Create Feature String ("Soup")
```python
# Combine genres and tags into single text feature
movies_with_tags['soup'] = (
    movies_with_tags['genres'].fillna('') + ' ' + 
    movies_with_tags['tag'].fillna('')
)

# Clean the soup
movies_with_tags['soup'] = movies_with_tags['soup'].str.lower()
movies_with_tags['soup'] = movies_with_tags['soup'].str.replace('|', ' ')

# Example soup for Toy Story:
# "adventure animation children comedy fantasy pixar fun childhood toys"
```

### 4.5 Train/Test Split Strategy

**Purpose**: Reserve some movies for testing recommendation quality.

**Approach**: Random split at movie level
```python
from sklearn.model_selection import train_test_split

# 80/20 split
train_movies, test_movies = train_test_split(
    movies_with_tags, 
    test_size=0.2, 
    random_state=42  # For reproducibility
)

print(f"Training movies: {len(train_movies)}")  # ~50k
print(f"Testing movies: {len(test_movies)}")    # ~12k
```

**How to Use**:
- Build similarity matrix on `train_movies`
- Evaluate recommendations on `test_movies`
- Check: Do test movies get good recommendations from training set?

**Note**: For simplicity in first version, you can skip train/test split and evaluate on full dataset.

### 4.6 Data Quality Issues to Watch For

**Issue 1**: Movies Without Tags
- ~60% of movies have no user tags
- **Impact**: These movies will have only genre features (less distinctive)
- **Solution**: Acceptable for v1; consider augmenting with external data in v2

**Issue 2**: Sparse Genres
- Some movies have only one genre
- **Impact**: Less information for similarity calculation
- **Solution**: This is reality; evaluation will show which movies recommend poorly

**Issue 3**: Tag Noise
- Tags can be misspelled, subjective, or irrelevant ("seen it", "to watch")
- **Impact**: Adds noise to features
- **Solution**: TF-IDF will downweight common/useless tags; advanced filtering in v2

**Issue 4**: Old Movies
- Dataset contains movies from 1900s to 2019
- Older movies often have fewer tags (fewer users have seen them)
- **Impact**: Bias toward popular, recent movies in recommendations
- **Solution**: Document as limitation; could add release year as feature in v2

**Issue 5**: Sequels and Series
- Many franchises in dataset (Harry Potter, Star Wars)
- **Impact**: High similarity between sequels (often recommended)
- **Solution**: This is actually good! But consider adding diversity metric in evaluation

### 4.7 Quick Data Validation Checklist

Before proceeding to feature engineering, verify:

- [ ] `movies.csv` loaded successfully
- [ ] `tags.csv` loaded successfully
- [ ] Number of unique `movieId` in movies matches expected (~62k)
- [ ] No null values in `movieId` or `title` columns
- [ ] Tags aggregated to one row per movie
- [ ] Merge between movies and tags completed (check resulting row count)
- [ ] "soup" column created for all movies
- [ ] At least 95% of movies have non-empty "soup"

**Validation Code**:
```python
# Run these checks in your notebook
assert movies['movieId'].nunique() == len(movies), "Duplicate movieIds!"
assert movies['title'].isnull().sum() == 0, "Null titles found!"
assert movies_with_tags['soup'].isnull().sum() == 0, "Null soups found!"

print("✓ All data validation checks passed!")
```

### 4.8 Sample Data Inspection

**Always inspect sample data** to build intuition:

```python
# Show a few complete records
sample = movies_with_tags.sample(5, random_state=42)
for idx, row in sample.iterrows():
    print(f"\nTitle: {row['title']}")
    print(f"Genres: {row['genres']}")
    print(f"Tags: {row['tag'][:100]}...")  # First 100 chars
    print(f"Soup: {row['soup'][:100]}...")
```

This helps you understand:
- What your features actually look like
- Whether preprocessing worked correctly
- If anything looks suspicious
