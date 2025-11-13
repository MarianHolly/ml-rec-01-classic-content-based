# PROJECT 1: CLASSIC CONTENT-BASED MOVIE RECOMMENDER
## DOCUMENT 5: CODE STRUCTURE / PROJECT ORGANIZATION

### 5.1 Directory Structure

```
movie-recommender/
│
├── data/
│   └── ml-25m/              # MovieLens dataset (downloaded)
│       ├── movies.csv
│       ├── tags.csv
│       ├── ratings.csv      # Optional
│       └── ...
│
├── notebooks/
│   └── movie_recommender.ipynb  # Main notebook (all code here)
│
├── models/                  # Optional: saved models
│   ├── tfidf_vectorizer.pkl
│   └── similarity_matrix.npy
│
├── .gitignore              # Git ignore file
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

### 5.2 Notebook Organization

Since you're working in pure Jupyter notebooks, organize your notebook with clear sections. This structure keeps your work organized and makes it easy for others (and future you) to navigate.

**Recommended Notebook Structure**:

```
====================================
MOVIE RECOMMENDATION SYSTEM
Content-Based Filtering Approach
====================================

[Markdown Cell]
## Table of Contents
1. Setup & Imports
2. Data Loading
3. Exploratory Data Analysis
4. Data Cleaning & Preprocessing
5. Feature Engineering
6. Model Building
7. Recommendation Function
8. Evaluation
9. Results & Conclusions
10. Future Work

---

[Markdown Cell]
## 1. Setup & Imports

### 1.1 Environment Information
[Code Cell]
# Document your environment
import sys
print(f"Python version: {sys.version}")
import pandas as pd
print(f"Pandas version: {pd.__version__}")
# ... other versions

### 1.2 Library Imports
[Code Cell]
# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

### 1.3 Configuration
[Code Cell]
# File paths
DATA_PATH = '../data/ml-25m/'
MOVIES_FILE = DATA_PATH + 'movies.csv'
TAGS_FILE = DATA_PATH + 'tags.csv'

# Hyperparameters
MAX_FEATURES = 5000  # TF-IDF max features
N_RECOMMENDATIONS = 10  # Default number of recommendations

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

---

[Markdown Cell]
## 2. Data Loading

### 2.1 Load Movies Dataset
[Code Cell]
# Load movies
movies = pd.read_csv(MOVIES_FILE)
print(f"Loaded {len(movies)} movies")
movies.head()

[Markdown Cell]
**Observations:**
- Dataset has X rows and Y columns
- Column descriptions: movieId (unique ID), title (name + year), genres (pipe-separated)

### 2.2 Load Tags Dataset
[Code Cell]
# Load tags
tags = pd.read_csv(TAGS_FILE)
print(f"Loaded {len(tags)} tags")
tags.head()

---

[Markdown Cell]
## 3. Exploratory Data Analysis

### 3.1 Movies Dataset Overview
[Code Cell]
# Basic statistics
movies.info()
print("\nMissing values:")
print(movies.isnull().sum())
print("\nData types:")
print(movies.dtypes)

### 3.2 Genre Distribution
[Code Cell]
# Analyze genres
genre_counts = movies['genres'].str.split('|').explode().value_counts()
print("Top 10 Genres:")
print(genre_counts.head(10))

# Visualize
plt.figure(figsize=(12, 6))
genre_counts.head(15).plot(kind='bar')
plt.title('Top 15 Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

[Markdown Cell]
**Insights:**
- Drama is the most common genre
- Action, Comedy, Thriller also very common
- Film-Noir is rare (important for our future niche project!)

### 3.3 Tags Analysis
[Code Cell]
# How many movies have tags?
movies_with_tags = tags['movieId'].nunique()
total_movies = movies['movieId'].nunique()
print(f"{movies_with_tags} out of {total_movies} movies have at least one tag")
print(f"That's {movies_with_tags/total_movies*100:.1f}%")

# Tag distribution per movie
tags_per_movie = tags.groupby('movieId').size()
tags_per_movie.describe()

---

[Continue with similar structure for remaining sections...]
```

### 5.3 Cell Organization Best Practices

**Markdown Cells**:
- Use `##` for main sections, `###` for subsections
- Start each section with markdown explaining what you'll do
- End each section with markdown summarizing findings
- Include your reasoning for decisions

**Code Cells**:
- One logical operation per cell (or closely related operations)
- Add comments explaining *why*, not just *what*
- Use descriptive variable names
- Print intermediate results to verify correctness

**Visualization Cells**:
- One visualization per cell (easier to adjust)
- Always include title, axis labels, legend if needed
- Add interpretation in markdown cell immediately after

**Example Section Flow**:
```
[Markdown] Explain what you're about to do
[Code] Do the thing
[Code] Visualize the result (if applicable)
[Markdown] Interpret the result and explain next steps
```

### 5.4 Naming Conventions

**DataFrames**:
- `movies` - original movies data
- `tags` - original tags data
- `movies_clean` - after cleaning
- `movies_with_tags` - after merging
- `movies_features` - after feature engineering

**Matrices/Arrays**:
- `tfidf_matrix` - TF-IDF vectors
- `cosine_sim` - similarity matrix

**Functions**:
- Use descriptive names: `get_recommendations()`, not `rec()`
- Use verb-noun pattern: `calculate_precision()`, `plot_distribution()`

**Variables**:
- `n_recommendations` - number of recommendations
- `movie_idx` - movie index
- `sim_scores` - similarity scores

### 5.5 Function Definitions

Even though you're in notebooks, define reusable functions. Put all function definitions in a dedicated section early in the notebook.

**Recommended Functions Section**:

```
[Markdown Cell]
## Helper Functions

[Code Cell]
def get_movie_index(title, df):
    """
    Find DataFrame index for a given movie title.
    
    Args:
        title (str): Movie title
        df (DataFrame): Movies DataFrame
    
    Returns:
        int: Index of movie
    
    Raises:
        ValueError: If movie not found
    """
    try:
        return df[df['title'] == title].index[0]
    except IndexError:
        raise ValueError(f"Movie '{title}' not found in dataset")

def get_recommendations(title, similarity_matrix, movies_df, n=10):
    """
    Get movie recommendations based on similarity.
    
    [Detailed docstring...]
    """
    # Implementation
    pass

def calculate_precision_at_k(recommendations, input_movie, movies_df, k=10):
    """
    Calculate precision@k for recommendations.
    
    [Detailed docstring...]
    """
    # Implementation
    pass

def plot_similarity_distribution(similarity_matrix):
    """
    Visualize distribution of similarity scores.
    
    [Detailed docstring...]
    """
    # Implementation
    pass
```

### 5.6 Saving Intermediate Results

For long computations (TF-IDF, similarity matrix), save results so you don't have to recompute:

```python
# After computing similarity matrix
import numpy as np
np.save('models/similarity_matrix.npy', cosine_sim)

# Later, load it
cosine_sim = np.load('models/similarity_matrix.npy')
```

For TF-IDF vectorizer (if you want to vectorize new data):
```python
import joblib
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')

# Later, load it
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
```

### 5.7 Checkpoint Cells

Add checkpoint cells at key milestones to verify everything is working:

```python
# === CHECKPOINT: Data Loading ===
assert 'movieId' in movies.columns, "movieId column missing!"
assert len(movies) > 60000, "Movies dataset too small!"
assert len(tags) > 1000000, "Tags dataset too small!"
print("✓ Data loading checkpoint passed")

# === CHECKPOINT: Feature Engineering ===
assert 'soup' in movies_features.columns, "soup column missing!"
assert movies_features['soup'].isnull().sum() == 0, "Null soups exist!"
assert movies_features['soup'].str.len().mean() > 10, "Soups too short!"
print("✓ Feature engineering checkpoint passed")

# === CHECKPOINT: Model Building ===
assert tfidf_matrix.shape[0] == len(movies_features), "TF-IDF shape mismatch!"
assert cosine_sim.shape == (len(movies_features), len(movies_features)), "Similarity matrix shape wrong!"
assert np.allclose(np.diag(cosine_sim), 1.0), "Diagonal should be 1.0!"
print("✓ Model building checkpoint passed")
```

### 5.8 Version Control Strategy

**Git Workflow**:

```bash
# Initial commit
git init
git add .
git commit -m "Initial commit: project structure"

# After each major phase
git add .
git commit -m "Complete Phase 2: Data loading and EDA"

# After fixing issues
git add .
git commit -m "Fix: Handle movies without tags in feature engineering"

# Final
git add .
git commit -m "Complete Project 1: Working recommendation system with evaluation"
```

**What to Commit**:
- ✅ Notebooks (.ipynb)
- ✅ README.md
- ✅ requirements.txt
- ✅ .gitignore
- ✅ Documentation files

**What NOT to Commit** (add to .gitignore):
```
# .gitignore content
data/
models/*.pkl
models/*.npy
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
```

### 5.9 Documentation Within Notebook

**Use Markdown Liberally**:
- Explain your thought process
- Document why you made decisions
- Note what didn't work and why
- Add visual separators between major sections

**Example Documentation Style**:
```markdown
## Why I Chose TF-IDF Over Count Vectorizer

I initially considered using simple word counts, but TF-IDF has advantages:
1. **Rare words get higher weight** - "Film-Noir" is more distinctive than "Drama"
2. **Common words get downweighted** - "movie", "film" appear everywhere
3. **Industry standard** - easier for others to understand

I tested both approaches (see code below) and TF-IDF gave 15% better precision@10.
```

### 5.10 Notebook Cleanup Checklist

Before considering your notebook "done":

- [ ] All cells run in order without errors
- [ ] No debugging/test cells left in
- [ ] All imports at the top
- [ ] Clear section headers with numbering
- [ ] Markdown explanations for each section
- [ ] Visualizations have titles and labels
- [ ] Functions have docstrings
- [ ] Results are interpreted, not just displayed
- [ ] Checkpoint cells verify correctness
- [ ] Final conclusion section summarizes findings
