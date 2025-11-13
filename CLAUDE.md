# CLAUDE.md - AI Assistant Guide
## Classic Content-Based Movie Recommender System

> **Purpose**: This document provides comprehensive guidance for AI assistants working with this repository. It explains the codebase structure, development workflows, conventions, and best practices to follow.

---

## Table of Contents
1. [Repository Overview](#repository-overview)
2. [Project Structure](#project-structure)
3. [Documentation Architecture](#documentation-architecture)
4. [Development Workflow](#development-workflow)
5. [Technical Specifications](#technical-specifications)
6. [Code Conventions](#code-conventions)
7. [Git Workflow](#git-workflow)
8. [AI Assistant Guidelines](#ai-assistant-guidelines)
9. [Quick Reference](#quick-reference)

---

## Repository Overview

### Project Type
**Machine Learning Education Project - Documentation Blueprint**

This is a **documentation-only repository** that serves as a comprehensive blueprint for building a content-based movie recommendation system. The repository does NOT currently contain implementation code, notebooks, or data files - only detailed documentation to guide development.

### Key Characteristics
- **Domain**: Recommender Systems (Content-Based Filtering)
- **Dataset**: MovieLens 25M (62,000+ movies)
- **Primary Technologies**: Python, pandas, NumPy, scikit-learn, Jupyter
- **Project Status**: Documentation complete, implementation pending
- **Target Audience**: Intermediate Python learners with beginner pandas/NumPy skills
- **Time Commitment**: 20-30 hours over 2-3 weeks

### Project Goals
1. Build a content-based movie recommender using TF-IDF and cosine similarity
2. Achieve Precision@10 ≥ 0.7 (70% of recommendations are relevant)
3. Create professional, well-documented Jupyter notebook
4. Learn full ML pipeline: data → features → model → evaluation

### Repository Status
```
Current Branch: claude/claude-md-mhxwwc0zqpn925vv-01BPFhdKSVqCHHrKHoB9esvP
Last Commit: fb638ba - "documentation to project"
Git Status: Clean (no uncommitted changes)
```

---

## Project Structure

### Current Directory Layout
```
ml-rec-01-classic-content-based/
├── .git/                              # Git version control
├── docs/                              # Complete project documentation (10 files)
│   ├── 00_START_HERE.md              # Entry point and navigation guide
│   ├── 01_project_brief.md           # Goals, scope, deliverables
│   ├── 02_technical_specification.md # Architecture and algorithms
│   ├── 03_implementation_plan.md     # 7-phase development roadmap
│   ├── 04_data_guide.md              # Dataset schema and preprocessing
│   ├── 05_code_structure.md          # Code organization guidelines
│   ├── 06_development_checklist.md   # 200+ actionable tasks
│   ├── 07_evaluation_framework.md    # Metrics and evaluation methods
│   ├── QUICK_REFERENCE.md            # Condensed reference guide
│   └── requirements.txt              # Python dependencies
└── CLAUDE.md                          # This file (AI assistant guide)
```

### Expected Project Structure (After Implementation)
```
ml-rec-01-classic-content-based/
├── data/
│   └── ml-25m/                       # MovieLens dataset (to be downloaded)
│       ├── movies.csv                # 62k movies with titles and genres
│       ├── tags.csv                  # 1.09M user-generated tags
│       └── ratings.csv               # 25M ratings (optional)
├── notebooks/
│   └── movie_recommender.ipynb       # Main implementation notebook
├── models/                           # Saved models (optional)
│   ├── tfidf_vectorizer.pkl
│   └── similarity_matrix.npy
├── docs/                             # Documentation (already exists)
├── .gitignore                        # Git ignore patterns
├── README.md                         # Project overview (to be created)
├── requirements.txt                  # Python dependencies
└── CLAUDE.md                         # This file
```

---

## Documentation Architecture

### Document Hierarchy and Usage

#### **Entry Point**
- **docs/00_START_HERE.md** - Read this first for navigation and overview

#### **Planning Documents** (Read Before Coding)
1. **docs/01_project_brief.md** - Understand project goals and success criteria
2. **docs/02_technical_specification.md** - Understand algorithms and architecture
3. **docs/03_implementation_plan.md** - Understand 7-phase development roadmap

#### **Reference Documents** (Consult During Development)
4. **docs/04_data_guide.md** - Dataset schema, preprocessing strategies
5. **docs/05_code_structure.md** - Code organization, naming conventions
6. **docs/06_development_checklist.md** - Task-by-task breakdown (200+ items)
7. **docs/07_evaluation_framework.md** - Metrics and evaluation methods

#### **Quick Access**
- **docs/QUICK_REFERENCE.md** - Code snippets, common patterns, troubleshooting

### Key Document Purposes

| Document | When to Reference | Key Information |
|----------|------------------|-----------------|
| `00_START_HERE.md` | Beginning of project | Navigation, learning path, success checklist |
| `01_project_brief.md` | Planning phase | Problem statement, goals, scope, deliverables |
| `02_technical_specification.md` | Architecture decisions | TF-IDF, cosine similarity, feature engineering |
| `03_implementation_plan.md` | During development | 7-phase roadmap with time estimates |
| `04_data_guide.md` | Working with data | CSV schemas, preprocessing code examples |
| `05_code_structure.md` | Writing code | Notebook organization, naming conventions |
| `06_development_checklist.md` | Task tracking | Concrete tasks organized by phase |
| `07_evaluation_framework.md` | Evaluation phase | Precision@K implementation and analysis |
| `QUICK_REFERENCE.md` | Anytime | Code snippets, parameters, debugging |

---

## Development Workflow

### 7-Phase Implementation Plan

#### **Phase 1: Environment Setup (1-2 hours)**
- Create project directory structure
- Set up virtual environment
- Install dependencies from `docs/requirements.txt`
- Download MovieLens 25M dataset
- Initialize Git repository

**Deliverable**: Working development environment with data

#### **Phase 2: Data Loading & EDA (4-6 hours)**
- Load CSV files into pandas DataFrames
- Explore data structure with `.info()`, `.describe()`
- Visualize genre distribution
- Analyze tag coverage
- Document initial observations

**Deliverable**: Comprehensive understanding of data

#### **Phase 3: Data Cleaning (3-4 hours)**
- Handle missing values in genres and tags
- Aggregate tags per movie (groupby + join)
- Merge movies and tags DataFrames
- Create master DataFrame with all features
- Validate data quality

**Deliverable**: Clean, merged dataset ready for feature engineering

#### **Phase 4: Feature Engineering (3-4 hours)**
- Create "soup" column combining genres and tags
- Preprocess text (lowercase, remove special characters)
- Handle movies without tags
- Validate feature quality

**Deliverable**: Dataset with engineered text features

#### **Phase 5: Vectorization & Similarity (2-3 hours)**
- Apply TF-IDF vectorization (max_features=5000)
- Calculate cosine similarity matrix
- Save similarity matrix for reuse
- Verify matrix properties (diagonal = 1.0)

**Deliverable**: Similarity matrix ready for recommendations

#### **Phase 6: Recommendation Function (2-3 hours)**
- Implement `get_recommendations()` function
- Handle edge cases (movie not found, etc.)
- Test with sample movies
- Verify output quality

**Deliverable**: Working recommendation function

#### **Phase 7: Evaluation & Documentation (4-6 hours)**
- Implement Precision@K metric
- Evaluate on test set
- Create visualizations
- Write README.md
- Clean up notebook
- Final Git commit

**Deliverable**: Complete, evaluated, documented project

### Checkpoint Pattern

After each phase, verify completion with checkpoint cells:

```python
# === CHECKPOINT: [Phase Name] ===
assert condition1, "Error message"
assert condition2, "Error message"
print("✓ [Phase Name] checkpoint passed")
```

---

## Technical Specifications

### Core Algorithms

#### **1. TF-IDF Vectorization**
**Purpose**: Convert text features (genres, tags) into numerical vectors

**Key Concepts**:
- TF (Term Frequency): Rewards words appearing often in document
- IDF (Inverse Document Frequency): Penalizes common words across all documents
- Result: Rare, distinctive words get higher weight

**Implementation**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,      # Limit vocabulary size
    stop_words='english'     # Remove common words
)
tfidf_matrix = tfidf.fit_transform(movies_df['soup'])
```

**Parameters**:
- `max_features=5000` - Balance between feature richness and computational cost
- `stop_words='english'` - Remove "the", "is", "a", etc.

#### **2. Cosine Similarity**
**Purpose**: Measure similarity between movie feature vectors

**Key Concepts**:
- Range: 0 (completely different) to 1 (identical)
- Measures angle between vectors, not magnitude
- Perfect for comparing TF-IDF vectors

**Implementation**:
```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# Result: N×N matrix where N = number of movies
```

**Properties**:
- Diagonal = 1.0 (each movie is identical to itself)
- Symmetric matrix (similarity(A,B) = similarity(B,A))
- Dense matrix for 62k movies ≈ 30GB (consider sparse storage)

#### **3. Content-Based Filtering Logic**
**Workflow**:
1. Represent each movie as feature vector (TF-IDF)
2. Calculate similarity between all movie pairs (cosine similarity)
3. For input movie X, find top N movies with highest similarity
4. Return recommendations sorted by similarity score

**Recommendation Function**:
```python
def get_recommendations(title, n=10):
    """Get n most similar movies to input title."""
    # Find movie index
    idx = movies_df[movies_df['title'] == title].index[0]

    # Get similarity scores for this movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity (descending), exclude input movie itself
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return movie titles
    return movies_df['title'].iloc[movie_indices].tolist()
```

### Evaluation Metrics

#### **Precision@K**
**Definition**: Of K recommendations, what percentage are relevant?

**Relevance Criterion**: Recommendation shares at least one genre with input movie

**Formula**: `Precision@K = (# relevant recommendations) / K`

**Target**: ≥ 0.7 (70% of recommendations should be relevant)

**Implementation**:
```python
def calculate_precision_at_k(input_movie, recommendations, movies_df, k=10):
    """Calculate precision@K metric."""
    input_genres = set(movies_df[movies_df['title'] == input_movie]['genres'].iloc[0].split('|'))

    relevant_count = 0
    for rec_title in recommendations[:k]:
        rec_genres = set(movies_df[movies_df['title'] == rec_title]['genres'].iloc[0].split('|'))
        if len(input_genres.intersection(rec_genres)) >= 1:
            relevant_count += 1

    return relevant_count / k
```

### Dependencies

**Python Version**: 3.8+

**Core Libraries** (from `docs/requirements.txt`):
```
numpy==1.24.3          # Numerical operations, arrays
pandas==2.0.3          # Data manipulation, DataFrames
scikit-learn==1.3.0    # TF-IDF, cosine similarity, metrics
matplotlib==3.7.2      # Basic plotting
seaborn==0.12.2        # Statistical visualizations
jupyter==1.0.0         # Notebook environment
notebook==7.0.2        # Jupyter notebook server
ipykernel==6.25.1      # Jupyter kernel
joblib==1.3.2          # Save/load models
tqdm==4.66.1           # Progress bars (optional)
```

**Installation**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r docs/requirements.txt
```

---

## Code Conventions

### Naming Conventions

#### **DataFrames**
```python
movies              # Original movies data
tags                # Original tags data
movies_clean        # After cleaning
movies_with_tags    # After merging
movies_features     # After feature engineering
```

#### **Matrices and Arrays**
```python
tfidf_matrix        # TF-IDF vectors
cosine_sim          # Similarity matrix
```

#### **Functions**
Use verb-noun pattern with descriptive names:
```python
get_recommendations()           # Main recommendation function
calculate_precision_at_k()      # Evaluation function
plot_similarity_distribution()  # Visualization function
get_movie_index()               # Helper function
```

#### **Variables**
```python
n_recommendations   # Number of recommendations
movie_idx          # Movie index
sim_scores         # Similarity scores
MAX_FEATURES       # Constants in UPPERCASE
DATA_PATH          # File path constants
```

### Notebook Organization

#### **Standard Notebook Structure** (10 sections)
1. **Setup & Imports** - Environment info, library imports, configuration
2. **Data Loading** - Read CSV files, initial display
3. **Exploratory Data Analysis** - Statistics, visualizations, insights
4. **Data Cleaning** - Handle missing values, merge datasets
5. **Feature Engineering** - Create "soup" column, preprocess text
6. **Model Building** - TF-IDF vectorization, similarity matrix
7. **Recommendation Function** - Define and test function
8. **Evaluation** - Precision@K, visualizations, analysis
9. **Results & Conclusions** - Summary of findings
10. **Future Work** - Next steps and improvements

#### **Cell Organization Pattern**
```
[Markdown] Explain what you're about to do and why
[Code] Implement the operation
[Code] Visualize results (if applicable)
[Markdown] Interpret results and explain next steps
```

#### **Checkpoint Cells**
Add verification cells at key milestones:
```python
# === CHECKPOINT: Data Loading ===
assert 'movieId' in movies.columns
assert len(movies) > 60000
assert len(tags) > 1000000
print("✓ Data loading checkpoint passed")
```

### Feature Engineering Pattern

**"Soup" Creation** - Combine all text features into single column:
```python
# Combine genres and tags
movies_features['soup'] = (
    movies_features['genres'].fillna('') + ' ' +
    movies_features['tag'].fillna('')
).str.lower().str.replace('|', ' ')
```

**Why "soup"?**: Industry term for combined text features fed into vectorizer

### Documentation Standards

#### **Markdown Cells**
- Use `##` for main sections, `###` for subsections
- Explain reasoning and decisions, not just steps
- Document what didn't work and why
- Add visual separators (`---`) between major sections

#### **Code Comments**
- Explain *why*, not *what* (code shows what)
- Add docstrings to all functions
- Use inline comments sparingly for complex logic

#### **Function Docstrings**
```python
def get_recommendations(title, similarity_matrix, movies_df, n=10):
    """
    Get movie recommendations based on content similarity.

    Args:
        title (str): Movie title exactly as it appears in dataset
        similarity_matrix (ndarray): Precomputed cosine similarity matrix
        movies_df (DataFrame): Movies DataFrame with titles and features
        n (int, optional): Number of recommendations to return. Defaults to 10.

    Returns:
        list: List of n recommended movie titles

    Raises:
        ValueError: If movie title not found in dataset

    Example:
        >>> get_recommendations("Toy Story (1995)", cosine_sim, movies, n=5)
        ['Toy Story 2 (1999)', 'Monsters, Inc. (2001)', ...]
    """
    # Implementation here
```

---

## Git Workflow

### Branch Strategy
- **Current Development Branch**: `claude/claude-md-mhxwwc0zqpn925vv-01BPFhdKSVqCHHrKHoB9esvP`
- All development should occur on this branch
- Branch naming pattern: `claude/claude-md-{session-id}`

### Commit Message Conventions

**Format**: `<type>: <description>`

**Types**:
- `feat` - New feature or functionality
- `fix` - Bug fix
- `docs` - Documentation changes
- `refactor` - Code refactoring without behavior change
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

**Examples**:
```bash
git commit -m "feat: Complete Phase 1 environment setup"
git commit -m "feat: Add data loading and initial EDA"
git commit -m "fix: Handle movies without tags in feature engineering"
git commit -m "feat: Implement recommendation function"
git commit -m "feat: Add Precision@K evaluation metric"
git commit -m "docs: Complete project README and documentation"
```

### Recommended Commit Points
1. After each phase completion
2. After fixing significant bugs
3. After adding major features
4. Before and after refactoring
5. Final commit when project is complete

### What to Track in Git

**Include**:
- ✅ Notebooks (`.ipynb`)
- ✅ Documentation files (`.md`)
- ✅ Requirements file (`requirements.txt`)
- ✅ README and configuration files
- ✅ `.gitignore` file

**Exclude** (add to `.gitignore`):
```
# Data files (too large)
data/
*.csv

# Model files (large binary files)
models/*.pkl
models/*.npy

# Jupyter checkpoints
.ipynb_checkpoints/

# Python cache
__pycache__/
*.pyc

# System files
.DS_Store
Thumbs.db
```

### Git Push Requirements
```bash
# Always use -u flag to set upstream
git push -u origin claude/claude-md-mhxwwc0zqpn925vv-01BPFhdKSVqCHHrKHoB9esvP

# CRITICAL: Branch must start with 'claude/' and end with session ID
# Otherwise push will fail with 403 error
```

**Retry Strategy**: If push fails due to network errors, retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s)

---

## AI Assistant Guidelines

### How to Approach This Repository

#### **1. Initial Orientation**
When first engaging with this repository:
1. Read `docs/00_START_HERE.md` for overview
2. Read `docs/01_project_brief.md` to understand goals
3. Read `docs/02_technical_specification.md` for technical details
4. Scan `docs/QUICK_REFERENCE.md` for quick patterns

#### **2. Understanding User Intent**

**If user asks to implement the project:**
1. Follow `docs/03_implementation_plan.md` phase by phase
2. Reference `docs/06_development_checklist.md` for specific tasks
3. Use `docs/04_data_guide.md` for data operations
4. Follow conventions in `docs/05_code_structure.md`
5. Implement evaluation using `docs/07_evaluation_framework.md`

**If user asks questions about the project:**
- Reference relevant documentation files
- Explain concepts from `docs/02_technical_specification.md`
- Provide code examples from `docs/QUICK_REFERENCE.md`

**If user asks to modify documentation:**
- Ensure consistency across all 10 documentation files
- Maintain the educational, beginner-friendly tone
- Update version information if significant changes made

#### **3. Development Best Practices**

**Phase-Based Development**:
- Work through phases sequentially (1 → 7)
- Complete all tasks in a phase before moving to next
- Add checkpoint cells at phase boundaries
- Commit after each phase completion

**Code Quality**:
- Follow naming conventions strictly
- Add comprehensive docstrings to functions
- Include markdown explanations before code cells
- Use checkpoint assertions to verify correctness
- Print intermediate results for verification

**Documentation**:
- Add markdown cells explaining reasoning
- Document what worked and what didn't
- Include interpretations of visualizations
- Write clear, educational comments

**Testing**:
- Test recommendation function with diverse inputs
- Verify edge cases (movie not found, no tags, etc.)
- Calculate metrics on multiple test movies
- Manually review sample recommendations

#### **4. Common Tasks and Approaches**

**Task: Help implement Phase 1 (Environment Setup)**
```python
# Create directory structure
# Set up virtual environment
# Install dependencies
# Download dataset
# Initialize Git
# Create initial notebook structure
```

**Task: Help with feature engineering**
- Reference `docs/04_data_guide.md` for preprocessing strategies
- Follow "soup" creation pattern from `docs/QUICK_REFERENCE.md`
- Verify feature quality with checkpoint cells

**Task: Debug recommendation issues**
1. Check similarity matrix properties (diagonal = 1.0)
2. Verify TF-IDF parameters (max_features, stop_words)
3. Inspect "soup" quality (length, content)
4. Test with known similar movies

**Task: Improve precision**
- Add more features (directors, keywords if available)
- Adjust TF-IDF parameters
- Filter out movies with sparse features
- Consider weighted feature combination

#### **5. Communication Style**

**Educational Approach**:
- Explain concepts clearly, assuming beginner pandas/NumPy skills
- Provide reasoning for decisions, not just code
- Reference relevant documentation sections
- Break complex operations into steps

**Code Examples**:
- Include complete, runnable code snippets
- Add comments explaining key lines
- Show expected output
- Demonstrate error handling

**Problem-Solving**:
- Use checkpoint cells to isolate issues
- Print intermediate DataFrames to inspect
- Suggest debugging steps from `docs/QUICK_REFERENCE.md`
- Recommend consulting specific documentation sections

### Key Principles for AI Assistants

1. **Follow the Documentation**: This repository has comprehensive docs - use them
2. **Phase-Based Approach**: Don't skip ahead, build foundation first
3. **Educational Focus**: Explain concepts, don't just provide solutions
4. **Quality Over Speed**: Proper documentation and testing matter
5. **Convention Adherence**: Follow naming and structure guidelines strictly
6. **Checkpoint Everything**: Verify each step before proceeding
7. **Git Hygiene**: Commit regularly with meaningful messages

### Red Flags to Avoid

❌ **Don't**:
- Skip phases in the implementation plan
- Ignore checkpoint failures
- Use inconsistent naming conventions
- Skip documentation in notebook
- Commit without testing
- Push to wrong branch
- Ignore evaluation metrics
- Create code without understanding the algorithms

✅ **Do**:
- Follow the 7-phase plan sequentially
- Add checkpoint cells and verify they pass
- Use standard naming from conventions section
- Add markdown explanations throughout
- Test thoroughly before committing
- Push to designated branch with -u flag
- Achieve Precision@10 ≥ 0.7 target
- Understand TF-IDF and cosine similarity concepts

---

## Quick Reference

### Essential File Paths
```
/home/user/ml-rec-01-classic-content-based/docs/00_START_HERE.md
/home/user/ml-rec-01-classic-content-based/docs/01_project_brief.md
/home/user/ml-rec-01-classic-content-based/docs/02_technical_specification.md
/home/user/ml-rec-01-classic-content-based/docs/03_implementation_plan.md
/home/user/ml-rec-01-classic-content-based/docs/04_data_guide.md
/home/user/ml-rec-01-classic-content-based/docs/05_code_structure.md
/home/user/ml-rec-01-classic-content-based/docs/06_development_checklist.md
/home/user/ml-rec-01-classic-content-based/docs/07_evaluation_framework.md
/home/user/ml-rec-01-classic-content-based/docs/QUICK_REFERENCE.md
/home/user/ml-rec-01-classic-content-based/docs/requirements.txt
```

### Key Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_features` | 5000 | TF-IDF vocabulary size |
| `stop_words` | 'english' | Remove common words |
| `n_recommendations` | 10 | Standard recommendation count |
| `precision@10_target` | ≥ 0.7 | Success metric |

### Common Code Patterns

**Load Data**:
```python
movies = pd.read_csv('data/ml-25m/movies.csv')
tags = pd.read_csv('data/ml-25m/tags.csv')
```

**Aggregate Tags**:
```python
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = movies.merge(movie_tags, on='movieId', how='left')
```

**Create Soup**:
```python
movies_df['soup'] = (
    movies_df['genres'].fillna('') + ' ' +
    movies_df['tag'].fillna('')
).str.lower().str.replace('|', ' ')
```

**Build Model**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['soup'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

**Get Recommendations**:
```python
def get_recommendations(title, n=10):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()
```

### Success Criteria Checklist
- [ ] Notebook runs end-to-end without errors
- [ ] Precision@10 ≥ 0.7 on test set
- [ ] All visualizations present and properly labeled
- [ ] Professional README.md created
- [ ] Clean Git history with meaningful commits
- [ ] Code follows naming conventions
- [ ] Comprehensive documentation in notebook
- [ ] Can explain how the system works

---

## Version Information

**CLAUDE.md Version**: 1.0
**Created**: 2024-11-13
**Last Updated**: 2024-11-13
**Compatible With**: Project 1 Documentation v1.0
**Repository**: ml-rec-01-classic-content-based

---

## Additional Resources

### When to Consult Which Document

| Working On... | Primary Reference | Supporting References |
|--------------|-------------------|----------------------|
| Project planning | 01_project_brief.md | 03_implementation_plan.md |
| Algorithm understanding | 02_technical_specification.md | QUICK_REFERENCE.md |
| Phase execution | 03_implementation_plan.md | 06_development_checklist.md |
| Data operations | 04_data_guide.md | QUICK_REFERENCE.md |
| Code writing | 05_code_structure.md | QUICK_REFERENCE.md |
| Task tracking | 06_development_checklist.md | 03_implementation_plan.md |
| Evaluation | 07_evaluation_framework.md | 02_technical_specification.md |
| Quick lookup | QUICK_REFERENCE.md | Any relevant guide |

### Dataset Information
- **Name**: MovieLens 25M
- **Source**: https://grouplens.org/datasets/movielens/25m/
- **Size**: ~250 MB compressed
- **Movies**: 62,423
- **Tags**: 1,093,360
- **Ratings**: 25,000,095 (optional for this project)

### Expected Outcomes

**Technical Deliverables**:
- One comprehensive Jupyter notebook (300-500 cells)
- Precision@10 ≥ 0.7
- Professional README.md
- Clean Git repository

**Learning Outcomes**:
- pandas mastery (filtering, merging, grouping)
- Feature engineering for ML
- Text vectorization with TF-IDF
- Similarity calculations
- ML evaluation without labeled data
- Professional documentation skills

---

## Contact and Support

For questions about this documentation or the project:
1. Review the specific documentation file for your phase
2. Check `QUICK_REFERENCE.md` for common solutions
3. Consult `06_development_checklist.md` for task breakdowns
4. Review `02_technical_specification.md` for algorithm details

---

**Remember**: This is a learning project. The goal is understanding, not just completion. Take time to experiment, document your thinking, and build a portfolio piece you're proud of.

**Happy coding!** 🎬🤖
