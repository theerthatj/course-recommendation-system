# course-recommendation-system
Developed a machine learning-powered course recommendation system that integrates courses from multiple online learning platforms and generates personalized recommendations using TF-IDF vectorization and content-based filtering.

---

## Problem Statement

With thousands of online courses spread across multiple platforms, learners
often struggle to find courses that match their skills, interests, and career
goals. This project builds a recommendation system that takes course content
(title, skills, level, platform, instructor/partner) and, optionally, a
learner's profile, and returns a ranked list of the most relevant courses.

---

## Dataset

- **Source:** [Multi-Platform Online Courses Dataset](https://www.kaggle.com/datasets/everydaycodings/multi-platform-online-courses-dataset) (Kaggle)
- **Platforms covered:** Coursera, Udemy, edX, Skillshare
- **Raw combined size:** 42,461 courses across all four platforms
- **Working sample:** 2,000 courses (500 Coursera + 700 Udemy + 300 edX + 500 Skillshare, `random_state=42`), used to keep TF-IDF and similarity computations fast
- **Key fields used:** course title, skills/description, level, rating, review count, partner/instructor, platform

---

## Project Pipeline

1. **Data Loading & Standardization** - load the four platform CSVs and map each
   to a common schema (`course`, `skills`, `rating`, `reviewcount`, `level`,
   `partner`, `duration`, `platform`).
2. **Sampling & Cleaning** - combine into a single `master_df`, sample down to
   2,000 rows, and fill missing values (skills, level, partner, duration,
   rating, review count).
3. **Feature Engineering** - build a combined `features` text field
   (course + skills + level + partner + platform) and compute normalized
   `rating_score` and `popularity_score` columns.
4. **Content-Based Modeling**
   - TF-IDF vectorization of `features` (5,000 max features, English stop words removed)
   - Cosine similarity matrix across all courses
   - A K-Nearest Neighbors model (cosine distance) over the TF-IDF matrix as an
     alternative retrieval method
5. **Recommendation Functions** (see below)
6. **Exploratory Data Analysis** - distribution plots for platform, course
   level, and rating, plus average rating by platform.
7. **Evaluation** - proxy Precision/Recall/F1/MAP/NDCG@K and a Precision@K curve.

---

## Recommendation Functions

| Function | Description |
|---|---|
| `recommend_courses(course_name, top_n=5)` | Finds courses similar to a given course/title using TF-IDF cosine similarity, blended with rating and popularity (`0.7 × similarity + 0.2 × rating_score + 0.1 × popularity_score`). |
| `recommend_courses_knn(skill_query, top_n=5)` | Returns the nearest courses to a free-text skill/topic query using the KNN model over the TF-IDF space. |
| `recommend_for_learner(skills, interests=None, level=None, top_n=10)` | Builds a learner profile from a list of skills/interests, optionally filters by course level, and returns the top matching courses ranked by similarity + rating + popularity. |
| `generate_learning_path(topic, top_n_per_level=2)` | Generates a sequenced learning path for a topic - top courses at Beginner, then Intermediate, then Advanced level - to support progressive skill-building. |

---

## Evaluation

Recommendation quality is evaluated using Precision@K, Recall@K, F1@K,
MAP@K, and NDCG@K, averaged over a random sample of 135 courses (top_n = 10):

| Metric | Score |
|---|---|
| Precision@10 | 0.598 |
| Recall@10 | 0.093 |
| F1-Score@10 | 0.096 |
| MAP@10 | 0.060 |
| NDCG@10 | 0.803 |

A Precision@K curve (K = 5, 10, 15, 20, 25, 30) is also plotted to show how
recommendation quality changes as the list length grows.

> **Important caveat:** Since the dataset has no real user-interaction or
> feedback data, "relevant" courses are defined as a proxy - courses that
> share at least 2 overlapping skill keywords with the query course's own
> TF-IDF features. These metrics are a **self-consistency check on the
> similarity ranking**, not a validation against real learner behavior. They
> should be read directionally (e.g., the high NDCG@10 indicates the most
> similar courses tend to be ranked first; the low Recall@10 reflects that the
> proxy "relevant set" is often much larger than 10 items) rather than as
> proof of real-world recommendation quality.

---

## Exploratory Data Analysis

The notebook includes the following visualizations:
- Course distribution across platforms
- Course level distribution
- Course rating distribution
- Average rating by platform

---

## Outputs

All generated artifacts are written to `../outputs/`:

- `evaluation_results_per_course.csv` - per-course evaluation metrics
- `evaluation_summary.csv` - averaged evaluation metrics
- `metric_comparison.png` - bar chart of evaluation metrics
- `precision_at_k_curve.png` - Precision@K curve
- `cybersecurity_recommendations.csv` - example output of `recommend_courses`
- `python_knn_recommendations.csv` - example output of `recommend_courses_knn`
- `recommend_for_learner_output.csv` - example learner-profile recommendation
- `ml_learning_path.csv` - example generated learning path for "Machine Learning"

---

## Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Modeling:** Scikit-learn (TF-IDF, cosine similarity, K-Nearest Neighbors)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## How to Run

1. Place the source CSVs in `../data/`:
   - `Coursera.csv`
   - `Udemy.csv`
   - `edx.csv`
   - `skillshare.csv`
2. Create an `../outputs/` directory for generated files.
3. Run the notebook cells top to bottom.
4. Example usage:

```python
# Find courses similar to a given course
recommend_courses("Machine Learning", top_n=5)

# Find courses for a free-text skill query (KNN)
recommend_courses_knn("Python", top_n=5)

# Recommend courses based on a learner profile
recommend_for_learner(
    skills=["python", "machine learning", "statistics"],
    interests=["data science"],
    level="Beginner",
    top_n=10
)

# Generate a beginner-to-advanced learning path for a topic
generate_learning_path("Machine Learning")
```

---

## Limitations & Future Work

- **No collaborative filtering:** The dataset contains course metadata only -
  no user-level interaction/rating history - so the system is purely
  content-based. A collaborative or hybrid approach would require either real
  user interaction logs or simulated interaction data.
- **Proxy evaluation:** As noted above, evaluation metrics use a skill-overlap
  proxy for relevance rather than real user feedback. Future work should
  incorporate real user feedback (clicks, completions, ratings) for genuine
  validation.
- **Sample size:** The model currently runs on a 2,000-course sample for
  performance; this could be scaled up to the full ~42,000-course dataset.
- **Planned enhancements:**
  - Incorporate real-time user feedback and behavioral signals
  - Expand to multi-modal recommendations (videos, articles, certifications)
  - Explore deep learning–based embeddings for richer course representations
  - Extend learning path generation across multiple related topics/skill chains