# ML Zoomcamp – Module 03: Classification

My homework for Module 03 (Classification) of the DataTalksClub Machine Learning Zoomcamp, 2026 cohort. The notebook predicts whether a lead converts into a course sign-up, using logistic regression from scikit-learn.

## Dataset

The [Course Lead Scoring dataset](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv), which the notebook loads straight from the course repo.

- **Categorical features:** `lead_source`, `industry`, `employment_status`, `location`
- **Numerical features:** `annual_income`, `number_of_courses_viewed`, `interaction_count`, `lead_score`
- **Target:** `converted` (0/1)

## What the notebook covers

1. **Data preparation:** fills missing categorical values with `'NA'` and missing numerical values with `0.0`.
2. **EDA:** finds the most frequent `industry` and the correlation matrix of the numerical features.
3. **Train/validation/test split:** 60/20/20 with `random_state=42`.
4. **Mutual information:** scores each categorical feature against `converted` on the training set.
5. **Logistic regression:** one-hot encodes with `DictVectorizer` and trains with `liblinear`, `C=1.0`.
6. **Feature elimination:** drops one feature at a time and measures the change in validation accuracy.
7. **Regularization:** tries several values of `C`.

## Results

| Question | Answer |
|---|---|
| Most frequent `industry` | `technology` |
| Most correlated pair of numerical features | `interaction_count` & `lead_score` (0.916) |
| Highest mutual information | `lead_source` (0.03) |
| Validation accuracy, all features | 0.65 |
| Feature whose removal changes accuracy least | `number_of_courses_viewed` (0.002) |
| Best `C` (of 1e-6, 1e-5, 1e-4, 1e-3) | 0.001 (accuracy 0.645) |

## How to run

    pip install numpy pandas scikit-learn jupyter
    jupyter notebook 03-Classification.ipynb

You need an internet connection because the dataset loads from GitHub.
