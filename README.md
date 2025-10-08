## 🎓 Course Overview

This project was developed as part of the [Introduction to Data Science](https://courses.mooc.fi/org/uh-cs/courses/introduction-to-data-science) course at the **University of Helsinki**.

This project integrates both **data analytics** and **machine learning** components, demonstrating end-to-end data science workflow — from data exploration and visualization to model training and evaluation.

---

## Data source

- Data about book https://www.goodreads.com/
  - You can get personal data like ratings used for train a book recommendation system
  - Personal data divided by region (change the parameter in the url)
    - https://www.goodreads.com/user/best_reviewers?country=FI&duration=a
    - https://www.goodreads.com/user/best_reviewers?country=US&duration=a
  - Book list with meaningful filters
    - https://www.goodreads.com/list/
  - collect data by **spider** (maybe)
---

## Target

- Book recommendation system
  - User-Based Collaborative Filtering Algorithm
  - no good visiulization
- Show Trend about book with visualization (by region, by time, by genre, by language ....)
- the average ratings from one region
- .....

---

## Data Collect

---

## 🔹 ALS Recommendation Module
This module implements an **ALS-based collaborative filtering model** using PySpark MLlib.
It provides utilities for data preprocessing, model training, hyperparameter tuning, evaluation, and incremental updates.

| Function                                                                                 | Description                                                                   |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `init_spark(app_name="BookRecommendation")`                                              | Initializes a Spark session with proper configurations.                       |
| `load_and_preprocess_data(spark, dataset_path)`                                          | Loads and cleans the dataset, handling type casting and indexing if needed.   |
| `train_als_model(train_df, rank, maxIter, regParam)`                                     | Trains a baseline ALS model with specified parameters.                        |
| `train_als_model_with_tuning(train_df)`                                                  | Runs grid search to find the best combination of rank, regParam, and maxIter. |
| `predict_and_evaluate(model, test_df)`                                                   | Evaluates model performance using RMSE.                                       |
| `save_model(model, path)` / `load_model(path)`                                           | Saves or loads a trained ALS model.                                           |
| `update_model_with_new_data(spark, old_dataset_path, new_dataset_path, model_save_path)` | Merges new data with existing data, retrains, and saves the updated model.    |

## 📊 Book Data Analysis Module
This module performs **exploratory data analysis (EDA)** and visualization on the `books.csv` dataset.
It provides insights into book ratings, publication trends, language distributions, and popular titles through various charts and plots.

| Category                  | Description                                                    | Visualization Type         |
| ------------------------- | -------------------------------------------------------------- | -------------------------- |
| **Rating Distribution**   | Distribution of average ratings among all books                | Histogram & Pie Chart      |
| **Publication Trends**    | Number of books published per year, and average rating by year | Line Charts                |
| **Language Distribution** | Number of books by language (including non-English)            | Horizontal Bar Charts      |
| **Top Rated Books**       | Top 10 books with the highest rating counts                    | Bar Chart (average rating) |
| **Rating Composition**    | Breakdown of 1–5 star ratings for top-rated books              | Stacked Bar Chart          |
| **Cold Books**            | High-rated books with few ratings                              | Stacked Bar Chart          |
| **Word Cloud**            | Common words appearing in book titles                          | Word Cloud                 |

### Output Examples

- 📈 Average Rating Distribution – shows the most common rating ranges

- 📉 Publication Trend – number of books published per year

- 🌍 Language Distribution – identifies the most common languages

- ⭐ Top 10 Most Rated Books – shows popularity and rating balance

- 🧊 Cold Book Detection – finds underrated gems

- ☁️ Word Cloud – highlights common words in book titles

