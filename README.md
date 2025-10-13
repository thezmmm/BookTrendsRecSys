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
---

## Data Collect
This dataset contains detailed ratings and metadata for 10,000 of the most popular books, covering a total of six million user ratings. It is primarily used for analyzing user preferences, recommendation system modeling, and book popularity studies.

- Book Ratings

Contains six million ratings from users for the 10,000 most popular books.

Ratings are typically on a scale from 1 to 5.
- Book Metadata

Each row represents a book and contains the following attributes:

| Column Name                 | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `book_id`                   | Unique internal identifier for each book.         |
| `goodreads_book_id`         | Book ID as listed on Goodreads.                   |
| `best_book_id`              | Best edition ID on Goodreads.                     |
| `work_id`                   | Identifier for the work, grouping all editions.   |
| `books_count`               | Number of editions of this book.                  |
| `isbn`                      | 10-digit ISBN of the book edition.                |
| `isbn13`                    | 13-digit ISBN of the book edition.                |
| `authors`                   | Names of the authors of the book.                 |
| `original_publication_year` | Year the book was originally published.           |
| `original_title`            | Original title of the book (if different).        |
| `title`                     | Main title of the book.                           |
| `language_code`             | Language code of the book (e.g., `en`).           |
| `average_rating`            | Average user rating for this book.                |
| `ratings_count`             | Number of ratings for this edition.               |
| `work_ratings_count`        | Total number of ratings across all editions.      |
| `work_text_reviews_count`   | Total number of text reviews across all editions. |
| `ratings_1` … `ratings_5`   | Count of ratings at each level (1–5 stars).       |
| `image_url`                 | URL of the book's main image.                     |
| `small_image_url`           | URL of a smaller version of the book image.       |

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

### Results
The ALS model was evaluated on the test set, achieving a RMSE of `0.8230`.

Interpretation:

On average, predicted ratings deviate by ~0.82 points on the 1–5 scale.

The model effectively captures user-book preferences and handles sparse rating data.

The model provides reasonably accurate book recommendations, helping users discover new books based on their previous ratings.


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
