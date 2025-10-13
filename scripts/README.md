/scripts
├── get_rating_of_users.py           # Fetch rating lists for users from review files
├── get_rating_of_books.py           # Fetch ratings for books based on a list of book_ids
├── build_goodreads_dataset.py       # Merge all rating files into a unified dataset
├── count_books.py                   # Count the number of  books reviewed by users
├── goodreads_reviews__{book_id}.json  # Original review files containing user_id
├── goodreads_ratings_{user_id}.json  # Rating results for individual users
└── goodreads_dataset.csv            # Merged training dataset (userid, bookid, rating)



# Data Collection and Crawling Strategy

## Background

This project aims to build a book recommendation system based on **user similarity**. The core idea of recommendation systems is that if two users share similar reading and rating behaviors, they are likely to have similar future preferences. To achieve this, we need to collect a large dataset containing user–book rating information. Goodreads, an online community with abundant books and reviews, serves as an ideal data source.

## Data Collection Approach

### Identify target books

We start from Goodreads’ popular book lists (Top 100 Books of the year or Most Read Books), and collect key information such as title, author, average rating and genre.

### Expand user review data

Then for each popular book,  we scrape the review section to extract user information. After getting the user's ID, we can visit each user’s profile page to obtain their ratings and reviews of other books unless they set their profiles private.

### Build the user–book rating dataset

Combine all data into a structured dataset as follows:

user_id | book_id | book_title | rating

Finally, count how many times each book appears, and retain only books with **sufficient review** frequency for reliable analysis.

## Scripts Analysis

The data collection process begins by carefully inspecting the Goodreads web pages and their HTML structure. By viewing the page source in a browser, we can identify the elements that contain the information we need, such as book titles, user ratings, review text, authors, and review dates. We then use CSS selectors in `BeautifulSoup` to locate these elements, for example `tr.bookalike.review` for each user review row or `.ShelfStatus span[aria-label*="out of 5"]` for ratings. Because Goodreads pages may have alternative layouts depending on the user or book, the code includes fallback logic to extract data from cover image cells, interactive star ratings, or other possible locations. For books or users with many reviews, pagination is handled by iterating through all pages until no new data is found. To avoid being blocked, custom headers, cookies, and randomized delays are used to simulate human browsing behavior, and retry mechanisms handle temporary network errors or rate limits. The collected data is finally stored in structured JSON files, preserving all relevant fields such as book ID, title, rating, read dates, shelves, and review URLs, which serve as the final dataset for building the recommendation system.

### get_rating_of_books.py

#### Overview

This script automatically scrapes user review data for popular books from the Goodreads website and saves the results as JSON files. It supports batch crawling of multiple books, includes retry mechanisms for network stability, and implements anti-blocking delay controls. The collected data will later be used to build the user–book rating dataset for the recommendation system.

#### Module Used

`requests`: Handles HTTP requests and web page retrieval.

`BeautifulSoup`: Parses HTML content to extract target elements from the webpage.

`json`: Saves the scraped data in JSON format for structured storage.

`pandas`: Reads and manages the list of target books (from `book_counts.csv`).It is used to identify books with relatively few ratings or reviews (less than 10) and perform additional crawling on these underrepresented books. This helps ensure a more balanced dataset with sufficient coverage across different books.

`time` & `random`: Controls randomized delays between requests to reduce blocking risk.

`HTTPAdapter` & `Retry`: Implements automatic reconnection and retry logic for failed requests.

#### Core Function

**`extract_book_details()`：**

This function extracts key information from a book’s HTML page, including the book title, average rating, and all user reviews listed on that page. Each review record contains the user name, user ID, rating score, review date, textual review content, as well as interaction data such as likes and comments. It uses CSS selectors in `BeautifulSoup` to locate and extract the corresponding HTML elements. The function returns the book’s basic information together with a list of all reviews found on the page.

Goodreads stores user ratings in `aria-label` like `"4 out of 5 stars"`, which is easy to extract programmatically.

```python
rating_elem = review_div.select_one('.ShelfStatus span[aria-label*="out of 5"]')
user_rating = rating_elem['aria-label'] if rating_elem and 'aria-label' in rating_elem.attrs else None
```

At last we can get a whole review:

```python
reviews.append({
            'user': user,
            'user_id': user_id,
            'user_rating': user_rating,
            'review_date': review_date,
            'review': review_text,
            'likes': likes,
            'comments': comments
        })
```

**`scrape_goodreads_all_reviews()`：**

This function performs large-scale crawling of all review pages for a given Goodreads book. It manages pagination, handles network errors, and ensures reliability by using the `Retry` and `HTTPAdapter` mechanisms for automatic retries. Random time delays are introduced between page requests to reduce the risk of being blocked by the website. All extracted reviews are aggregated into a single list and returned along with the book’s title and average rating.

**`save_reviews_to_json()`：**

This function stores the collected data in a structured JSON format. Each book’s review data is saved as a file named after its book ID (e.g., `goodreads_reviews_12345.json`). 



In the main execution block, the script reads a CSV file (`book_counts.csv`) using `pandas`, identifies books with relatively few reviews (e.g., fewer than ten), and performs additional crawling on these underrepresented books. It automatically skips books that have already been processed, allowing the crawler to resume from previous progress and improving efficiency.

### get_rating_of_users.py

#### Overview

This script crawls user rating and review data from Goodreads. It reads **book review** JSON files previously collected and then iterates through the users who wrote these reviews, fetching their full rating histories. The script uses sessions with retry mechanisms, cookies, headers, and randomized delays to avoid IP blocking and handle network instability. The collected data is organized in JSON files per user and serves as the primary dataset for the recommendation system.

The final json file is like:

```JSON
{
  "source_user_id": 123456,
  "count": 1,
  "items": [
    {
      "book_id": 10222,
      "book_title": "The Far Pavilions",
      "book_url": "https://www.goodreads.com/book/show/10222.The_Far_Pavilions",
      "author": "Kaye, M.M.",
      "user_rating_text": "it was amazing",
      "user_rating": 5,
      "date_read": "not set",
      "date_added": "Jan 07, 2007",
      "shelves": [],
      "review_url": "https://www.goodreads.com/review/show/7338"
    }
  ]
}

```

#### Core Function

**`parse_list_page()`**

This function parses a Goodreads user review list page and extracts detailed information for each book the user has rated. For every book row (`<tr>`), it collects the book title, author, Goodreads URL, and numeric book ID. The function also retrieves the user’s rating (both text and numeric value), read date, date added to shelves, associated shelves/tags, and the review URL.

It handles multiple HTML structures: if the book title or ID is not found in the main title cell, it attempts to extract them from the cover image cell or alternative HTML elements. Rating extraction prioritizes static stars but falls back to interactive stars if needed. The function uses CSS selectors in `BeautifulSoup` for element identification and returns a list of all books with their metadata and user-specific information.

```python
soup = BeautifulSoup(html, "lxml")
    rows = soup.select("tr.bookalike.review") or soup.select("table#books tr") or []
for tr in rows:
    td_rating = None
        for td in tr.select("td.field"):
            label = td.select_one("label")
            if label and label.get_text(strip=True).lower().endswith("'s rating"):
                td_rating = td
                break
        if td_rating is None:
            td_rating = tr.select_one("td.field.rating")
```

**`extract_book_details()`**

This function extracts key information from a book’s HTML page, including the book title, average rating, and all user reviews listed on that page. Each review record contains the user name, user ID, rating score, review date, textual review content, as well as interaction data such as likes and comments. It uses CSS selectors in `BeautifulSoup` to locate and extract the corresponding HTML elements. The function returns the book’s basic information together with a list of all reviews found on the page.

**`scrape_goodreads_all_reviews()`**

This function crawls review pages of a given book on Goodreads. It handles pagination, network errors, and rate limiting by using session-based requests with retry and backoff strategies. Randomized delays between requests are introduced to avoid IP blocking. The function aggregates all extracted reviews and returns the book’s title, average rating, and a list of all reviews.

### count_books.py

This script is designed to process all user rating JSON files and extract a unique identifier for each book, called a "book slug," from the Goodreads URLs. By iterating over files that contain individual users’ ratings, it reads the list of rated books (`items`) in each JSON, extracts the `book_url` for each book, and then parses out the slug using a regular expression. A `Counter` object tallies how many times each book appears across all users, effectively counting the number of ratings per book. The final output is written to a CSV file (`book_counts.csv`) with two columns: `book_slug` and `count`, sorted by the most frequently reviewed books. This approach allows us to identify which books have sufficient ratings to be included in the recommendation system, while low-frequency books can be filtered out to focus the analysis on titles with enough user data.

#### build_goodreads_dataset.py

This script processes all the individual user rating JSON files collected from Goodreads and combines them into a single structured dataset suitable for recommendation system analysis. It searches the current directory for files matching the pattern `goodreads_ratings_{user_id}.json`, extracts the `user_id` from the filename, and reads the JSON content. For each book in the user's rating list (`items`), it retrieves the `book_id` and the `user_rating`. Only entries with both valid book ID and rating are included. The script stores these records in a Python list and then converts it into a Pandas DataFrame with columns `userid`, `bookid`, and `rating`. Finally, the dataset is saved as a CSV file (`goodreads_dataset.csv`). This dataset represents the final structured input for the recommendation system, containing all users, all books they rated, and the corresponding rating values.

