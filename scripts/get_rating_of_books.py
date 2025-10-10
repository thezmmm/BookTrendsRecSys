import requests
from bs4 import BeautifulSoup
import json
import re
import time
import pandas as pd
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import random

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

def extract_book_details(html):
    soup = BeautifulSoup(html, 'html.parser')
    title_elem = soup.select_one('h1.H1Title a[data-testid="title"]')
    title = title_elem.text.strip() if title_elem else None

    rating_elem = soup.select_one('div.RatingStatistics span.RatingStars')
    rating = rating_elem['aria-label'] if rating_elem and 'aria-label' in rating_elem.attrs else None

    reviews = []
    for review_div in soup.select('article.ReviewCard'):
        # 用户名 & 用户ID
        user_elem = review_div.select_one('.ReviewerProfile__name a')
        user = user_elem.text.strip() if user_elem else None
        user_id = None
        if user_elem and 'href' in user_elem.attrs:
            href = user_elem['href']
            if "/user/show/" in href:
                user_id = href.split("/user/show/")[-1].split("-")[0]

        # 用户评分
        rating_elem = review_div.select_one('.ShelfStatus span[aria-label*="out of 5"]')
        user_rating = rating_elem['aria-label'] if rating_elem and 'aria-label' in rating_elem.attrs else None

        # 评论时间
        date_elem = review_div.select_one('.ReviewCard__row span.Text a')
        review_date = date_elem.text.strip() if date_elem else None

        # 评论内容
        review_text_elem = review_div.select_one('.ReviewText__content span.Formatted')
        review_text = review_text_elem.get_text(" ", strip=True) if review_text_elem else None

        # 点赞数
        likes_elem = review_div.select_one('footer.SocialFooter button span:contains("likes")')
        likes = likes_elem.text.strip() if likes_elem else None

        # 评论数
        comments_elem = review_div.select_one('footer.SocialFooter button span:contains("comments")')
        comments = comments_elem.text.strip() if comments_elem else None

        reviews.append({
            'user': user,
            'user_id': user_id,
            'user_rating': user_rating,
            'review_date': review_date,
            'review': review_text,
            'likes': likes,
            'comments': comments
        })

    return title, rating, reviews

# 爬取 Goodreads 所有评论
def scrape_goodreads_all_reviews(base_url, max_page, start_page=1, headers=None):
    all_reviews = []
    page = start_page
    title, rating = None, None

    # ---- 创建带重试机制的 Session ----
    session = requests.Session()
    retries = Retry(
        total=5,                  # 最多重试5次
        backoff_factor=2,         # 每次重试间隔指数递增（2, 4, 8... 秒）
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    while page <= max_page:
        url = f"{base_url}?page={page}"
        try:
            response = session.get(url, headers=headers, timeout=(10, 60))
            if response.status_code != 200:
                print(f"[{response.status_code}] Skipping page {page}")
                break

            t, r, reviews = extract_book_details(response.text)
            if not reviews:
                print(f"[Empty] No reviews on page {page}, stopping.")
                break

            if not title:
                title, rating = t, r

            all_reviews.extend(reviews)
            print(f"[OK] Page {page}: {len(all_reviews)} total reviews")

        except requests.exceptions.ReadTimeout:
            print(f"[Timeout] Page {page} timed out, retrying later...")
            time.sleep(random.uniform(5, 10))
            page += 1
            continue

        except requests.exceptions.ConnectionError as e:
            print(f"[ConnectionError] {e}, retrying after delay...")
            time.sleep(random.uniform(10, 20))
            continue

        except requests.exceptions.RequestException as e:
            print(f"[Error] Unexpected error on page {page}: {e}")
            break

        # 防止被封：每页后暂停 2~5 秒
        time.sleep(random.uniform(2, 5))
        page += 1

    session.close()

    return {"title": title, "rating": rating, "reviews": all_reviews}

def save_reviews_to_json(data, book_id):
    filename = f"goodreads_reviews_{book_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved as {filename}")

# if __name__ == "__main__":
#     base_url = "https://www.goodreads.com/book/show/1420.Hamlet/reviews"
#     MAX_PAGE = 30
#     match = re.search(r'/book/show/(\d+)', base_url)
#     book_id = match.group(1) if match else "unknown"
#
#     book_reviews = scrape_goodreads_all_reviews(base_url,max_page=MAX_PAGE)
#     if book_reviews:
#         save_reviews_to_json(book_reviews, book_id)


if __name__ == "__main__":
    #brute-force traversal
    df = pd.read_csv('book_counts.csv')
    low_count_books = df[df['count'] < 10]

    MAX_PAGE = 30
    for index, row in low_count_books.iterrows():
        book_slug = row['book_slug']

        match = re.match(r"(\d+)", book_slug)
        book_id = match.group(1) if match else "unknown"
        out_path = f"goodreads_reviews_{book_id}.json"
        if os.path.exists(out_path):
            print(f"Skipping book {book_id} (file already exists)")
            continue

        base_url = f"https://www.goodreads.com/book/show/{book_slug}/reviews"
        match = re.search(r'/book/show/(\d+)', base_url)
        book_id = match.group(1) if match else "unknown"

        # 爬取书评数据
        book_reviews = scrape_goodreads_all_reviews(base_url, max_page=MAX_PAGE)
        if book_reviews:
            save_reviews_to_json(book_reviews, book_id)
