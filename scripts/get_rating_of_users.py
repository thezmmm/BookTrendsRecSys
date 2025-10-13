import json
import time
from urllib.parse import urljoin, urlparse, parse_qs
import re
import requests
from bs4 import BeautifulSoup
import os
import random
import glob
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============== Cookies & Headers ===============
cookies = {
    "ccsid": "379-0866094-5504217",
    "ubid-main": "133-4921598-2766758",
    "likely_has_account": "true",
    "csm-sid": "633-9155635-2102392",
    "allow_behavioral_targeting": "true",
    "lc-main": "en_US",
    "blocking_sign_in_interstitial": "true",
    "session-id": "143-3116519-0844337",
    "session-id-time": "2389244579l",
    "session-token": "4/2rc/eN7zn3FyZDPlk9pWea4iQQRZiwA0Z8nrvAmWvrVPmAMtDH5WQbPo0LVKMa0KJPXrW+vcPaJ+J8Pe1/4cSI9PK0R5gvROgDBG9tXOBbrrQwspXdLMrUg35XH0SJcwnB0DMPW/cH3kfi+qNCMfmrhuSEzs5fQzPHsUefAn1RcJjcdWwBOFmq+onPzYNPBdDdE8VCdz8GqdRwHcQWkc0899s1oc4O5ERpVR/rT8AJz3S9XIiSRo5d4fhV5fwWUNpLMUJV1vc7FCjyq7ke5b/Jpx1yuTa3upmIKWt+cpwPWt7p0bGQLKzBhVWgPYy/wnUoIDOfH92+xE4cCD7Eys+py0Gw4GithsYEvvPqcgAMj4RfNgHAZA==",
    "x-main": "ZgcOvBWIryMtH3oYMD6LrlWZWCi@ZcogAivSg1Bw@mZrviNUWCJ8TtbtdR650faZ",
    "at-main": "Atza|IwEBINqgfWdQKICLp8KKSbt2IwLUbjEC9bbFkt_tgb18smSJgf673E5xPsxYtbnIU313eITsTw7eJXwVw745O42G_to6JsJaVIwbog9kA7cFS3MB2gj3Vc9AHtWX5TRu8LYGmjgGA5Bl4L56nvcOwBYeEFZJ0g5efvzACAOsZ3h8deCNe7XBj-ydxSkmqdP-KO6kE3FkWW2eyzuKrqaFPgqXSBIFYp4gki--BlM3SDBN2Nj4B4jhKGESq0IxuU64-Ti4GCQ",
    "sess-at-main": "q5+uQocIxfCJ1e8KACcWRsBQPgrWWXeblTCS6Vys8P0=",
    "_session_id2": "ee4d764d32851fd7ac20d2ac07abb546",
    "locale": "en",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Referer": "https://www.goodreads.com/",
    "Upgrade-Insecure-Requests": "1",
}

# =============== Rating Mapping ===============
rating_map = {
    "did not like it": 1,
    "it was ok": 2,
    "liked it": 3,
    "really liked it": 4,
    "it was amazing": 5
}

# =============== Utility Functions ===============
def get_text(el):
    return el.get_text(strip=True) if el else None

def extract_book_id_from_href(href: str):
    if not href:
        return None
    m = re.search(r"/book/show/(\d+)", href)
    if m:
        return int(m.group(1))
    try:
        qs = parse_qs(urlparse(href).query)
        if "book_id" in qs:
            return int(qs["book_id"][0])
    except Exception:
        pass
    return None

def extract_rating_from_static_stars(td_rating):
    if not td_rating:
        return None, None
    stars = td_rating.select_one("span.staticStars[title]")
    if stars and stars.has_attr("title"):
        title = stars["title"].strip().lower()
        return title, rating_map.get(title)
    inner = td_rating.select_one("span.staticStars span[title]")
    if inner and inner.has_attr("title"):
        title = inner["title"].strip().lower()
        return title, rating_map.get(title)
    return None, None

def extract_my_rating_fallback(td_shelves):
    if not td_shelves:
        return None, None
    stars = td_shelves.select_one("div.stars")
    if not stars:
        return None, None
    for attr in ("data-rating", "data-restore-rating"):
        if stars.has_attr(attr):
            val = stars.get(attr)
            try:
                iv = int(val)
                for k, v in rating_map.items():
                    if v == iv:
                        return k, iv
            except Exception:
                pass
    return None, None

# =============== Parse List Page ===============
def parse_list_page(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("tr.bookalike.review") or soup.select("table#books tr") or []
    items = []

    for tr in rows:
        book_title, book_href, book_url, book_id = None, None, None, None

        td_title = tr.select_one("td.field.title, td.title")
        if td_title:
            a_book = td_title.select_one("a.bookTitle, a[href*='/book/show/']")
            if a_book:
                book_title = get_text(a_book)
                book_href = a_book.get("href")
                if book_href:
                    book_url = urljoin(base_url, book_href)
                    book_id = extract_book_id_from_href(book_href)

        if not (book_id and book_title and book_url):
            td_cover = tr.select_one("td.field.cover")
            if td_cover:
                a_cover = td_cover.select_one("a[href*='/book/show/']")
                if a_cover:
                    href2 = a_cover.get("href")
                    if href2:
                        if not book_href:
                            book_href = href2
                        if not book_url:
                            book_url = urljoin(base_url, href2)
                        res_div = td_cover.select_one("[data-resource-id]")
                        if res_div and res_div.has_attr("data-resource-id"):
                            try:
                                book_id = int(res_div["data-resource-id"])
                            except Exception:
                                pass
                        if not book_id:
                            book_id = extract_book_id_from_href(href2)
                if not book_title:
                    img = td_cover.select_one("img[alt]")
                    if img:
                        book_title = (img.get("alt") or "").strip() or None

        td_author = tr.select_one("td.field.author, td.author")
        a_author = td_author.select_one("a") if td_author else None
        author_name = get_text(a_author) or get_text(td_author)

        td_rating = None
        for td in tr.select("td.field"):
            label = td.select_one("label")
            if label and label.get_text(strip=True).lower().endswith("'s rating"):
                td_rating = td
                break
        if td_rating is None:
            td_rating = tr.select_one("td.field.rating")

        rating_text, rating_value = extract_rating_from_static_stars(td_rating)
        if rating_value is None:
            td_shelves = None
            for td in tr.select("td.field"):
                label = td.select_one("label")
                if label and label.get_text(strip=True).lower() == "my rating":
                    td_shelves = td
                    break
            if td_shelves is None:
                td_shelves = tr.select_one("td.field.shelves")
            rt2, rv2 = extract_my_rating_fallback(td_shelves)
            if rv2 is not None:
                rating_text, rating_value = rt2, rv2

        date_read, date_added = None, None
        for td in tr.select("td.field"):
            label = td.select_one("label")
            if not label:
                continue
            lab = label.get_text(strip=True).lower()
            if lab == "date read":
                date_read = get_text(td.select_one("div.value"))
            elif lab == "date added":
                date_added = get_text(td.select_one("div.value"))

        shelves = []
        td_shelves_text = None
        for td in tr.select("td.field"):
            label = td.select_one("label")
            if label and label.get_text(strip=True).lower() == "shelves":
                td_shelves_text = td
                break
        if td_shelves_text:
            for a in td_shelves_text.select("a[href*='/review/list']"):
                tag = get_text(a)
                if tag:
                    shelves.append(tag)

        review_link = tr.select_one("a[href*='/review/show/']")
        review_url = urljoin(base_url, review_link["href"]) if review_link and review_link.has_attr("href") else None

        if book_id and book_title and book_url:
            items.append({
                "book_id": book_id,
                "book_title": book_title,
                "book_url": book_url,
                "author": author_name,
                "user_rating_text": rating_text,
                "user_rating": rating_value,
                "date_read": date_read,
                "date_added": date_added,
                "shelves": shelves,
                "review_url": review_url,
            })

    return items


# =============== Crawl Goodreads Pages ===============
def crawl_goodreads_list(user_id, max_pages, headers, cookies):
    BASE_URL = "https://www.goodreads.com"
    all_items = []

    sess = requests.Session()
    sess.headers.update(headers)
    for k, v in cookies.items():
        sess.cookies.set(k, v)

    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    for page in range(1, max_pages + 1):
        page_url = f"{BASE_URL}/review/list/{user_id}?page={page}&sort=rating&view=reviews"
        try:
            resp = sess.get(page_url, timeout=(10, 60))
            print(f"[{resp.status_code}] GET {page_url}")
        except requests.exceptions.ReadTimeout:
            print(f"[Timeout] Skipping page {page} for user {user_id}")
            continue
        except requests.exceptions.RequestException as e:
            print(f"[Error] {e}")
            continue

        if resp.status_code != 200:
            break

        items = parse_list_page(resp.text, BASE_URL)
        all_items.extend(items)

        if not items:
            break

    sess.close()
    return all_items


def crawl_user_reviews(user_id, max_pages=30):
    out_path = f"./goodreads_ratings_{user_id}.json"
    if os.path.exists(out_path):
        print(f"File already exists: {out_path}")
        return

    data = crawl_goodreads_list(user_id, max_pages, headers, cookies)

    if data:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"source_user_id": user_id, "count": len(data), "items": data},
                f, ensure_ascii=False, indent=2
            )
        print(f"Saved: {out_path} ({len(data)} items)")
    else:
        print(f"No data found for user: {user_id}")

    time.sleep(random.uniform(2, 5))


def process_all_users(book_path, max_pages=30):
    with open(book_path, "r", encoding="utf-8") as f:
        reviews_data = json.load(f)

    reviews = reviews_data.get("reviews", [])
    for review in reviews:
        user_id = review["user_id"]
        crawl_user_reviews(user_id, max_pages)


# =============== Main Entry ===============
if __name__ == "__main__":
    MAX_PAGES = 40
    review_files = glob.glob("./goodreads_reviews_*.json")
    for book_path in review_files:
        print(f"Processing: {book_path}")
        process_all_users(book_path, MAX_PAGES)
