import os
import json
import re
from collections import Counter

def extract_slug(book_url: str):
    """
    从 https://www.goodreads.com/book/show/4671.The_Great_Gatsby/reviews
    提取出 4671.The_Great_Gatsby
    """
    if not book_url:
        return None
    m = re.search(r"/book/show/([^/?#]+)", book_url)
    if m:
        return m.group(1)
    return None

def collect_book_slugs(prefix="goodreads_ratings", out_file="book_counts.csv"):
    counter = Counter()

    for fname in os.listdir("."):
        if fname.startswith(prefix) and fname.endswith(".json"):
            print(f"处理文件: {fname}")
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items = data.get("items", [])
                    for item in items:
                        slug = extract_slug(item.get("book_url"))
                        if slug:
                            counter[slug] += 1
            except Exception as e:
                print(f"读取 {fname} 出错: {e}")

    # 写出结果
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("book_slug,count\n")
        for slug, count in counter.most_common():
            f.write(f"{slug},{count}\n")

    print(f"统计完成，结果写入 {out_file}，共 {len(counter)} 本书。")

if __name__ == "__main__":
    collect_book_slugs()
