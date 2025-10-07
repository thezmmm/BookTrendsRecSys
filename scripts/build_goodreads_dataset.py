import os
import json
import pandas as pd
import re

def collect_goodreads_data(directory="."):
    data = []

    #goodreads_ratings_{user_id}.json
    pattern = re.compile(r"goodreads_ratings_(\d+).*\.json$", re.IGNORECASE)

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if not match:
            continue

        user_id = match.group(1)
        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)

            for item in content.get("items", []):
                book_id = item.get("book_id")
                rating = item.get("user_rating")

                if book_id is not None and rating is not None:
                    data.append({
                        "userid": int(user_id),
                        "bookid": int(book_id),
                        "rating": int(rating)
                    })

        except Exception as e:
            print(f"{e}")

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = collect_goodreads_data(".")
    output_file = "goodreads_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved as {output_file}")
