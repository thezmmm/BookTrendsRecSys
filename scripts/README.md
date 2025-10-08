/scripts
├── get_rating_of_users.py           # Fetch rating lists for users from review files
├── get_rating_of_books.py           # Fetch ratings for books based on a list of book_ids
├── build_goodreads_dataset.py       # Merge all rating files into a unified dataset
├── count_books.py                   # Count the number of  books reviewed by users
├── goodreads_reviews__{book_id}.json  # Original review files containing user_id
├── goodreads_ratings_{user_id}.json  # Rating results for individual users
└── goodreads_dataset.csv            # Merged training dataset (userid, bookid, rating)

