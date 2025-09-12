## Data source

- Data about book https://www.goodreads.com/
  - You can get personal data like ratings used for train a book recommendation system
  - Personal data divided by region (change the parameter in the url)
    - https://www.goodreads.com/user/best_reviewers?country=FI&duration=a
    - https://www.goodreads.com/user/best_reviewers?country=US&duration=a
  - Book list with meaningful filters
    - https://www.goodreads.com/list/
  - collect data by **spider** (maybe)

## Target

- Book recommendation system
  - User-Based Collaborative Filtering Algorithm
  - no good visiulization
- Show Trend about book with visualization (by region, by time, by genre, by language ....)
- the average ratings from one region
- .....

## Train set and Test set

This system only recommend book for users who have rated some books (maybe min 20?)

That's mean all users will be put in train set, which leads to a high score in test cause the test data used to train.

To fix it, we divide train set and test set by **Excluding 1 or 2 rating data from training each user.**

Example: 

```python
dataset
userid  bookid   rating
1		1		2
1 		2		3
1		3		4

trainset
userid  bookid   rating
1		1		2
1 		2		3

testset
userid  bookid   rating
1		3		4
```

## Data Collect

### User-Item Rating matrix

1. select 200 (testing scale) books with the most ratings
2. Get the user info from reviews
3. Collect all ratings from these users
4. format: userId, itemId, Rating
