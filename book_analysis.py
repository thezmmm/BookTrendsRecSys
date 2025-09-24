import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# read data
df = pd.read_csv("./datasets/books.csv")

# rating analysis
# average rating diagram
plt.figure(figsize=(8,5))
sns.histplot(df['average_rating'], bins=20, kde=True, color='skyblue')
plt.title("Average Rating Distribution")
plt.xlabel("Average Rating")
plt.ylabel("Number of Books")
plt.show()

# average rating percentage
bins = [1.0 + i*0.5 for i in range(9)]  # 1.0-1.5, 1.5-2.0,...,5.0
labels = [f"{round(bins[i],1)}-{round(bins[i+1],1)}" for i in range(len(bins)-1)]
df['rating_bin'] = pd.cut(df['average_rating'], bins=bins, labels=labels, include_lowest=True)
rating_bin_counts = df['rating_bin'].value_counts().sort_index()
rating_bin_percent = rating_bin_counts / rating_bin_counts.sum() * 100

plt.figure(figsize=(10,8))

# plot pie chart
wedges, texts = plt.pie(
    rating_bin_percent.values,
    labels=None,
    autopct=None,
    startangle=140,
    wedgeprops={'edgecolor':'white'}
)

plt.title("Percentage of Books by Average Rating")
legend_labels = [f"{label}: {percent:.1f}%" for label, percent in zip(rating_bin_percent.index, rating_bin_percent.values)]
plt.legend(wedges, legend_labels, title="Rating Range", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# published years analysis
# number of published books per year
plt.figure(figsize=(12,5))
pub_count = df['original_publication_year'].value_counts().sort_index()
sns.lineplot(x=pub_count.index, y=pub_count.values)
plt.title("Number of Books Published per Year")
plt.xlabel("Year")
plt.ylabel("Number of Books")
plt.show()

# average ratings per year
avg_rating_by_year = df.groupby('original_publication_year')['average_rating'].mean()

plt.figure(figsize=(12,5))
sns.lineplot(x=avg_rating_by_year.index, y=avg_rating_by_year.values)
plt.title("Average Rating by Publication Year")
plt.xlabel("Year")
plt.ylabel("Average Rating")
plt.show()

# language distribution analysis
# number of books by language code
lang_count = df['language_code'].value_counts()
plt.figure(figsize=(10,8))
sns.barplot(x=lang_count.values, y=lang_count.index, palette="coolwarm")
plt.title("Number of Books by Language")
plt.xlabel("Number of Books")
plt.ylabel("Language Code")
plt.show()

# number of book by language code (filter en)
df_non_en = df[~df['language_code'].str.contains("en", na=False)]
lang_count = df_non_en['language_code'].value_counts()
plt.figure(figsize=(10,8))
sns.barplot(x=lang_count.values, y=lang_count.index, palette="coolwarm")
plt.title("Number of Books by Non-English Languages")
plt.xlabel("Number of Books")
plt.ylabel("Language Code")
plt.show()

# hot books analysis (most rating counts)
top_books = df.sort_values(by='ratings_count', ascending=False).head(10)
top_books_titles = top_books['title'].values

print("Top 10 books by ratings_count:")
print(top_books[['title', 'ratings_count', 'average_rating']])
# average ratings
plt.figure(figsize=(12,6))
sns.barplot(x='title', y='average_rating', data=top_books, palette="viridis")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Average Rating")
plt.title("Average Rating of Top 10 Most Rated Books")
plt.show()

# ratings from 1 to 5 distribution
ratings_cols = ['ratings_1','ratings_2','ratings_3','ratings_4','ratings_5']
top_books_ratings = top_books[['title'] + ratings_cols].set_index('title')

top_books_ratings_percent = top_books_ratings.div(top_books_ratings.sum(axis=1), axis=0) * 100

top_books_ratings_percent.plot(kind='bar', stacked=True, figsize=(12,6),
                               colormap='viridis')
plt.ylabel("Percentage of Ratings (%)")
plt.title("Ratings 1~5 Distribution of Top 10 Most Rated Books")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Stars")
plt.show()

# find books with few rating counts and high ratings
high_rating_threshold = 4.0
low_rating_count_threshold = df['ratings_count'].quantile(0.20)

cold_books = df[(df['average_rating'] >= high_rating_threshold) &
                (df['ratings_count'] <= low_rating_count_threshold)]

top_cold_books = cold_books.sort_values(by='average_rating', ascending=False).head(10)

ratings_cols = ['ratings_1','ratings_2','ratings_3','ratings_4','ratings_5']
cold_books_ratings = top_cold_books[['title'] + ratings_cols].set_index('title')

cold_books_ratings_percent = cold_books_ratings.div(cold_books_ratings.sum(axis=1), axis=0) * 100

cold_books_ratings_percent.plot(kind='bar', stacked=True, figsize=(12,6),
                                colormap='viridis')
plt.ylabel("Percentage of Ratings (%)")
plt.title("1~5 Star Rating Distribution of Top 10 High-Rating Low-Rating-Count Books")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Stars")
plt.show()