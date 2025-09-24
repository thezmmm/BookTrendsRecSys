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
