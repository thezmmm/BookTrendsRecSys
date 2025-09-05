from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

# 1. initialize SparkSession
spark = SparkSession.builder \
    .appName("BookRecommendation") \
    .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
    .config("spark.python.worker.faulthandler.enabled", "true") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# 2. read data from csv
dataset_path = "./datasets/ratings_test.csv"

try:
    ratings_df = spark.read.csv(
        dataset_path,
        header=True,
        inferSchema=True
    )
    print("Data loading complete")
except Exception as e:
    print("Data loading fail:", e)
    spark.stop()
    exit()

# 3. check data
ratings_df.show(5)
ratings_df.printSchema()

# remove null val
ratings_df = ratings_df.na.drop()

# 4. 将 book_id 映射成整数 ID
item_indexer = StringIndexer(inputCol="book_id", outputCol="itemId")
ratings_df = item_indexer.fit(ratings_df).transform(ratings_df)

# 5. make sure userId and rating are right type
ratings_df = ratings_df.select(
    ratings_df["user_id"].cast("int").alias("userId"),
    ratings_df["itemId"].cast("int"),
    ratings_df["rating"].cast("float").alias("rating")
)

# If multiple records exist for the same (userId, itemId) → Take the average
# ratings_df = ratings_df.groupBy("userId", "itemId").agg(F.avg("rating").alias("rating"))

# Data partition
ratings_df = ratings_df.repartition(8)

# 6. Split the data into training and test sets
train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)

# 7. ALS model training
als = ALS(
    userCol="userId",
    itemCol="itemId",
    ratingCol="rating",
    nonnegative=True,
    implicitPrefs=False,
    coldStartStrategy="drop",
    rank=10,
    maxIter=10,
    regParam=0.1
)

model = als.fit(train_df)


# 8. model prediction
predictions = model.transform(test_df)
predictions.show(5)

# 9. evaluate RMSE
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)
print(f"test set RMSE: {rmse:.4f}")

# stop Spark
spark.stop()
