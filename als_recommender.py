from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
import os

def init_spark(app_name="BookRecommendation"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true") \
        .config("spark.python.worker.faulthandler.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    return spark


def load_and_preprocess_data(spark, dataset_path):
    # read CSV
    df = spark.read.csv(dataset_path, header=True, inferSchema=True)
    df = df.na.drop()

    # recognize follow order: user, item, rating
    cols = df.columns[:3]
    df = df.select(
        df[cols[0]].cast("int").alias("userId"),
        df[cols[1]].cast("int").alias("itemId"),
        df[cols[2]].cast("float").alias("rating")
    )

    # if itemId is not int, use StringIndexer
    from pyspark.ml.feature import StringIndexer
    if dict(df.dtypes)["itemId"] not in ("int", "bigint"):
        item_indexer = StringIndexer(inputCol="itemId", outputCol="itemIdIndexed")
        df = item_indexer.fit(df).transform(df).drop("itemId")
        df = df.withColumnRenamed("itemIdIndexed", "itemId")

    df = df.repartition(8)
    return df


def train_als_model(train_df, rank=10, maxIter=10, regParam=0.1):
    als = ALS(
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop",
        rank=rank,
        maxIter=maxIter,
        regParam=regParam
    )
    model = als.fit(train_df)
    return model

def save_model(model, path="./als_model"):
    if os.path.exists(path):
        # remove old model
        import shutil
        shutil.rmtree(path)
    model.save(path)

def load_model(path="./als_model"):
    return ALSModel.load(path)

def predict_and_evaluate(model, test_df):
    predictions = model.transform(test_df)
    predictions.show(5)

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Test set RMSE: {rmse:.4f}")
    return predictions, rmse


def update_model_with_new_data(spark, old_dataset_path, new_dataset_path, model_save_path="./als_model"):
    old_df = load_and_preprocess_data(spark, old_dataset_path)
    new_df = load_and_preprocess_data(spark, new_dataset_path)

    combined_df = old_df.union(new_df).dropDuplicates(["userId", "itemId"])

    train_df, test_df = combined_df.randomSplit([0.8, 0.2], seed=42)

    model = train_als_model(train_df)

    save_model(model, model_save_path)

    predict_and_evaluate(model, test_df)

    return model, combined_df

if __name__ == "__main__":
    spark = init_spark()

    data_path = "./datasets/ratings.csv"
    df = load_and_preprocess_data(spark, data_path)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    model = train_als_model(train_df)
    save_model(model, path="./als_model")
    # model = load_model("./als_model")
    predict_and_evaluate(model, test_df)

    spark.stop()
