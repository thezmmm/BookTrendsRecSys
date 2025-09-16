from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
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


def train_als_model(train_df):
    als = ALS(
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        nonnegative=True,
        implicitPrefs=False,
        coldStartStrategy="drop",
    )
    paramGrid = (ParamGridBuilder()
                 .addGrid(als.rank, [10, 20, 30])
                 .addGrid(als.regParam, [0.01, 0.05, 0.1])
                 .addGrid(als.maxIter, [5, 10])
                 .build())

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    tvs = TrainValidationSplit(
        estimator=als,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=4
    )

    tvsModel = tvs.fit(train_df)
    bestModel = tvsModel.bestModel
    print("\nBest Model Parameters:")
    print(f"  rank: {bestModel.rank}")
    print(f"  maxIter: {bestModel._java_obj.parent().getMaxIter()}")
    print(f"  regParam: {bestModel._java_obj.parent().getRegParam()}")

    return bestModel

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

    # combined_df = old_df.union(new_df).dropDuplicates(["userId", "itemId"])
    combined_df = old_df.union(new_df).groupBy("userId", "itemId").agg(F.avg("rating").alias("rating"))

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
