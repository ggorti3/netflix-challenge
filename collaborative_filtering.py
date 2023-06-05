from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import *
from pyspark.sql.functions import *

if __name__ == "__main__":
    spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "15g") \
        .getOrCreate()
    sc = spark.sparkContext

    rating_data = spark.read.csv("data/all_data.csv")
    rating_data = rating_data.select(
        rating_data._c0.cast("int").alias("movie_id"),
        rating_data._c1.cast("int").alias("user_id"),
        rating_data._c2.cast("int").alias("rating"),
        rating_data._c3.cast("date").alias("date"),
    )

    probe = spark.read.csv("data/probe.csv")
    probe = probe.select(
        probe._c0.cast("int").alias("movie_id"),
        probe._c1.cast("int").alias("user_id")
    )

    training = rating_data.join(probe, ["movie_id", "user_id"], "left_anti")
    probe = rating_data.join(probe, ["movie_id", "user_id"], "inner")

    (val, test) = rating_data.randomSplit([0.5, 0.5], seed=1)

    als = ALS(
        maxIter=5,
        rank=5,
        regParam=0.1,
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    model = als.fit(training)

    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))


