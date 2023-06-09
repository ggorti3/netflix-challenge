from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import *
from pyspark.sql.functions import *

def get_data(data_path):
    rating_data = spark.read.csv(data_path)
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
    return training, val, test

def get_means(training):
    mu = training.agg(mean("rating")).take(1)[0][0]

    res = training.select(
        "movie_id",
        "user_id",
        (col("rating") - mu).alias("rating")
    )

    mu_users = res.groupBy("user_id").agg(mean("rating").alias("mu_user"))
    res = res.join(mu_users).select("movie_id", "user_id", (col("rating") - col("mu_user")).alias("rating"))

    mu_items = res.groupBy("movie_id").agg(mean("rating").alias("mu_movie"))
    res = res.join(mu_items).select("movie_id", "user_id", (col("rating") - col("mu_movie")).alias("rating"))
    return mu, mu_users, mu_items, res

def normalize(mu, mu_users, mu_items, data):
    res = data.select(
        "movie_id",
        "user_id",
        (col("rating") - mu).alias("rating")
    )

    res = res.join(mu_users).select(
        "movie_id",
        "user_id",
        (col("rating") - col("mu_user")).alias("rating")
    )

    res = res.join(mu_items).select(
        "movie_id",
        "user_id",
        (col("rating") - col("mu_movie")).alias("rating")
    )

    return res

def inverse_normalize(mu, mu_users, mu_items, pred):
    pass

if __name__ == "__main__":
    spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    sc = spark.sparkContext

    training, val, test = get_data("data/all_data.csv")
    normalize(training)

    # als = ALS(
    #     maxIter=5,
    #     rank=5,
    #     regParam=0.1,
    #     userCol="user_id",
    #     itemCol="movie_id",
    #     ratingCol="rating",
    #     coldStartStrategy="drop"
    # )
    # model = als.fit(training)

    # predictions = model.transform(test)
    # evaluator = RegressionEvaluator(
    #     metricName="rmse",
    #     labelCol="rating",
    #     predictionCol="prediction"
    # )
    # rmse = evaluator.evaluate(predictions)
    # print("Root-mean-square error = " + str(rmse))


