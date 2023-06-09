from pyspark import SparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import *
from pyspark.sql.functions import *

def predict(training, test):
    """
    training is a dataframe with user_id, movie_id, and rating columns
    test is a dataframe with user_id, movie_id columns
    """

    # compute pearson similarities between movies, calculate over common users only
    pass

def similarities(training, k, alpha):
    """
    training is a dataframe with user_id, movie_id, and rating columns
    """
    movie_means = training.groupBy("movie_id").agg(mean("rating").alias("movie_mean"))
    temp = training.join(movie_means, "movie_id")
    residuals = temp\
        .withColumn("residual", temp.rating - temp.movie_mean)\
        .drop("movie_mean")
    residuals = residuals.select("user_id", struct(residuals.movie_id, residuals.residual))
    residuals.printSchema()
    residuals = residuals.rdd.groupByKey().mapValues(list)

    def product_reducer(row):
        products = []
        user_id = row[0]
        movie_ids_and_residuals = row[1]
        for i, (movie_id1, residual1) in enumerate(movie_ids_and_residuals):
            j = i
            while j < len(movie_ids_and_residuals):
                movie_id2, residual2 = movie_ids_and_residuals[j]
                products.append((user_id, movie_id1, movie_id2, residual1 * residual2, residual1**2, residual2**2))
                j += 1
        return products
    
    products = residuals.flatMap(product_reducer).toDF()
    products = products.select(
        products._1.alias("user_id"),
        products._2.alias("movie_id1"),
        products._3.alias("movie_id2"),
        products._4.alias("product"),
        products._5.alias("sq_res1"),
        products._6.alias("sq_res2")
    )
    p1 = products.filter(products.movie_id1 <= products.movie_id2)

    p2 = products.filter(products.movie_id1 > products.movie_id2)
    p2.select(
        "user_id",
        col("movie_id2").alias("movie_id1"),
        col("movie_id1").alias("movie_id2"),
        "product",
        "sq_res1",
        "sq_res2"
    )
    products = p1.union(p2)
    sims = products.groupBy("movie_id1", "movie_id2").agg(
        sum("product").alias("prod_sum"),
        sum("sq_res1").alias("sq_res1_sum"),
        sum("sq_res2").alias("sq_res2_sum"),
        count("*").alias("support")
    )
    sims = sims.select(
        "movie_id1",
        "movie_id2",
         ((col("prod_sum") / pow(col("sq_res1_sum") * col("sq_res2_sum"), 0.5)) * (col("support") / (col("support") + alpha))).alias("similarity")
    )
    temp = sims.filter(col("movie_id1") < col("movie_id2")).select(
        col("movie_id2").alias("movie_id1"),
        col("movie_id1").alias("movie_id2"),
        "similarity"
    )
    sims = sims.union(temp)

    windowSpec = Window.partitionBy("movie_id1").orderBy("similarity")
    sims = sims\
        .withColumn("row_num", row_number().over(windowSpec))\
        .filter(col("row_num") <= k)\
        .orderBy("movie_id1", "row_num")

    return sims


if __name__ == "__main__":
    from collaborative_filtering import get_data

    spark = SparkSession.builder \
        .master('local[*]') \
        .config("spark.driver.memory", "3g") \
        .config("spark.executor.memory", "16g") \
        .getOrCreate()
    sc = spark.sparkContext

    training, val, test = get_data("data/all_data.csv")

    k = 500
    alpha = 100

    sims = similarities(training, k, alpha)
    sims.write.csv("./data/similarities_k={}_alpha={}".format(k, alpha))

    