from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, collect_list, mean
from pyspark.sql.types import FloatType, StructType, StructField
import math


def mean_average_precision(recommended_songs, actual_songs):
    ap_sum = 0
    relevant_items = 0

    for k, rec_song in enumerate(recommended_songs, 1):
        if rec_song in actual_songs:
            relevant_items += 1
            ap_sum += relevant_items / k

    return ap_sum / len(actual_songs) if len(actual_songs) > 0 else 0.0


def precision_at_k(recommended_songs, actual_songs, k):
    relevant_items = 0

    for rec_song in recommended_songs[:k]:
        if rec_song in actual_songs:
            relevant_items += 1

    return relevant_items / k if k > 0 else 0.0


def ndcg_at_k(recommended_songs, actual_songs, k):
    dcg = 0
    idcg = 0

    for i, rec_song in enumerate(recommended_songs[:k], 1):
        if rec_song in actual_songs:
            dcg += 1 / math.log2(i + 1)

    for i in range(len(actual_songs)):
        idcg += 1 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(actual_songs, top_100_songs_list, k):
    ap = mean_average_precision(top_100_songs_list, actual_songs)
    pk = precision_at_k(top_100_songs_list, actual_songs, k)
    ndcg = ndcg_at_k(top_100_songs_list, actual_songs, k)
    return (ap, pk, ndcg)

def main(spark, k):
    # Read the training and validation data from the given paths
    val_counts = spark.read.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/final_validation_data.parquet")
    top_100_songs = spark.read.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/final_top_100_songs.parquet")

    # Collect the top 100 songs as a list
    top_100_songs_list = [row['recording_mbid'] for row in top_100_songs.collect()]

    # Create a UDF to calculate the mean average precision, precision at k, and NDCG per user
    compute_metrics_udf = udf(lambda actual_songs: compute_metrics(actual_songs, top_100_songs_list, k), StructType([
        StructField("mean_average_precision", FloatType(), False),
        StructField("precision_at_k", FloatType(), False),
        StructField("ndcg_at_k", FloatType(), False)
    ]))

    # Apply the UDF to calculate the metrics per user in parallel
    val_counts = val_counts.groupBy('user_id').agg(collect_list('recording_mbid').alias('actual_songs'))
    val_counts = val_counts.withColumn('metrics', compute_metrics_udf(col('actual_songs')))

    # Calculate the overall mean average precision, precision at k, and NDCG
    #mean_map = val_counts.agg({"metrics.mean_average_precision": "mean"}).collect()[0]["avg(metrics.mean_average_precision)"]
    #mean_pk = val_counts.agg({"metrics.precision_at_k": "mean"}).collect()[0]["avg(metrics.precision_at_k)"]
    #mean_ndcg = val_counts.agg({"metrics.ndcg_at_k": "mean"}).collect()[0]["avg(metrics.ndcg_at_k)"]

    # Calculate the overall mean average precision, precision at k, and NDCG
    '''
    metrics_summary = val_counts.agg(
        ({"metrics.mean_average_precision": "mean"}).alias("mean_map"),
        ({"metrics.precision_at_k": "mean"}).alias("mean_pk"),
        ({"metrics.ndcg_at_k": "mean"}).alias("mean_ndcg")
    ).collect()[0]
    '''

    metrics_summary = val_counts.agg(
        mean(col("metrics.mean_average_precision")).alias("mean_map"),
        mean(col("metrics.precision_at_k")).alias("mean_pk"),
        mean(col("metrics.ndcg_at_k")).alias("mean_ndcg")
    ).collect()[0]

    mean_map = metrics_summary["mean_map"]
    mean_pk = metrics_summary["mean_pk"]
    mean_ndcg = metrics_summary["mean_ndcg"]

    print("Mean Average Precision: ", mean_map)
    print(f"Precision at {k}: ", mean_pk)
    print(f"Normalized Discounted Cumulative Gain at {k}: ", mean_ndcg)

if _name_ == "_main_":
    # Configure Spark
    spark = SparkSession.builder.appName("EvaluationMetrics").getOrCreate()
    k=10
    print("k=",k)

    # Call the main function
    main(spark, k)

    # Stop Spark
    spark.stop()
