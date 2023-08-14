# preprocessing

import pandas as pd
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, isnull

def main(spark):
    # Read the tracks, interactions, and users dataframes
    tracks_df = spark.read.parquet("hdfs:///user/bm106_nyu_edu/1004-project-2023/tracks_train_small.parquet", header=True)
    interactions_df = spark.read.parquet("hdfs:///user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet", header=True)
    users_df = spark.read.parquet("hdfs:///user/bm106_nyu_edu/1004-project-2023/users_train_small.parquet", header=True)

    # Assign aliases to the columns
    interactions_df = interactions_df.withColumnRenamed("recording_msid", "interactions_msid")
    tracks_df = tracks_df.withColumnRenamed("recording_msid", "tracks_msid")
    users_df = users_df.withColumnRenamed("user_id", "users_user_id")

    # Join interactions and tracks dataframes on 'recording_msid' column
    joined_df = interactions_df.join(tracks_df, interactions_df['interactions_msid'] == tracks_df['tracks_msid'], 'inner')

    # Join the result with users dataframe on 'user_id' column
    joined_df = joined_df.join(users_df, joined_df['user_id'] == users_df['users_user_id'], 'inner')

    #print(joined_df.columns)
    # Replace null values in 'recording_mbid' column with values from 'recording_msid' column
    joined_df = joined_df.withColumn('recording_mbid', when(isnull(joined_df['recording_mbid']), joined_df['tracks_msid']).otherwise(joined_df['recording_mbid']))
    joined_df =joined_df.drop("tracks_msid", "artist_name","track_name","_index_level_0_","user_name","users_user_id")
    # Display the joined and updated dataframe
    joined_df.show()
    joined_df.write.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/final_interactions_train.parquet")
    
if _name_ == "_main_":
    # Configure Spark
    spark = SparkSession.builder.appName("PartitionData").getOrCreate()

    # Call the main function
    main(spark)

    # Stop Spark
    spark.stop()
    
    
# train - val split

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, row_number
from pyspark.sql.window import Window

def split_data(spark, file_path_interactions, split_ratio):
    # Load the interactions data
    interactions = spark.read.parquet(file_path_interactions)

    # Sort the data by timestamp for each user
    window = Window.partitionBy('user_id').orderBy('timestamp')
    interactions = interactions.withColumn('row_num', row_number().over(window))
    sorted_interactions = interactions.orderBy('user_id', 'timestamp')

    # Calculate the split point for each user
    split_point = sorted_interactions.groupBy('user_id').agg({'row_num': 'max'}) \
        .withColumnRenamed('max(row_num)', 'split_point') \
        .withColumn('split_point', (col('split_point') * split_ratio).cast('integer'))

    # Split the data into training and validation sets for each user
    train_set = sorted_interactions.join(split_point, on='user_id') \
        .where(col('row_num') <= col('split_point')) \
        .drop('row_num', 'split_point') \
        .repartition(10).persist()
    train_set.show()
    #print(train_set.count())

    val_set = sorted_interactions.join(split_point, on='user_id') \
        .where(col('row_num') > col('split_point')) \
        .drop('row_num', 'split_point') \
        .repartition(10).persist()
    val_set.show()
    #print(val_set.count())
    

    # Save the training and validation data to disk as Parquet files
    train_set.write.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/final_training_data.parquet")
    val_set.write.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/final_validation_data.parquet")

if _name_ == "_main_":
    # Configure Spark
    spark = SparkSession.builder.appName("SplitData").getOrCreate()

    # Set the input file path and split ratio
    file_path_interactions = "/user/mk8463_nyu_edu/final-project-group-38/final/final_interactions_train.parquet"
    split_ratio = 0.8

    # Call the split_data function
    split_data(spark, file_path_interactions, split_ratio)

    # Stop Spark
    spark.stop()
    
# counts of each recording corresponding to each user

import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand

def main(spark):
    # Load the interactions data

    train_file="/user/mk8463_nyu_edu/final-project-group-38/final/final_training_data.parquet"
    val_file="/user/mk8463_nyu_edu/final-project-group-38/final/final_validation_data.parquet"
    
    train_set = spark.read.parquet(train_file, header=True).select(col('user_id'), col('recording_mbid'))
    val_set = spark.read.parquet(val_file, header=True).select(col('user_id'), col('recording_mbid'))
    print("SUCCESS: interactions")


    # Aggregate the interaction events into count data in parallel
    train_counts = train_set.groupBy('user_id', 'recording_mbid').count()
    train_counts.show()

    val_counts = val_set.groupBy('user_id', 'recording_mbid').count()
    val_counts.show()

    # Save the training and validation count data to disk as Parquet files
    train_counts.write.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/count_training_data.parquet")
    val_counts.write.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/count_validation_data.parquet")

if _name_ == "_main_":
    # Configure Spark
    spark = SparkSession.builder.appName("PartitionData").getOrCreate()

    # Call the main function
    main(spark)

    # Stop Spark
    spark.stop()
    
# getting the top 100 songs

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, desc, sum as _sum, count as _count

def popularity_baseline(train_counts, beta):
    recording_count = train_counts.groupBy('recording_mbid').agg(_sum('count').alias('recording_count'),
                                                                 _count('user_id').alias('user_count'))
    recording_count = recording_count.withColumn('average_count', (col('recording_count') / (col('user_count') + beta)))
    top_100_songs = recording_count.orderBy(desc('average_count')).limit(100)
    return top_100_songs

def main(spark):
    # Read the training counts data from the given path
    train_counts = spark.read.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/count_training_data.parquet")

    # Create the popularity baseline model
    top_100_songs = popularity_baseline(train_counts, beta=100000)

    top_100_songs.show()
    
    # Save the top 100 songs as a Parquet file
    top_100_songs.write.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/final_top_100_songs.parquet")

if _name_ == "_main_":
    # Configure Spark
    spark = SparkSession.builder.appName("PopularityBaselineModelWithBias").getOrCreate()

    # Call the main function
    main(spark)

    # Stop Spark
    spark.stop()
    
# Evaluation

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
