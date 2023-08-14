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
