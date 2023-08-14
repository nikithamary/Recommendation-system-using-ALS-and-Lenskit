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
    
