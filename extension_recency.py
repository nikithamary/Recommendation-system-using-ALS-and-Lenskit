import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
from pyspark.sql import Window
from pyspark.sql.functions import row_number, exp

def main(spark):
    # Load the interactions data
    train_file="/user/ivs225_nyu_edu/may12_exploration/final_training_data_small.parquet"
    val_file="/user/ivs225_nyu_edu/may12_exploration/final_validation_data_small.parquet"
    
    train_set = spark.read.parquet(train_file, header=True).select(col('user_id'), col('recording_mbid'), col('timestamp'))
    val_set = spark.read.parquet(val_file, header=True).select(col('user_id'), col('recording_mbid'), col('timestamp'))

    # Create a Window, partitioned by 'user_id', 'recording_mbid', and ordered by 'timestamp' in descending order
    # # # window = Window.partitionBy('user_id', 'recording_mbid').orderBy(col('timestamp').desc())
    window = Window.partitionBy('user_id').orderBy(col('timestamp').desc())

    # each USER has records in DESCENDING ORDER.

    # Add a 'recency' column, which is e^(1 - row number)
    # this will order training set by descending timestamp, assign an index, and add a value for recency.
    train_set = train_set.orderBy(col('timestamp').desc())  # not necessary, done above.
    train_set = train_set.withColumn('index', row_number().over(window))\
                         .withColumn('recency', exp(1 - col('index')))\
                         .drop('index')\
    train_set.show()

# this will order validation set by descending timestamp, assign an index, and add a value for recency.
    val_set = val_set.orderBy(col('timestamp').desc())  # not necessary, done above.
    val_set = val_set.withColumn('index', row_number().over(window))\
                     .withColumn('recency', exp(1 - col('index')))\
                     .drop('index')\
    val_set.show()

    # Save the training and validation count data to disk as Parquet files
    train_set.write.parquet("/user/ivs225_nyu_edu/may12_exploration/ranked_recency_training_data_small.parquet")
    val_set.write.parquet("/user/ivs225_nyu_edu/may12_exploration/ranked_recency_validation_data_small.parquet")

if _name_ == “_main_”:
    # Configure Spark
    spark = SparkSession.builder.appName("PartitionData").getOrCreate()

    # Call the main function
    main(spark)

    # Stop Spark
    spark.stop()
