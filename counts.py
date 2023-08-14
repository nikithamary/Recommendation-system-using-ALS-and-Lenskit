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
    
