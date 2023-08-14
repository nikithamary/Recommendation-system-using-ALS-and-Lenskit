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
