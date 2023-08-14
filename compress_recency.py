import pyspark.sql.functions as F
from pyspark.sql import SparkSession

def main(spark):
    # Read the interactions dataframe
    interactions_df = spark.read.parquet("/user/ivs225_nyu_edu/may12_exploration/ranked_recency_training_data_small.parquet")

    # Apply log1p function to count column to compress values
    interactions_df = interactions_df.withColumn('rating', F.log1p(1 + interactions_df[‘recency’]))

    # Print the compressed dataframe
    print("Compressed interactions dataframe:")
    interactions_df.show()

    # Write the compressed dataframe to a new Parquet file
    interactions_df.write.mode('overwrite').parquet("/user/ivs225_nyu_edu/may12_exploration/training_compressed_small.parquet")

if _name_ == “_main_”:
    # Configure Spark
    spark = SparkSession.builder.appName("CompressInteractions").getOrCreate()

    # Call the main function
    main(spark)

    # Stop Spark
    spark.stop()
