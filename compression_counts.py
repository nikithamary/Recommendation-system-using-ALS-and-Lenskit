import pyspark.sql.functions as F
from pyspark.sql import SparkSession

def main(spark):
    # Read the interactions dataframe
    interactions_df = spark.read.parquet("/user/sg7374_nyu_edu/final-project-group-38/big_validation_counts.parquet")

    # Apply log1p function to count column to compress values
    interactions_df = interactions_df.withColumn('rating', F.log1p(1 + interactions_df['count']))

    # Print the compressed dataframe
    print("Compressed interactions dataframe:")
    interactions_df.show()

    # Write the compressed dataframe to a new Parquet file
    interactions_df.write.mode('overwrite').parquet("/user/sg7374_nyu_edu/final-project-group-38/big_validation_compressed.parquet")

if __name__ == "__main__":
    # Configure Spark
    spark = SparkSession.builder.appName("CompressInteractions").getOrCreate()

    # Call the main function
    main(spark)

    # Stop Spark
    spark.stop()
