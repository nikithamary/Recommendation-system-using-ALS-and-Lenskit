from pyspark.sql import SparkSession
from pyspark.sql.functions import dense_rank
from pyspark.sql.window import Window

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("CategoricalToNumerical") \
        .getOrCreate()

    # Read the compressed dataset from the parquet file
    compressed_data = spark.read.parquet("/user/sg7374_nyu_edu/final-project-group-38/big_training_compressed.parquet")

    # Repartition the DataFrame for efficient processing
    num_partitions = 200  # Adjust this value based on your cluster resources
    compressed_data = compressed_data.repartition(num_partitions)

    # Create numerical indices for recording_mbid
    recording_mbid_df = compressed_data.select("recording_mbid").distinct().withColumn("recording_mbid_index", dense_rank().over(Window.orderBy("recording_mbid")) - 1)

    # Join the original DataFrame with the numerical indices DataFrames
    indexed_data = compressed_data.join(recording_mbid_df, on="recording_mbid", how="left")

    # Drop the original categorical columns
    indexed_data = indexed_data.drop("recording_mbid")

    # Save the transformed dataset to a parquet file
    indexed_data.write.parquet("/user/sg7374_nyu_edu/final-project-group-38/big_training_indexed.parquet", mode = 'overwrite')

    # Show the transformed dataset
    indexed_data.show()

if __name__ == "__main__":
    main()
