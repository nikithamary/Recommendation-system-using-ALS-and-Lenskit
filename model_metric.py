from pyspark.sql import SparkSession
import time
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType


def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("ALSRecommendation") \
        .getOrCreate()

    # Read the indexed dataset from the parquet file
    indexed_data = spark.read.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/big_training_indexed.parquet")
    
    

    # Randomly select 10% of the rows
    #indexed_data = indexed_data.sample(withReplacement=False, fraction=0.9, seed=42)  # Adjust the seed as needed
    
    
    
    # Read the held-out interactions from the parquet file
    held_out_data = spark.read.parquet("/user/mk8463_nyu_edu/final-project-group-38/final/big_test_indexed.parquet")

    start_time = time.time()

    # Define the ALS model
    rank = 20
    alpha = 5.0
    reg_param = 0.5
    als = ALS(
        maxIter=10,
        rank=rank,
        alpha=alpha,
        regParam=reg_param,
        userCol="user_id",
        itemCol="recording_mbid_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
    )
    print(f"ALS model parameters:\n- Rank (dimension) of the latent factors: {rank}\n- Implicit feedback parameter (alpha): {alpha}\n- Regularization parameter: {reg_param}")

    # Train the ALS model
    model = als.fit(indexed_data)

    # Make predictions on the held-out data
    predictions = model.transform(held_out_data)

    # Cast the prediction column to type double
    predictions = predictions.withColumn("prediction", col("prediction").cast("double"))

    end_time = time.time()
    print(f"Time taken to implement the algorithm: {end_time - start_time} seconds")
    
    
   # Select the top 100 recommendations for each user based on prediction scores
    window = Window.partitionBy("user_id").orderBy(desc("prediction"))
    top_predictions = predictions.withColumn("rank", row_number().over(window)).where(col("rank") <= 100)
    
    # Sort the held-out data by rating and select the top 100 items for each user
    window = Window.partitionBy("user_id").orderBy(desc("rating"))
    top_held_out_data = held_out_data.withColumn("rank", row_number().over(window)).where(col("rank") <= 100)
    
    
    top_predictions = top_predictions.drop("count","rating","prediction")
    
    top_held_out_data = top_held_out_data.drop("count","rating")
     
    joined = top_held_out_data.alias("a").join(top_predictions.alias("b"), ['user_id','rank'], 'inner') \
    .select("a.user_id", 
            col("a.recording_mbid_index").alias("true_recording_mbid_index"),  
            col("b.recording_mbid_index").alias("predicted_recording_mbid_index"), 
            )
   
    joined = joined.withColumn("true_recording_mbid_index", array("true_recording_mbid_index"))
    joined = joined.withColumn("predicted_recording_mbid_index", array("predicted_recording_mbid_index"))

    # Define UDF to calculate percentage of common elements in two arrays
    def calculate_common_percentage(true_arr, pred_arr):
        common_count = len(list(set(true_arr) & set(pred_arr)))
        return common_count / len(true_arr) * 100 if len(true_arr) > 0 else 0.0

    common_percentage_udf = udf(calculate_common_percentage, FloatType())

    # Calculate percentage of common elements between true and predicted recording_mbid_index
    common_percentage_df = joined.groupBy("user_id") \
    .agg(size(array_intersect(collect_list("true_recording_mbid_index"), collect_list("predicted_recording_mbid_index"))).alias("common_count"), \
         size(collect_list("true_recording_mbid_index")).alias("true_count")) \
    .withColumn("common_percentage", col("common_count") / col("true_count")) \
    .select("user_id", "common_percentage")

    # Compute average percentage across all users
    avg_common_percentage = common_percentage_df.agg(avg("common_percentage")).collect()[0][0]
    print("MAP Score: {:.6f}".format(avg_common_percentage))
   
      


if _name_ == "_main_":
    main()
