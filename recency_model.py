from pyspark.sql.functions import col, log, when
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("ALSRecommendation") \
        .getOrCreate()

    # Read the indexed dataset from the parquet file
    indexed_data = spark.read.parquet("user/ivs225_nyu_edu/may12_exploration/training_indexed_small.parquet")

    # Read the held-out interactions from the parquet file
    held_out_data = spark.read.parquet("user/ivs225_nyu_edu/may12_exploration/validation_indexed_small.parquet")

    # feed in RECENCY VALUES.
    indexed_data = indexed_data.withColumn("rating", indexed_data[“recency”])
    held_out_data = held_out_data.withColumn("rating", held_out_data[“recency”])

    # Create a binary classification dataframe from the held-out data
    binary_data = held_out_data.withColumn("label", when(held_out_data["rating"] >= 4.0, 1.0).otherwise(0.0))

    # Define the ALS model
    rank = 10
    alpha = 40.0
    reg_param = 0.05
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

    # Create a binary classification dataframe from the predictions
    binary_predictions = predictions.withColumn("label", when(predictions["rating"] >= 4.0, 1.0).otherwise(0.0))

    # Evaluate the model using AUC
    binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label", metricName="areaUnderROC")
    auc = binary_evaluator.evaluate(binary_predictions)
    print(f"Area under ROC curve = {auc}")

if name == "main":
    main()
