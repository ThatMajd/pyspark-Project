from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import when
import os
import time

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.types import DoubleType, ArrayType, StringType
import pyspark.sql.functions as f
from datetime import datetime
import numpy as np
import math
from pyspark.ml.linalg import Vectors, VectorUDT



spark = SparkSession.builder.appName('demo_app')\
    .config("spark.kryoserializer.buffer.max", "512m")\
    .getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
topic_static = "static"


print ('after spark init')



norm = f.udf(lambda x : sum([i ** 2 for i in x]), DoubleType())

seconds_in_day = 24 * 60 * 60

seconds_sin = f.udf(lambda x: sec_sin_aux(int(x)))
seconds_cos = f.udf(lambda x: sec_cos_aux(int(x)))


position_label_transformer = Pipeline(stages=[
    VectorAssembler(inputCols=["x", "y", "z",], outputCol="position"),
    StringIndexer(inputCol = 'gt', outputCol = 'label'),
])

pday_transformer = Pipeline(stages=[
    VectorAssembler(inputCols=["Morning", "Noon", "Night"],
                    outputCol="pdayVec")
])

feature_transformer = Pipeline(stages=[
    VectorAssembler(inputCols=["sinSec", "cosSec", "positionNorm", "position",
                               "User_n"],
                    outputCol="features")
])

user_transformer = Pipeline(stages=[
    StringIndexer(inputCol='User', outputCol='User_n')
])


print ('after norm')



def sec_sin_aux(ts):
  time  = datetime.fromtimestamp(ts/1000)
  seconds_since_midnight = (time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
  return (seconds_since_midnight/seconds_in_day) * 2 * np.pi

def sec_cos_aux(ts):
  time  = datetime.fromtimestamp(ts/1000)
  seconds_since_midnight = (time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
  return (seconds_since_midnight/seconds_in_day) * 2 * np.pi


def process_data(df):
    print ('inside process data')
    # Time manipulation
    df = df.withColumn("Arrival_Time1", f.from_unixtime(f.col("Arrival_Time")/1000))
    df = df.withColumn("Arrival_Date", f.split("Arrival_Time1", " ").getItem(0))
    df = df.withColumn("Arrival_Hour", f.split("Arrival_Time1", " ").getItem(1))

  
    df = df.withColumn("p_day", when((6 <= f.hour("Arrival_Hour")) & (f.hour("Arrival_Hour") <= 12),"Morning")
                                .when((12 < f.hour("Arrival_Hour")) & (f.hour("Arrival_Hour") <= 19),"Noon")
                                .otherwise("Night"))
    categories = ["Morning", "Noon", "Night"]

    exprs = [f.when(f.col("p_day") == category, 1).otherwise(0).alias(category) \
             for category in categories]

    df = df.select('*', *exprs)

    df = pday_transformer.fit(df).transform(df)
    df = df.drop(*categories)

    df = df.withColumn("sinSec", seconds_sin("Arrival_Time"))
    df = df.withColumn("sinSec", f.sin("sinSec"))

    df = df.withColumn("cosSec", seconds_cos("Arrival_Time"))
    df = df.withColumn("cosSec", f.cos("cosSec"))

    # Position manipulation
    model = position_label_transformer.fit(df)
    df = df.withColumn("positionNorm", norm(f.array("x", "y", "z")))
    df = model.transform(df)
    df = df.drop("x", "y", "z")

    # User manipulation
    df = user_transformer.fit(df).transform(df)

    # Final composition
    df = feature_transformer.fit(df).transform(df)
    df = df.select("features", "label")

    return df





print ('after functions')


SCHEMA = StructType([StructField("Arrival_Time",LongType(),True), 
                     StructField("Creation_Time",LongType(),True),
                     StructField("Device",StringType(),True), 
                     StructField("Index", LongType(), True),
                     StructField("Model", StringType(), True),
                     StructField("User", StringType(), True),
                     StructField("gt", StringType(), True),
                     StructField("x", DoubleType(), True),
                     StructField("y", DoubleType(), True),
                     StructField("z", DoubleType(), True)])




static_df = spark.read\
                  .format("kafka")\
                  .option("kafka.bootstrap.servers", kafka_server)\
                  .option("subscribe", topic_static)\
                  .option("startingOffsets", "earliest")\
                  .option("failOnDataLoss",False)\
                  .option("maxOffsetsPerTrigger", 432)\
                  .load()\
                  .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

topic_stream = "activities"
streaming = spark.readStream\
                  .format("kafka")\
                  .option("kafka.bootstrap.servers", kafka_server)\
                  .option("subscribe", topic_stream)\
                  .option("startingOffsets", "earliest")\
                  .option("failOnDataLoss",False)\
                  .option("maxOffsetsPerTrigger", 500000)\
                  .load()\
                  .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

print ('after read stream')


static_df = process_data(static_df)



train_df = static_df.cache()

acc_sum = 0
acc_count = 0

max_d=20
numT=5

count = 0

def my_func(batch_df, batch_id):
    global train_df
    global rfModel
    global acc_sum
    global acc_count
    global max_d
    global numT
    global count

    print('Current train size: ', train_df.count())

    rf = RandomForestClassifier(featuresCol='features', labelCol='label',
                                maxDepth=max_d, numTrees=numT)
    rfModel = rf.fit(train_df)

    print("This is batch number: ")
    print(batch_id)
    
    processed_df = process_data(batch_df)
    print ('New data in batch recieved ', processed_df.count())
    count += processed_df.count()
    print("New Data that has been predicted on since start : ", count)
    
    predictions = rfModel.transform(processed_df)
    
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)

    print("Accuracy for current batch = %s" % (accuracy))


    acc_sum += accuracy
    acc_count += 1

    mean_acc = acc_sum / acc_count

    print("Current accuracy mean is ", mean_acc)


    # adding streamed data to training data
    train_df = train_df.union(processed_df)
    print ('Current train size after union', train_df.count())
    rfModel = rf.fit(train_df)
    print('Model retrained!')
    if train_df.count() >= 1000000:
      train_size = math.floor(10 * (700000/train_df.count()))/10.0
      delete_size = 1 - train_size
      print("Training data too big, scaling it down!")
      train_df, delete_df = train_df.randomSplit([train_size, delete_size])
      delete_df.unpersist(blocking = True)
    print("------------------------ITER DONE------------------")
    print("")


    
streaming.writeStream.foreachBatch(my_func).start().awaitTermination()

