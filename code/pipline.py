

import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.4-bin-hadoop2.7"

import findspark
findspark.init()
from pyspark.sql import SparkSession
from time import time
from dill.source import getfile

def init_spark(app_name: str):
  spark = SparkSession.builder.appName(app_name).getOrCreate()
  sc = spark.sparkContext
  return spark, sc
  
spark, sc = init_spark('proj2')
print(sc.version)

from pyspark.sql.types import *
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
df = spark.read.json('/content/Static\ data/data.json',schema=SCHEMA)

"""# Data pre-processing

Functions to be used
"""

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql.types import DoubleType, ArrayType, StringType
import pyspark.sql.functions as f
from datetime import datetime
import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT

# These functions ideally could have been wrote using spark, however, I kept
# running into many errors without a soultion so I had to use udf

norm = f.udf(lambda x : sum([i ** 2 for i in x]), DoubleType())

seconds_in_day = 24 * 60 * 60

# Explained later
def sec_sin_aux(ts):
  time  = datetime.fromtimestamp(ts/1000)
  seconds_since_midnight = (time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
  return (seconds_since_midnight/seconds_in_day) * 2 * np.pi 

def sec_cos_aux(ts):
  time  = datetime.fromtimestamp(ts/1000)
  seconds_since_midnight = (time - time.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
  return (seconds_since_midnight/seconds_in_day) * 2 * np.pi


seconds_sin = f.udf(lambda x: sec_sin_aux(int(x)))
seconds_cos = f.udf(lambda x: sec_cos_aux(int(x)))



from pyspark.sql.functions import when
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
                               "User_number"],
                    outputCol="features")
])

user_transformer = Pipeline(stages=[
    StringIndexer(inputCol='User', outputCol='User_number')
])

def process_data(df):

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

data = process_data(df)

test, train = data.randomSplit([0.7, 0.3])


"""

# Random Forest
"""

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label',
                            maxDepth=10, numTrees=50)
rfModel = rf.fit(train)

# Evaluating our model

predictions = rfModel.transform(train)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Training accuracy = %s" % (accuracy))


predictions = rfModel.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Testing ccuracy = %s" % (accuracy))
