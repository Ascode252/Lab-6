from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("Lab9A-Foundations") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark version:", spark.version)

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]
df = spark.read.csv(
    "hdfs://localhost:9000/user/student/lab9/adult.csv",
    header=False,
    inferSchema=True
).toDF(*columns)


print("\n=== Schema ===")
df.printSchema()

print("\n=== First 5 rows ===")
df.show(5, truncate=False)

total_count = df.count()
print("Total record count:", total_count)

df = df.withColumn(
    "label",
    F.when(F.trim(F.col("income")) == ">50K", 1).otherwise(0)
)

print("\n=== Class distribution ===")
class_dist = df.groupBy("label").count().orderBy("label")
class_dist = class_dist.withColumn(
    "percentage",
    F.round((F.col("count") / total_count) * 100, 2)
)
class_dist.show()

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import time

categorical_cols = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race",
    "sex", "native_country"
]

numeric_cols = [
    "age", "fnlwgt", "education_num",
    "capital_gain", "capital_loss", "hours_per_week"
]

indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe")
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=[c + "_ohe" for c in categorical_cols] + numeric_cols,
    outputCol="raw_features"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=False,
    withStd=True
)

# Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.01
)

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, lr])

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

t0 = time.time()
model = pipeline.fit(train_df)
train_time = time.time() - t0
print(f"\nLR training time: {train_time:.3f}s")

predictions = model.transform(test_df)

evaluator_roc = BinaryClassificationEvaluator(metricName="areaUnderROC")
evaluator_pr = BinaryClassificationEvaluator(metricName="areaUnderPR")

print("LR AUC-ROC:", evaluator_roc.evaluate(predictions))
print("LR AUC-PR :", evaluator_pr.evaluate(predictions))

# Random Forest
rf = RandomForestClassifier(numTrees=100, seed=42)

rf_pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, rf])

rf_model = rf_pipeline.fit(train_df)
rf_predictions = rf_model.transform(test_df)

print("\nRF AUC-ROC:", evaluator_roc.evaluate(rf_predictions))
print("RF AUC-PR :", evaluator_pr.evaluate(rf_predictions))
print("\n=== Pipeline stages ===")
for i, stage in enumerate(pipeline.getStages()):
    print(f"Stage {i+1}: {stage.__class__.__name__}")

print("Total number of stages:", len(pipeline.getStages()))

print("\n=== Execution Plan ===")
train_df.explain(True)
print("\n=== Partition Scaling ===")

print("Partitions before:", train_df.rdd.getNumPartitions())

partition_counts = [1, 2, 4, 8, 16]

for n in partition_counts:
    df_rep = train_df.repartition(n)
    print(f"Partitions after repartition({n}):", df_rep.rdd.getNumPartitions())

    t0 = time.time()
    _ = pipeline.fit(df_rep)
    elapsed = time.time() - t0

    print(f"Partitions: {n} | Training time: {elapsed:.3f}s")


from pyspark.sql.functions import broadcast

lookup = spark.createDataFrame(
    [(i, f"category_{i}") for i in range(100)],
    ["id", "category"]
)

large = train_df.select(
    (F.col("age") % 100).alias("id"),
    F.col("label")
)

t0 = time.time()
joined_no_bc = large.join(lookup, on="id")
_ = joined_no_bc.count()
t_no_bc = time.time() - t0

t0 = time.time()
joined_bc = large.join(broadcast(lookup), on="id")
_ = joined_bc.count()
t_bc = time.time() - t0

print(f"\nWithout broadcast: {t_no_bc:.3f}s")
print(f"With broadcast   : {t_bc:.3f}s")
print(f"Speedup          : {t_no_bc / t_bc:.2f}x")

print("\n=== Explain without broadcast ===")
joined_no_bc.explain()

print("\n=== Explain with broadcast ===")
joined_bc.explain()



spark.stop()
