# Parallel & Distributed Computing Foundations

Python experiments exploring why large-scale data and ML workloads need parallel systems, ending with a distributed data pipeline in Apache Spark.

## What's inside

| File | Topic |
|---|---|
| `amdahl.py` | **Amdahl's law:** maximum speedup from 1 to 64 workers for different serial fractions |
| `memory_wall.py` | **The memory wall:** how memory use and generation time grow with dataset size, from 10K to 5M rows |
| `collectives.py` | **Collective operations:** simulated broadcast, reduce and all-reduce (naive vs. ring), the patterns behind distributed training |
| `spark_pipeline.py` | **Spark pipeline:** loading the UCI Adult income dataset from HDFS and processing it with PySpark |
| `lab9a_foundations.pdf` | Lab handout |

## Tech

Python · NumPy · Apache Spark (PySpark) · HDFS

## Run it

```bash
python amdahl.py
python memory_wall.py
python collectives.py
spark-submit spark_pipeline.py   # needs Spark, and HDFS with adult.csv
```
