#!/bin/sh
source ../../../env.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /part_1/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part_1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/p1d/train.csv /part_1/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/p1d/test.csv /part_1/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./model.py hdfs://$SPARK_MASTER:9000/part_1/input/
