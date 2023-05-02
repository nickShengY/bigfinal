#!/bin/sh
source ../../../env.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /part_2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part_2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/p2d/framingham.csv /part_2/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./model.py hdfs://$SPARK_MASTER:9000/part_2/input/
