#!/bin/sh
source ../../../env.sh

/usr/local/hadoop/bin/hdfs dfs -rm -r /part_4/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /part_4/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/p3d/train.csv /part_4/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../data/p3d/test.csv /part_4/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./model.py hdfs://$SPARK_MASTER:9000/part_4/input/train.csv hdfs://$SPARK_MASTER:9000/part_4/input/test.csv
