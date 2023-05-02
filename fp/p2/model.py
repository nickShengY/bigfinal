from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, when
import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("HeartDiseasePrediction")\
        .getOrCreate()

    # Load data
    dataset = spark.read.format("csv").options(header='true', inferschema='true').load(sys.argv[1], header=True)
    print("Number of rows in the dataset:", dataset.count())

    # Data preprocessing
    dataset = dataset.drop('education')
    dataset = dataset.withColumnRenamed('male', 'Sex_male')
    dataset = dataset.dropna()
    print("Number of rows in the dataset after dropping missing values:", dataset.count())

    # Feature scaling and assembling
    feature_cols = ["age", "Sex_male", "cigsPerDay", "totChol", "sysBP", "glucose"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

    # Train-test split
    train_data, test_data = dataset.randomSplit([0.8, 0.2], seed=5)

    # Logistic Regression model
    logreg = LogisticRegression(featuresCol="scaled_features", labelCol="TenYearCHD")

    # Pipeline
    pipeline = Pipeline(stages=[assembler, scaler, logreg])

    # Fit the pipeline
    model = pipeline.fit(train_data)

    # Make predictions
    predictions = model.transform(test_data)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="TenYearCHD", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("Area under ROC curve: ", auc)

    spark.stop()
