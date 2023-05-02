from __future__ import print_function

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, when
import sys

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("CensusIncomePrediction")\
        .getOrCreate()

    # Load data
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    
    train_data = spark.read.format("csv").options(header='true', inferschema='true', delimiter=',').load(train_data_path, header=True)
    test_data = spark.read.format("csv").options(header='true', inferschema='true', delimiter=',').load(test_data_path, header=True)
    print("Number of rows in the training dataset:", train_data.count())
    print("Number of rows in the test dataset:", test_data.count())

    # Data preprocessing
    # Combine train and test data for preprocessing
    combined_data = train_data.union(test_data)
    
    # List of categorical and numerical columns
    categorical_columns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    numerical_columns = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    
    # Index and encode categorical columns
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index") for column in categorical_columns]
    encoders = [OneHotEncoder(inputCol=column + "_index", outputCol=column + "_encoded") for column in categorical_columns]

    # Assemble the feature vector
    assembler_input = [column + "_encoded" for column in categorical_columns] + numerical_columns
    assembler = VectorAssembler(inputCols=assembler_input, outputCol="features")
    
    # Scale the feature vector
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

    # Logistic Regression model
    logreg = LogisticRegression(featuresCol="scaled_features", labelCol="income")

    # Pipeline
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler, logreg])

    # Fit the pipeline
    model = pipeline.fit(train_data)

    # Make predictions
    predictions = model.transform(test_data)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="income", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print("Area under ROC curve: ", auc)

    spark.stop()
