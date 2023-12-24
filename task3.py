from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
# 创建SparkSession
spark = SparkSession.builder.appName("LoanDefaultClassification").getOrCreate()

# 读取CSV文件
df = spark.read.csv("/lab4/application_data.csv", header=True, inferSchema=True)
from pyspark.sql.functions import abs

df = df.withColumn("DAYS_BIRTH", abs(col("DAYS_BIRTH")))
df = df.withColumn("difference", col("AMT_CREDIT") - col("AMT_INCOME_TOTAL")+200000000)
df = df.withColumn("DAYS_LAST_PHONE_CHANGE", abs(col("DAYS_LAST_PHONE_CHANGE")))
df = df.dropna()
# 数据集拆分为训练集和测试集
total_count = df.count()  # 计算数据集总行数
train_count = int(total_count * 0.8)  # 计算训练集行数

train_df = df.limit(train_count)
test_df = df.subtract(train_df)
# 特征向量化
feature_cols = ['difference','DAYS_BIRTH','REGION_RATING_CLIENT','DAYS_LAST_PHONE_CHANGE']  # 替换为您选择的特征属性列

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 选择贝叶斯分类器
classifier = NaiveBayes(labelCol="TARGET", featuresCol="features")
# 构建机器学习管道
pipeline = Pipeline(stages=[assembler, classifier])

# 拟合模型
model = pipeline.fit(train_df)
predictions = model.transform(test_df)

# 将预测结果和真实标签进行比较
predictions = predictions.select("prediction", "TARGET")

# 计算TP、TN、FP、FN
TP = predictions.filter((predictions.prediction == 1) & (predictions.TARGET == 1)).count()
TN = predictions.filter((predictions.prediction == 0) & (predictions.TARGET == 0)).count()
FP = predictions.filter((predictions.prediction == 1) & (predictions.TARGET == 0)).count()
FN = predictions.filter((predictions.prediction == 0) & (predictions.TARGET == 1)).count()

print("True Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
# 计算准确度（accuracy）
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy:", accuracy)

# 计算精确度（precision）
precision = TP / (TP + FP)
print("Precision:", precision)

# 计算召回率（recall）
recall = TP / (TP + FN)
print("Recall:", recall)

# 计算F1分数（F1 score）
f1 = 2 * (precision * recall) / (precision + recall)
print("F1 Score:", f1)
# 停止SparkSession
spark.stop()