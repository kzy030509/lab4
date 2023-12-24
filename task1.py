from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col
from pyspark.sql.functions import floor

# 创建SparkSession
spark = SparkSession.builder.appName("LoanAmountDistribution").getOrCreate()

# 读取CSV文件
df = spark.read.csv("/lab4/application_data.csv", header=True, inferSchema=True)

# 计算贷款金额区间
df_with_interval = df.withColumn("credit_interval", floor(df["AMT_CREDIT"] / 10000) * 10000)

# 统计每个区间的记录数
result = df_with_interval.groupBy("credit_interval").count().orderBy("credit_interval")

# 格式化输出结果
formatted_result = result.rdd.map(lambda row: (f"({row['credit_interval']}, {row['credit_interval']+10000})", row['count']))

# 输出结果
formatted_result.foreach(print)
# 任务二：统计贷款金额和收入的差值情况
credit_income_comparison_high = (
    df.withColumn("difference", col("AMT_CREDIT") - col("AMT_INCOME_TOTAL"))
    .select("SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_INCOME_TOTAL", "difference")
    .orderBy(-col("difference"))
    .limit(10)
)
credit_income_comparison_low = (
    df.withColumn("difference", col("AMT_CREDIT") - col("AMT_INCOME_TOTAL"))
    .select("SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_INCOME_TOTAL", "difference")
    .orderBy(col("difference"))
    .limit(10)
)
credit_income_comparison_high.show(credit_income_comparison_high.count(), truncate=False)
credit_income_comparison_low.show(credit_income_comparison_low.count(), truncate=False)