from pyspark.sql import SparkSession
from pyspark.sql.functions import expr

# 创建SparkSession
spark = SparkSession.builder.appName("Statistics").getOrCreate()

# 读取CSV文件
df = spark.read.csv("/lab4/application_data.csv", header=True, inferSchema=True)

# 注册DataFrame为临时表
df.createOrReplaceTempView("application_data")

# 任务一：统计男性客户的小孩个数类型比例
child_ratios = spark.sql("""
    SELECT CNT_CHILDREN, COUNT(*) / (SELECT COUNT(*) FROM application_data WHERE CODE_GENDER = 'M') AS ratio
    FROM application_data
    WHERE CODE_GENDER = 'M'
    GROUP BY CNT_CHILDREN
    ORDER BY CNT_CHILDREN
""")

# 输出结果
child_ratios.show(truncate=False)

# 任务二：统计每个客户出生以来每天的平均收入，保存为CSV文件
avg_income_per_day = spark.sql("""
    SELECT SK_ID_CURR, AMT_INCOME_TOTAL / -DAYS_BIRTH AS avg_income
    FROM application_data
    WHERE AMT_INCOME_TOTAL / -DAYS_BIRTH > 1
    ORDER BY avg_income DESC
""")

avg_income_per_day.show(truncate=False)
# 保存结果为CSV文件
avg_income_per_day.write.csv("/lab4/avg_income_per_day.csv", header=True)

# 停止SparkSession
spark.stop()