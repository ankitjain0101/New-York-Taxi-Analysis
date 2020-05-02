#pip install pyspark
import matplotlib.pyplot as plt
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import unix_timestamp, from_unixtime, date_format
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor

conf = SparkConf().setMaster("local").setAppName("PySpark_feature_eng")
spark = SparkSession.builder.getOrCreate()
print(spark)

nyt_df = (spark.read.format("csv").options(header="true")
    .load("E:/College/Analytics/Big Data/train.csv"))

#Data Cleaning
nyt_df=nyt_df.withColumn("trip_duration",nyt_df["trip_duration"].cast("bigint"))
nyt_df=nyt_df.withColumn("pickup_longitude",nyt_df["pickup_longitude"].cast("double"))
nyt_df=nyt_df.withColumn("pickup_latitude",nyt_df["pickup_latitude"].cast("double"))
nyt_df=nyt_df.withColumn("dropoff_longitude",nyt_df["dropoff_longitude"].cast("double"))
nyt_df=nyt_df.withColumn("dropoff_latitude",nyt_df["dropoff_latitude"].cast("double"))
nyt_df=nyt_df.withColumn("passenger_count",nyt_df["passenger_count"].cast("bigint"))
nyt_df=nyt_df.withColumn("dropoff_datetime",nyt_df["dropoff_datetime"].cast("timestamp"))
nyt_df=nyt_df.withColumn("pickup_datetime",nyt_df["pickup_datetime"].cast("timestamp"))

nyt_df=nyt_df.drop('id','vendor_id')
nyt_df.printSchema()
nyt_df.cache()

for col in nyt_df.columns:
    print(col, "\t", "with null values: ", nyt_df.filter(nyt_df[col].isNull()).count())

nyt_df.head()
nyt_df = nyt_df.filter((nyt_df['passenger_count'] > 0))
nyt_df.select(['passenger_count','trip_duration']).describe().show()

#Data Manipulation
dropoff_datetime = nyt_df.select(unix_timestamp(nyt_df.dropoff_datetime, 'yyyy/M/dd hh:mm:ss').alias('ut'))\
  .select(from_unixtime('ut').alias('dty'))\
  .select(date_format('dty', 'yyyy').alias('Year_drop').cast('int'),
          date_format('dty', 'M').alias('Month_drop').cast('int'),
          date_format('dty', 'dd').alias('Day_drop').cast('int'),
          date_format('dty', 'HH').alias('hour_drop').cast('int'),
          date_format('dty', 'mm').alias('min_drop').cast('int'),
          date_format('dty', 'ss').alias('sec_drop').cast('int'))\

dropoff_datetime.show()
dropoff_datetime.printSchema()

pickup_datetime = nyt_df.select(unix_timestamp(nyt_df.pickup_datetime, 'yyyy/M/dd hh:mm:ss a').alias('ut'))\
  .select(from_unixtime('ut').alias('dty'))\
  .select(date_format('dty', 'yyyy').alias('Year').cast('int'),
          date_format('dty', 'M').alias('Month').cast('int'),
          date_format('dty', 'dd').alias('Day').cast('int'),
          date_format('dty', 'HH').alias('hour').cast('int'),
          date_format('dty', 'mm').alias('min').cast('int'),
          date_format('dty', 'ss').alias('sec').cast('int'))\

pickup_datetime.show()
pickup_datetime.printSchema()

dropoff_datetime = dropoff_datetime.withColumn("id", monotonically_increasing_id())
pickup_datetime = pickup_datetime.withColumn("id", monotonically_increasing_id())

df10 = dropoff_datetime.join(pickup_datetime, "id", "outer")
df10.sort('id', ascending=True).show(2)

df10.printSchema()

nyt_df = nyt_df.withColumn("id", monotonically_increasing_id())

df_final = nyt_df.join(df10, "id", "outer")
df_final.printSchema()
df_final.show()

 
df_final.createOrReplaceTempView('df_final')

df11 = spark.sql("""select Month, count(trip_duration)Total_PickUp_Trip_Duration
                from df_final group by Month order by Total_PickUp_Trip_Duration DESC""")
df11.show(10)
df12 = spark.sql("""select passenger_count, count(trip_duration)Total_Trip_dur 
                from df_final group by passenger_count order by Total_Trip_dur DESC""")
df12.show(10)

df14 = spark.sql("""select hour, count(trip_duration)Total_Trip_dur 
                from df_final group by hour order by Total_Trip_dur DESC""")
df14.show(10)

##Data Visualization
data_eda = df_final.select(date_format('pickup_datetime', 'E').alias('Day_of_Week'), date_format('pickup_datetime', 'HH').alias('Hour'))
data_eda.show()

data_eda1 = data_eda.toPandas()

eda = data_eda1['Day_of_Week'].value_counts()

eda.plot(kind='bar')
plt.xlabel('Days of a week')
plt.ylabel('Number of Taxi rides')
plt.title('Day wise Taxi Rush')

eda1 = data_eda1['Hour'].value_counts()
eda1.plot(kind='bar')
plt.xlabel('Hours of a day')
plt.ylabel('Number of Taxi rides')
plt.title('Hourly Taxi Rush')

data_eda2 = nyt_df.select('trip_duration')

df_pandas = data_eda2.toPandas()

eda2 = df_pandas['trip_duration'].value_counts()
eda2.plot(kind='hist', bins = 25, title='Trip Duration')

# Features Extraction
vectorAssembler = VectorAssembler(inputCols = ['Year_drop','Month_drop','Day_drop','hour_drop','min_drop','sec_drop','Year','Month','Day','hour','min','sec','passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], outputCol = 'features')
df_ml = vectorAssembler.transform(df_final)
df_ml = df_ml.select(['features', 'trip_duration'])
df_ml.show(3)

splits = df_ml.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

#Linear Regression
lr = LinearRegression(featuresCol = 'features', labelCol='trip_duration', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(train_df)

modelSummary = lrModel.summary
print("RMSE: %f" % modelSummary.rootMeanSquaredError)
print("r2: %f" % modelSummary.r2)

lr_predictions = lrModel.transform(test_df)
lr_predictions.select("prediction","trip_duration","features").show(5)
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="trip_duration",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

#Random Forest
rf = RandomForestRegressor(featuresCol="features", labelCol = 'trip_duration')
rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)
rf_evaluator = RegressionEvaluator(
    labelCol="trip_duration", predictionCol="prediction", metricName="rmse")
rmse = rf_evaluator.evaluate(rf_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="trip_duration",metricName="r2")
print("R Squared (R2) on test data = %g" % rf_evaluator.evaluate(rf_predictions))
