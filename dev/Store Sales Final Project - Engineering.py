# Databricks notebook source
# MAGIC %md
# MAGIC # Final Project: Building Intelligent Lakehouse for Store Sales dataset
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %fs ls /mnt/data/2024-kaggle-final/store-sales-time-series-forecasting/

# COMMAND ----------

# DBTITLE 1,Create Team Database
databaseName = "team_texas"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {databaseName}")
spark.sql(f"use {databaseName}")

tables = spark.sql("SHOW TABLES")

display(tables)

# Use the DESCRIBE DATABASE command to get the database metadata
database_metadata = spark.sql(f"DESCRIBE DATABASE EXTENDED {databaseName}")

# Show the result
database_metadata.show(truncate=False)

# COMMAND ----------

# DBTITLE 1,Create Bronze Tables Using Upsert
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

# Create Train Table  3M rows, 34 unique families using to parittion 

spark.sql("""
  CREATE TABLE IF NOT EXISTS bronze_train (
    id INT, 
    date DATE,
    store_nbr INT,
    family STRING, 
    sales FLOAT,
    onpromotion INT
  )
  USING DELTA
  PARTITIONED BY (family, store_nbr) 
""")

# Load the CSV file into a DataFrame with the specified schema
train_df = spark.read.csv("dbfs:/mnt/data/2024-kaggle-final/store-sales-time-series-forecasting/train.csv", header=True, inferSchema=True)

# Write the DataFrame to a Delta table, partitioned by 'family'
delta_table = DeltaTable.forName(spark, "bronze_train")
delta_table.alias("target").merge(
    train_df.alias("source"),
    "target.id = source.id AND target.date = source.date AND target.store_nbr = source.store_nbr"
).whenMatchedUpdate(set={
    "family": "source.family",
    "onpromotion": "source.onpromotion", 
    "sales": "source.sales"

}).whenNotMatchedInsertAll().execute()


# COMMAND ----------

# DBTITLE 1,Test Table - Bronze
# Create Test Table - 28.5K rows
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

spark.sql("""
  CREATE TABLE IF NOT EXISTS bronze_test (
    id INT, 
    date DATE,
    store_nbr INT,
    family STRING, 
    onpromotion INT
  )
  USING DELTA
  PARTITIONED BY (family, store_nbr)
""")

# Load the CSV file into a DataFrame with the specified schema
test_df = spark.read.csv("dbfs:/mnt/data/2024-kaggle-final/store-sales-time-series-forecasting/test.csv", header=True, inferSchema=True)

# Write the DataFrame to a Delta table, partitioned by 'family'
delta_table = DeltaTable.forName(spark, "bronze_test")
delta_table.alias("target").merge(
    test_df.alias("source"),
    "target.id = source.id AND target.date = source.date AND target.store_nbr = source.store_nbr"
).whenMatchedUpdate(set={
    "family": "source.family",
    "onpromotion": "source.onpromotion"
}).whenNotMatchedInsertAll().execute()

# COMMAND ----------

# DBTITLE 1,Stores Table - Bronze
# Stores table has 54 rows - no partition needed

spark.sql("""
    CREATE TABLE IF NOT EXISTS bronze_stores (
    store_nbr INT, 
    city STRING,
    state STRING,
    type STRING, 
    cluster INT
    )
    USING DELTA
""")

# Load the CSV file into a DataFrame with the specified schema
stores_df = spark.read.csv("dbfs:/mnt/data/2024-kaggle-final/store-sales-time-series-forecasting/stores.csv", header=True, inferSchema=True)

stores_delta_table = DeltaTable.forName(spark, "bronze_stores")
stores_delta_table.alias("target").merge(
    stores_df.alias("source"),
    "target.store_nbr = source.store_nbr"
).whenMatchedUpdate(set={
    "city": "source.city",
    "state": "source.state", 
    "type": "source.type",
    "cluster": "source.cluster"
}).whenNotMatchedInsertAll().execute()

# COMMAND ----------

# DBTITLE 1,Oil Table - Bronze
# Oil table - TODO unsure what to parition by... create year /month fields?

spark.sql("""
CREATE TABLE IF NOT EXISTS bronze_oil (
  date DATE,
  dcoilwtico FLOAT
)
USING DELTA
""")

# Load the CSV file into a DataFrame with the specified schema
oil_df = spark.read.csv("dbfs:/mnt/data/2024-kaggle-final/store-sales-time-series-forecasting/oil.csv", header=True, inferSchema=True)

oil_delta_table = DeltaTable.forName(spark, "bronze_oil")
oil_delta_table.alias("target").merge(
    oil_df.alias("source"),
    "target.date = source.date"
).whenMatchedUpdate(set={
    "dcoilwtico": "source.dcoilwtico",
}).whenNotMatchedInsertAll().execute()

# COMMAND ----------

# DBTITLE 1,Holiday Events - Bronze
# Holidays Events only has 350 rows, no need to partition
spark.sql("""
CREATE TABLE IF NOT EXISTS bronze_holidays_events (
  date DATE,
  type STRING,
  locale STRING, 
  locale_name STRING, 
  description STRING, 
  transferred BOOLEAN
)
USING DELTA
""")

# Load the CSV file into a DataFrame with the specified schema
stores_df = spark.read.csv("dbfs:/mnt/data/2024-kaggle-final/store-sales-time-series-forecasting/holidays_events.csv", header=True, inferSchema=True)

stores_delta_table = DeltaTable.forName(spark, "bronze_holidays_events")
stores_delta_table.alias("target").merge(
    stores_df.alias("source"),
    "target.date = source.date"
).whenMatchedUpdate(set={
    "type": "source.type",
    "locale": "source.locale",
    "locale_name": "source.locale_name",
    "description": "source.description",
    "transferred": "source.transferred"
}).whenNotMatchedInsertAll().execute()

# COMMAND ----------

# DBTITLE 1,Transactions - Bronze
# Holidays Events only has 350 rows, no need to partition
spark.sql("""
CREATE TABLE IF NOT EXISTS bronze_transactions (
  date DATE,
  store_nbr INT,
  transactions FLOAT
)
USING DELTA
PARTITIONED BY (store_nbr)
""")

# Load the CSV file into a DataFrame with the specified schema
transactions_df = spark.read.csv("dbfs:/mnt/data/2024-kaggle-final/store-sales-time-series-forecasting/transactions.csv", header=True, inferSchema=True)

stores_delta_table = DeltaTable.forName(spark, "bronze_transactions")
stores_delta_table.alias("target").merge(
    transactions_df.alias("source"),
    "target.date = source.date AND target.store_nbr = source.store_nbr"
).whenMatchedUpdate(set={
   "transactions": "source.transactions"
}).whenNotMatchedInsertAll().execute()

# COMMAND ----------

# DBTITLE 1,Interpolate Missing Oil Prices
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import last, first, coalesce
from pyspark.sql.window import Window

oil_raw = spark.sql("SELECT * FROM bronze_oil ORDER BY date")

# get distinct dates from train and test to interpolate over 
dates = spark.sql("select DISTINCT date FROM bronze_train UNION ALL SELECT DISTINCT date FROM bronze_test")


# join to oil
oil_full = oil_raw.join(dates, "date", "full").sort("date")

nan_counts =  oil_full.select(sum(col("dcoilwtico").isNull().cast("int")).alias("None_Count_dcoilwtico"))
# Show the result
nan_counts.show()

# COMMAND ----------

# DBTITLE 1,LOCF Missing Values
# Define window specification
window_spec_forward = Window.orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
window_spec_backward = Window.orderBy("date").rowsBetween(Window.currentRow, Window.unboundedFollowing)

# Fill missing values with the last non-null value
clean_oil = oil_full.withColumn("dcoilwtico", last("dcoilwtico", ignorenulls=True).over(window_spec_forward))

# fill backwards (first value is null)
clean_oil = clean_oil.withColumn("dcoilwtico", last("dcoilwtico", ignorenulls=True).over(window_spec_backward))

clean_oil.select(sum(col("dcoilwtico").isNull().cast("int")).alias("None_Count_dcoilwtico")).show()

# write to silver_oil table / overwrite existing 
clean_oil.write.format("delta").mode("overwrite").saveAsTable("silver_oil")

# COMMAND ----------

# DBTITLE 1,Clean Holidays
bronze_holidays = spark.sql("SELECT * FROM bronze_holidays_events ORDER BY date")

display(bronze_holidays)

# COMMAND ----------

# DBTITLE 1,Fix Transfer Holidays
from pyspark.sql.functions import col, monotonically_increasing_id, when

# assuming date ordered bronze_holidays... maybe better way to do this
bronze_holidays = bronze_holidays.orderBy("date")

# Reconcile Transfered Holidays
# get all transfered holidays. set index for join
transfered_1 = bronze_holidays.filter((col("type") == "Holiday") & (col("transferred") == True)).drop("transferred").withColumnRenamed("date", "bad_date").withColumn("index", monotonically_increasing_id())

# get corresponding 'Transfer' holidays
transfered_2 = bronze_holidays.filter(col("type") == "Transfer").drop("transferred").withColumn("index", monotonically_increasing_id())

# rename cols for select later 
columns_to_rename = [col for col in transfered_2.columns if col not in ["date", "index"]]
for col in columns_to_rename:
    transfered_2 = transfered_2.withColumnRenamed(col, "bad_" + col)

# use transfer date as true holiday date
transfered_clean = transfered_1.join(transfered_2, ["index"], "left").select("date", "type", "locale", "locale_name", "description")

# # Filter out transfered holidays and add back 
bronze_holidays2 = bronze_holidays.filter("type != 'Transfer' and !transferred").drop("transferred")

# remove where type is 'Work Day' 
silver_holidays_events = bronze_holidays2.union(transfered_clean).filter("type != 'Work Day'")

# call bridge and additional holidays
silver_holidays_events_1 = silver_holidays_events.withColumn(
    "type",
    when(silver_holidays_events.type.isin("Bridge", "Additional"), "Holiday").otherwise(silver_holidays_events.type)
)


# COMMAND ----------

# DBTITLE 1,Split Holiday Types for Store Number Join
# split on locale - join to stores to get store_nbr
stores = spark.sql("SELECT * FROM bronze_stores")

# locale Local joined by city 
local_holidays = silver_holidays_events_1.filter("locale == 'Local'").join(stores.select("store_nbr", "city"), silver_holidays_events_1.locale_name == stores.city, "left").drop("city")

# regional local joined by state
regional_holidays = silver_holidays_events_1.filter("locale == 'Regional'").join(stores.select("store_nbr", "state"), silver_holidays_events_1.locale_name == stores.state, "left").drop("state")

# national holidays are all stores. cartesian join 
national_holidays = silver_holidays_events_1.filter("locale == 'National'")
national_holidays = national_holidays.limit(national_holidays.count()) #crossjoin buggy
store_nbr = stores.select("store_nbr")
national_holidays1 = national_holidays.crossJoin(store_nbr)

# union of local, regional and national 
silver_holidays_events_clean = local_holidays.union(regional_holidays).union(national_holidays1)

display(silver_holidays_events_clean) # for analysis perhaps use distinct date and store_nbr to discern if holiday

silver_holidays_events_clean.write.format("delta").mode("overwrite").partitionBy("store_nbr").saveAsTable("silver_holidays")

# COMMAND ----------

# DBTITLE 1,Ecuadorian City Coordinates
import pandas as pd

# latitude longitude of Ecuadorian Cities
lats_longs = pd.read_csv("/Workspace/Users/mcaval08@gmail.com/Final Project - Forecasting Sales/lats_longs.csv")

pyspark_lats_longs = spark.createDataFrame(lats_longs)

# Rename columns to lowercase
for col in pyspark_lats_longs.columns:
    pyspark_lats_longs = pyspark_lats_longs.withColumnRenamed(col, col.lower())

pyspark_lats_longs.write.format("delta") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .mode("overwrite") \
    .saveAsTable("ecuador_cities")


# COMMAND ----------

# DBTITLE 1,Silver Train
# join train with silver oil and silver holidays for more feature selection

# TODO add stores to join 
qry = """WITH is_holiday AS (
        SELECT DISTINCT(date), store_nbr, 1 as is_holiday 
        FROM silver_holidays),

        oil_data AS (
            SELECT date, dcoilwtico 
            FROM silver_oil)

        SELECT bt.*, 
                CASE WHEN i.is_holiday IS NOT NULL THEN TRUE ELSE FALSE END as is_holiday,
                o.dcoilwtico, 
                bs.city, 
                bs.state, 
                bs.type, 
                bs.cluster, 
                e.latitude, 
                e.longitude
        FROM bronze_train bt 
        LEFT JOIN is_holiday i
        ON bt.date = i.date 
        AND bt.store_nbr = i.store_nbr
        LEFT JOIN oil_data o
        ON bt.date = o.date
        LEFT JOIN bronze_stores bs
        on bt.store_nbr = bs.store_nbr
        LEFT JOIN ecuador_cities e
        ON e.city = bs.city
        ORDER BY DATE"""

silver_train = spark.sql(qry)      

# COMMAND ----------

# DBTITLE 1,Save to Temp Train Silver
from pyspark.sql.functions import date_format
# add day of the week feature 
silver_train = silver_train.withColumn("day_of_week", date_format("date", "EEEE"))

silver_train.write.format("delta") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .mode("overwrite") \
    .partitionBy("family", "store_nbr") \
    .saveAsTable("temp_silver_train")

# COMMAND ----------

# DBTITLE 1,Save to Stream Silver Train
# join test with silver oil and silver holidays for more feature selection
from pyspark.sql.functions import current_timestamp, date_format

silver_train = spark.readStream.format("delta").table("temp_silver_train")

# add day insetrted load time stamp and verbose name of date 
silver_train = silver_train.withColumn("inserted", current_timestamp()).withColumn("day_of_week", date_format("date", "EEEE"))

# Write the stream to the silver_train table with trigger once and partition by family and store_nbr
silver_train.writeStream \
    .format("delta") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .outputMode("append") \
    .partitionBy("family", "store_nbr") \
    .trigger(once=True) \
    .option("checkpointLocation", "/tmp/silver_train_stream_checkpoint") \
    .table("silver_train_stream")

# COMMAND ----------

# DBTITLE 1,Store Dispersion
# MAGIC %sql 
# MAGIC select * from silver_train_stream where date = '2014-06-30'

# COMMAND ----------

# DBTITLE 1,temp_silver_test
# join test with silver oil and silver holidays for more feature selection

# TODO add stores to join 
qry = """WITH is_holiday AS (
        SELECT DISTINCT(date), store_nbr, 1 as is_holiday 
        FROM silver_holidays),

        oil_data AS (
            SELECT date, dcoilwtico 
            FROM silver_oil)

        SELECT bt.*, 
                CASE WHEN i.is_holiday IS NOT NULL THEN TRUE ELSE FALSE END as is_holiday,
                o.dcoilwtico, 
                bs.city, 
                bs.state, 
                bs.type, 
                bs.cluster, 
                e.latitude, 
                e.longitude
        FROM bronze_test bt 
        LEFT JOIN is_holiday i
        ON bt.date = i.date 
        AND bt.store_nbr = i.store_nbr
        LEFT JOIN oil_data o
        ON bt.date = o.date
        LEFT JOIN bronze_stores bs
        on bt.store_nbr = bs.store_nbr
        LEFT JOIN ecuador_cities e
        ON e.city = bs.city
        ORDER BY DATE"""

silver_test = spark.sql(qry)  

# cannot stream from a temporary view. creating a temp delta table 
silver_test.write.format("delta") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .mode("overwrite") \
    .partitionBy("family", "store_nbr") \
    .saveAsTable("temp_silver_test")

# COMMAND ----------

# DBTITLE 1,silver_test_stream
# join test with silver oil and silver holidays for more feature selection
from pyspark.sql.functions import current_timestamp, date_format

silver_test = spark.readStream.format("delta").table("temp_silver_test")

# add day insetrted load time stamp and verbose name of date 
silver_test = silver_test.withColumn("inserted", current_timestamp()).withColumn("day_of_week", date_format("date", "EEEE"))

# Write the stream to the silver_test table with trigger once and partition by family and store_nbr
silver_test.writeStream \
    .format("delta") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .outputMode("append") \
    .partitionBy("family", "store_nbr") \
    .trigger(once=True) \
    .option("checkpointLocation", "/tmp/silver_test_stream_checkpoint") \
    .table("silver_test_stream")

# COMMAND ----------

# DBTITLE 1,Gold Insights
from pyspark.sql.functions import year, month, sum

# create monthly aggregates from silver_train 

#silver_train =  spark.readStream.format("delta").table("silver_train_stream")
silver_train = spark.sql("select * from temp_silver_train")

# Add year and month columns
silver_train = silver_train.withColumn("year", year("date")).withColumn("month", month("date"))

# Group by 'store_nbr', 'year', 'month', and 'family' and aggregate the sales
aggregated_df = silver_train.groupBy("store_nbr", "year", "month", "family").agg(sum("sales").alias("monthly_sales"))

# aggregated_df.writeStream \
#     .format("delta") \
#     .option("mergeSchema", "true") \
#     .option("overwriteSchema", "true") \
#     .outputMode("append") \
#     .partitionBy("family", "store_nbr") \
#     .trigger(once=True) \
#     .option("checkpointLocation", "/tmp/gold_insights_stream_checkpoint") \
#     .table("gold_insights_stream")


#save to gold_insights delta table
aggregated_df.write.format("delta") \
    .option("mergeSchema", "true") \
    .option("overwriteSchema", "true") \
    .mode("overwrite") \
    .partitionBy("family", "store_nbr") \
    .saveAsTable("gold_insights")



# COMMAND ----------



# COMMAND ----------

###########################################################

# COMMAND ----------

# DBTITLE 1,more features?
# maybe agg by store? can then join to transactions... 

from pyspark.sql.functions import sum 
agg_sales =  silver_train.groupBy("date", "store_nbr").agg(sum("sales").alias("daily_store_sales"))

transactions = spark.sql("select * from bronze_transactions")

full = agg_sales.join(transactions, ["date", "store_nbr"])

display(full)
from pyspark.sql.functions import corr
full.select(corr("daily_store_sales", "transactions")).collect()[0][0] # 84% pearson corr .... 

# COMMAND ----------

# TODO gold table with aggregations of monthly level 
# create month - year column for easier aggregation

# COMMAND ----------

# DBTITLE 1,**modified silver work
# from pyspark.sql.functions import col
# from pyspark.sql.types import StringType
# from pyspark.sql.functions import udf

# silver_train = spark.sql("select * from silver_train")
# group_by_columns = [col for col in silver_train.columns if col not in ['family', 'id', "sales"]]

# pivot_df = silver_train.groupBy(*group_by_columns).pivot("family").sum("sales")

# transactions = spark.sql("select * from bronze_transactions")
# full_silver_train = pivot_df.join(transactions, on=['date', 'store_nbr'], how="left")

# # clean names 
# def clean_column_name(col_name):
#     return col_name.lower().replace(" ", "_").replace(",", "_").replace("/", "_")

# # register
# clean_column_name_udf = udf(clean_column_name, StringType())

# for col_name in full_silver_train.columns:
#     full_silver_train = full_silver_train.withColumnRenamed(col_name, clean_column_name(col_name))

# # Show the result
# full_silver_train.columns

# display(full_silver_train.filter("store_nbr == 1").sort("date"))

# full_silver_train.write.format("delta") \
#     .option("mergeSchema", "true") \
#     .option("overwriteSchema", "true") \
#     .mode("overwrite") \
#     .partitionBy("store_nbr") \
#     .saveAsTable("silver_train_modified")

# COMMAND ----------

# DBTITLE 1,Categorical References
# Capture categorical Variables

# silver_family_df  = spark.sql("""
#                          With families as (select distinct family from bronze_train 
#                                                        union 
#                                            select distinct family from bronze_test) 
                                           
#                         select distinct trim(family) as family from families 

#                       """
#                     )

# silver_locale_df = spark.sql("""select distinct trim(locale) as locale from bronze_holidays_events""")

# silver_locale_name_df  = spark.sql(""" select  distinct trim(locale_name) as locale_name from bronze_holidays_events""")

# COMMAND ----------


# spark.sql( """
#              CREATE TABLE IF NOT EXISTS silver_families (
#                                                     id BIGINT GENERATED ALWAYS AS IDENTITY, 
#                                                     family STRING
#                                                 )
#                                                 USING DELTA
#           """
#           )


# spark.sql( """
#              CREATE TABLE IF NOT EXISTS silver_locale (
#                                                     id BIGINT GENERATED ALWAYS AS IDENTITY, 
#                                                     locale STRING
#                                                 )
#                                                 USING DELTA
#           """
#           )


# spark.sql( """
#              CREATE TABLE IF NOT EXISTS silver_locale_name (
#                                                     id BIGINT GENERATED ALWAYS AS IDENTITY, 
#                                                     locale_name STRING
#                                                 )
#                                                 USING DELTA
#           """
#           )




# COMMAND ----------

# DBTITLE 1,Merge Categories
# from delta.tables import DeltaTable

# try :
#     families_delta_table = DeltaTable.forName(spark, "silver_families")
#     families_delta_table.alias("target").merge(
#         silver_family_df.alias("source"),
#         "target.family= source.family"
#     ).whenNotMatchedInsert(values = {
#         "target.family": "source.family"
#         }).execute()

# except Exception as e:
#     print(f'Error while merging Families table: {e}')

# try:
#     locale_delta_table = DeltaTable.forName(spark, "silver_locale")
#     locale_delta_table.alias("target").merge(
#         silver_locale_df.alias("source"),
#         "target.locale= source.locale"
#     ).whenNotMatchedInsert(values = {
#         "target.locale": "source.locale"
#         }).execute()

# except Exception as e:
#     print(f'Error while merging Locale table: {e}')

# try:
#         locale_name_delta_table = DeltaTable.forName(spark, "silver_locale_name")
#         locale_name_delta_table.alias("target").merge(
#             silver_locale_name_df.alias("source"),
#             "target.locale_name= source.locale_name"
#         ).whenNotMatchedInsert(values = {
#             "target.locale_name": "source.locale_name"
#             }).execute()
        
# except Exception as e:
#     print(f'Error while merging Locale_Name table: {e}')
