from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
import math
import pyspark.sql.functions as function
from matplotlib import pyplot as plt
from pyspark.ml import Pipeline



#Standardize time to nearest hour
def timeToHour(value1, value2):
    if value2 > 40:
        return (value1+1)%24
    else:
        return (value1)%24

#Convert X and Y co-ordinates to grid number (1 to 20)
def convertToGrid(longitude,latitude):
    x_coor = math.floor((longitude-long_min)/x_res)
    y_coor = math.floor((latitude - lat_min) / y_res)
    ans = 20 * y_coor + x_coor
    return ans

# Standardize time to nearest hour
def timeOfTheDay(value):
    if value <3 and value>=0:
        return "After Midnight"
    if value >3 and value<=7:
        return "Dawn"
    if value >7 and value>=11:
        return "Morning"
    if value >11 and value>=16:
        return "Afternoon"
    if value >16 and value<=20:
        return "Evening"
    else:
        return "Night"

# Method to convert a date into Season
def dateToSeason(value):
    if 12==value or 1==value or 2==value:
        return 'Winter'
    elif 3==value or 4==value  or 5==value:
        return 'Spring'
    elif 6==value  or 7==value or 8==value:
        return 'Summer'
    else:
        return 'Fall'


# Method to extract Street or Block information from address field
def blockOrJunc(value):
    if '/' in value:
        return int(1)
    else:
        return int(0)


# Converting the Description field to Category
def descriptionToCategory(value1, value2):
    if 'LICENSE' in value1 or 'TRAFFIC' in value1 or 'SPEEDING' in value1 or 'DRIVING' in value1:
        return 'TRAFFIC VIOLATION'
    if 'BURGLARY TOOLS' in value1 or 'AIR GUN' in value1 or 'TEAR GAS' in value1 or 'WEAPON LAWS' in value2\
            or 'WEAPON' in value1:
        return 'DEADLY TOOL POSSESSION'
    if 'SEX' in value1 or 'SEX OFFENSES' in value2:
        return 'SEXUAL OFFENSES'
    if 'FORGERY' in value1 or 'FRAUD' in value1 or 'BAD CHECKS' in value2 or 'COUNTERFEITING' in value2\
            or 'EMBEZZLEMENT' in value2:
        return 'FRAUD/COUNTERFEITING'
    if 'TOBACCO' in value1 or 'DRUG' in value1:
        return 'DRUG/NARCOTIC'
    if 'INDECENT EXPOSURE' in value1 or 'OBSCENE' in value1 or 'DISORDERLY CONDUCT' in value1:
        return 'PORNOGRAPHY/OBSCENE MAT'
    if 'DOMESTIC VIOLENCE' in value1:
        return 'DOMESTIC VIOLENCE'
    if 'SUSPICIOUS OCC' in value2:
        return 'SUSPICIOUS PERSON/ACT'
    if 'TREA' in value2:
        return 'LOITERING'
    if 'WARRANTS' in value2:
        return 'WARRANT ISSUED'
    if 'VANDALISM' in value2:
        return 'ARSON'
    if 'HARASSING' in value1 or 'FAMILY OFFENSES' in value2:
        return 'ASSAULT'
    if 'INFLUENCE OF ALCOHOL' in value1:
        return 'DRUKENNESS'
    else:
        return value2

def addressToStreet(value):
    if '/' in value:
        return value.split("/")[0]
    else:
        return value.split("Block of")[1]

def preprocess(dframe):
    filter_category_list = ['NON-CRIMINAL', 'RECOVERED VEHICLE','SECONDARY CODES']
    dfFinal = dframe.where(col('Category').isin(filter_category_list)==False)
    #dfFinal = dfFinal.select(function.year("Dates").alias('Year'), function.month("Dates").alias('Month'),
    #                        function.hour("Dates").alias('Hour'), function.dayofmonth("Dates").alias('Day')
    #                         , function.minute("Dates").alias('Minute'), 'Dates','DayOfWeek'
    #                         ,'PdDistrict','Address','X','Y','id')

    udfDescriptionToCategory = udf(descriptionToCategory, StringType())
    dfFinal = dfFinal.withColumn("Category_New", udfDescriptionToCategory("Descript", "Category"))


    udfconvertToGrid = udf(convertToGrid, FloatType())
    dfFinal = dfFinal.withColumn("GridNo", udfconvertToGrid("X", "Y"))


    #udfAddressToStreet = udf(addressToStreet, StringType())
    udfBlockOrJunc = udf(blockOrJunc, IntegerType())
    dfFinal = dfFinal.withColumn("BlockOrJunc", udfBlockOrJunc("Address"))   #dfFinal.select(function.year(dfFinal.Dates))


    udfTimeToHour = udf(timeToHour, IntegerType())
    dfFinal = dfFinal.withColumn("Hour", udfTimeToHour("Hour","Minute"))

    udfTimeOfTheDay = udf(timeOfTheDay, StringType())
    dfFinal = dfFinal.withColumn("TimeOfDay", udfTimeOfTheDay("Hour"))

    udfDateToSeason = udf(dateToSeason, StringType())
    dfFinal = dfFinal.withColumn("Season", udfDateToSeason("Month"))

    #dfFinal = dfFinal.withColumnRenamed('Category','Category_New')


    #.withColumn("Street", udfAddressToStreet("Address"))
    drop_list = ['Resolution', 'Descript', 'Address', 'Dates', 'Minute']
    dfFinal = dfFinal.select([column for column in dfFinal.columns if column not in drop_list]).na.drop()
    return dfFinal



sc = SparkContext()
sqlContext = SQLContext(sc)
dfTrainRaw = sqlContext.read.load('New_Change_Notice__Police_Department_Incidents.csv',
                                  format='com.databricks.spark.csv',
                                  header='true',
                                  inferSchema='true')


#dfTestRaw = sqlContext.read.load('Kaggle/test.csv',
#                                  format='com.databricks.spark.csv',
#                                  header='true',
#                                  inferSchema='true')
#origColList = dfTrainRaw.columns
dfTrainRaw = dfTrainRaw.select('IncidntNum', 'Category', 'Descript', 'DayOfWeek', 'Date', 'PdDistrict',
                               'Resolution', 'Address', 'X', 'Y', 'Location', 'PdId' ,
                               function.to_date(dfTrainRaw.Date, 'MM/dd/yyyy').alias('Dates'),
                               function.to_timestamp(dfTrainRaw.Time, 'HH:mm').alias('Time'))

#Defining Latitude and Long max and min
lat_min = 37.6040
long_min = -123.0137
lat_max = 37.8324
long_max = -122.3549

x_res = (long_max-long_min)/20
y_res = (lat_max-lat_min)/20

dfTrainRaw = dfTrainRaw.drop('Date')

dfTrainRaw = dfTrainRaw.select('IncidntNum', 'Category', 'Descript','Dates', 'DayOfWeek', 'PdDistrict',
                               'Resolution', 'Address', 'X', 'Y', 'Location', 'PdId' ,
                  function.year("Dates").alias('Year'), function.month("Dates").alias('Month'),
                  function.hour("Time").alias('Hour'), function.minute("Time").alias('Minute'),
                  function.dayofmonth("Dates").alias('Day'))


dfTrainRaw = dfTrainRaw.filter(dfTrainRaw.X < -122.3549)

dfMain = dfTrainRaw
dfTrain = dfTrainRaw.filter(dfTrainRaw.Year <= 2015)
dfTest = dfTrainRaw.filter(dfTrainRaw.Year > 2015)


#Preprocessing Train and Test
print("=======TRAIN=======")
dfTrain = preprocess(dfTrain)
dfTrain.show(5)

print("=======TEST=======")
dfTest = preprocess(dfTest)
dfTest.show(5)

print("=======MAIN=======")
#dfMain = preprocess(dfMain)
#dfMain.show(5)

#dfMain.coalesce(1).write.option("header", "true").csv('./NEW/main', header="true", mode="overwrite")
dfTrain.coalesce(1).write.option("header", "true").csv('./NEW/train', header="true", mode="overwrite")
dfTest.coalesce(1).write.option("header", "true").csv('./NEW/test', header="true", mode="overwrite")

#dfTest = preprocess(dfTestRaw)

'''


#dfTrain = dfTrain.withColumn("TimeOfDay" , udfTimeOfTheDay("Hour"))
#dfTest = dfTest.withColumn("TimeOfDay" , udfTimeOfTheDay("Hour"))

dfTrain.coalesce(1).write.option("header", "true").csv('./NEW/train1', header="true", mode="overwrite")
#dfTest.coalesce(1).write.option("header", "true").csv('./Kaggle/Test', header="true", mode="overwrite")

dfTrain.show(20)

=========================================================

schema = StructType([
    StructField("IncidntNum", IntegerType()),
    StructField("Category", StringType()),
    StructField("Descript", StringType()),
    StructField("Time", StringType()),
    StructField("DayOfWeek", StringType()),
    StructField("PdDistrict", StringType()),
    StructField("Date", DateType()),
    StructField("Resolution", StringType()),
    StructField("Address", StringType()),
    StructField("X", FloatType()),
    StructField("Y", FloatType()),
    StructField("Location", DecimalType()),
    StructField("PdId", IntegerType())
])

dfTrainRaw = sqlContext.read.csv("_Change_Notice__Police_Department_Incidents.csv",
                                 header=True,
                                 mode="DROPMALFORMED", schema=schema)



'''


sc.stop()
