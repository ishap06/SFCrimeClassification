import pandas as pd
import numpy as np

from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql.functions import col,array, countDistinct, Column
from pyspark.sql.functions import udf
from pyspark.ml.feature import VectorAssembler,VectorIndexer, StringIndexer
from pyspark.mllib.linalg import DenseVector
from itertools import groupby
from functools import reduce
from operator import itemgetter
from operator import add

# USING TEST2 FOLDER

def createDF(val):
    result =[]
    for x in val:
        temp = (x, 0.0)
        result.append(temp)
        #print(list(result))
    return result


def gridToArray(grids, dfMatrix):
    unique2, counts2 = np.unique(dfMatrix[:, 0], return_counts=True)
    sizeOfCount2 = counts2.size
    np.set_printoptions(suppress=True)
    for val in grids:
        dfMatrixNew = dfMatrix[dfMatrix[:, 1] == val]
        unique, counts = np.unique(dfMatrixNew[:, 0], return_counts=True)
        sizeOfCount = counts.size
        zerosNeeded = sizeOfCount2 - sizeOfCount
        counts = np.pad(counts, (0, zerosNeeded), 'constant')
        probailities = counts.astype(float) / counts2
        dfTrainRaw['ProbabVector'] = pd.DataFrame(probailities)




#dfTrainRaw = pd.read_csv("Train2/part-00000-bb0bd7c0-7fc5-4ba4-9855-9631099bac06-c000.csv")
#dfNew = gridToArray(dfTrainRaw['GridNo'], dfTrainRaw['Category_New'].values)

sc = SparkContext()
sqlContext = SQLContext(sc)

sc.setLogLevel("ERROR")

dfTrainRaw = sqlContext.read.load('Train2/part-00000-1ecd69df-e6fa-4039-a594-bd92bf379226-c000.csv',
                                  format='com.databricks.spark.csv',
                                  header='true',
                                  inferSchema='true')
dfTestRaw = sqlContext.read.load('Test2/part-00000-46c2a84c-4328-455f-a572-bae14b38a2f2-c000.csv',
                                  format='com.databricks.spark.csv',
                                  header='true',
                                  inferSchema='true')

dfUsedCat = dfTrainRaw.select('Category_New').rdd.flatMap(lambda x: x).collect()
dfUsedGrid = dfTrainRaw.select('GridNo').rdd.flatMap(lambda x: x).collect()

print("-------STEP 1------")
dfUsed = sc.parallelize(zip(dfUsedGrid, dfUsedCat))

print("-------STEP 2------")
dfUsed_mapping = dfUsed.map(lambda row: (row[1], 1.0)).reduceByKey(add).collect()
dfUsed_mapping.sort(key=itemgetter(0))
# print(dfUsed_mapping)

print("-------STEP 3------")
dfUsed_Parrallel = sc.parallelize(dfUsed_mapping)
generalCounts = dfUsed_Parrallel.values().collect()
#print(generalCounts)

print("-------STEP 4------")
dfUsed_forGrid_mapping = dfUsed
listOfKeysAll = dfUsed_forGrid_mapping.keys().collect()
listOfValuesAll = dfUsed_forGrid_mapping.values().collect()
answer = dict()
newcol = []
newCol2 = []
for val in listOfKeysAll:
    if val in answer:
        newcol.append(answer[val])
        #print(val, " : ", answer[val])
        #print ("--------------------------------")
    else:
        # print("For Grid: ", val)
        dfUsed_forGrid_reduced = dfUsed_forGrid_mapping.filter(lambda x: x[0] == val)
        dfUsed_forGrid_reduced = dfUsed_forGrid_reduced.map(lambda row: (row[1], 1.0)).reduceByKey(
            lambda x, y: x + y)
        listOfValuesGrid = dfUsed_forGrid_reduced.keys().collect()
        dfUsed_forGrid_reduced = dfUsed_forGrid_reduced.collect()
        dfUsed_forGrid_reduced.sort(key=itemgetter(0))
        missingKeyList = list(set(listOfValuesAll) - set(listOfValuesGrid))
        # print("MISSING KEY LIST: ", missingKeyList)
        if (len(missingKeyList) > 0):
            newTuple = createDF(missingKeyList)
            dfUsed_forGrid_reduced = dfUsed_forGrid_reduced + newTuple
            # print("New Dataframe/RDD")
            dfUsed_forGrid_reduced.sort(key=itemgetter(0))
            dfUsed_Parrallel_Grid = sc.parallelize(dfUsed_forGrid_reduced)
            gridCounts = dfUsed_Parrallel_Grid.values().collect()
            answer[val] = [x / y for x, y in zip(gridCounts, generalCounts)]
            newcol.append(answer[val])
        else:
            dfUsed_Parrallel_Grid = sc.parallelize(dfUsed_forGrid_reduced)
            gridCounts = dfUsed_Parrallel_Grid.values().collect()
            answer[val] = [x / y for x, y in zip(gridCounts, generalCounts)]

            newcol.append(answer[val])
        #print(val, " : ", answer[val])
        #print("--------------------------------")




df = sqlContext.createDataFrame(newcol,ArrayType(FloatType())).toDF('ndArray')
df.show()



l = sc.parallelize([1, 2, 3])
l = sc.parallelize(newcol)


index = sc.parallelize(range(0, l.count()))
z = index.zip(l)


rdd = sc.parallelize(dfTrainRaw.collect())
rdd_index = index.zip(rdd)


# just in case!
assert(rdd.count() == l.count())
# perform an inner join on the index we generated above, then map it to look pretty.
new_rdd = rdd_index.join(z).map(lambda x: [x[1]])
new_df = new_rdd.toDF(['new_col'])
#new_df.show(14, False)

df_final = new_df.select('new_col')
#print("~~~~~~~~~~~~~~~~~~~~~~~`")
#df_final.show()

df_final = df_final.select('new_col.*')
df_final = df_final.select('_1.*',col('_2').alias("ndArray").cast(StringType()))
#df_final = df_final.withColumn("nDArray", df_final["ndArray"].cast(IntegerType()))
#df_final = df_final.select('ndArray').drop()

#df_final.coalesce(1).write.option("header", "true").csv('Train', header="true", mode="overwrite")
df_final.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("train.csv")

#============================================================================================================================
dfUsedGrid_Test = dfTestRaw.select('GridNo').rdd.flatMap(lambda x: x).collect()
dfUsed_Test = sc.parallelize(zip(dfUsedGrid_Test))
listOfKeysAll_test = dfUsed_Test.keys().collect()

zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

for val in listOfKeysAll_test:
    if val in answer:
        newCol2.append(answer[val])
    else:
        newCol2.append(zero)

df_test = sqlContext.createDataFrame(newCol2,ArrayType(FloatType())).toDF('ndArray')
l2 = sc.parallelize(newCol2)
index2 = sc.parallelize(range(0, l2.count()))
z2 = index2.zip(l2)
rdd2 = sc.parallelize(dfTestRaw.collect())
rdd_index2 = index2.zip(rdd2)
assert(rdd2.count() == l2.count())
# perform an inner join on the index we generated above, then map it to look pretty.
new_rdd2 = rdd_index2.join(z2).map(lambda x: [x[1]])
new_df2 = new_rdd2.toDF(['new_col'])
df_final2 = new_df2.select('new_col')
df_final2 = df_final2.select('new_col.*')
df_final2 = df_final2.select('_1.*',col('_2').alias("ndArray").cast(StringType()))
df_final2.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("test.csv")








#df_final.toJSON().saveAsTextFile("aaaaaaTest")

#df_final.show(1, False)

#df_final.printSchema()

#df2 = dfTrainRaw.withColumn("Newcol", df['ndArray'])

#df2.show()

'''
print("New Col: ")
print(newcol[0], "\n")
print(newcol[1], "\n")
print(newcol[2], "\n")
print(newcol[3], "\n")
print(newcol[4], "\n")
print(newcol[5], "\n")
print(newcol[6], "\n")
print(newcol[7], "\n")
print(newcol[8], "\n")
print(newcol[9], "\n")

'''