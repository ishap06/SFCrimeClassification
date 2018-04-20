from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import numpy as np
import math
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('ishap06', '6sID9P0Z030qUtg7vxKX')


def prepareForDT(dfCategorical):
    le = preprocessing.LabelEncoder()
    dfCategorical = dfCategorical.apply(le.fit_transform)
    return dfCategorical

main = pd.read_csv('New_Change_Notice__Police_Department_Incidents.csv')
train = pd.read_csv('./NEW/train/train.csv')
test = pd.read_csv('./NEW/test/test.csv')

print ("@@@@@@@@@@@@  ORIGINAL  @@@@@@@@@@@@")
#print (train.dtypes)
#print train.head(6)
#print test.head(6)
#print train.groupby('X').X.count()
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')


# =========================================================================================================================================================

dfTrain = pd.get_dummies(train, columns=["Season"], prefix=["season"])

dfNumericalTrain = dfTrain.select_dtypes(include=[np.number])
print(dfNumericalTrain.head(4).dtypes)


dfCategoricalTrain = dfTrain.select_dtypes(include=['object']).copy()
dfCategoricalTrain = prepareForDT(dfCategoricalTrain)
dfTrain = pd.concat([dfCategoricalTrain, dfNumericalTrain], axis=1)
print ('@@@@@@@@@@@@  TRAIN  @@@@@@@@@@@@')
print (dfTrain.head(5))
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')

dfTest = pd.get_dummies(test, columns=["Season"], prefix=["season"])
dfCategoricalTest = dfTest.select_dtypes(include=['object']).copy()
dfNumericalTest = dfTest.select_dtypes(include=[np.number])
dfCategoricalTest = prepareForDT(dfCategoricalTest)
dfTest = pd.concat([dfCategoricalTest, dfNumericalTest], axis=1)
print ('@@@@@@@@@@@@  TEST  @@@@@@@@@@@@')
#print dfTest.head(5)
print ('@@@@@@@@@@@@@@@@@@@@@@@@@')

#-------------------------------------------------------------------------------------------
#---------------------APPLYING THE STANDARD SCALAR AND REMOVING OUTLIERS FROM THE DATA------------------------------------
features = ['DayOfWeek','PdDistrict','Category_New','X','Y','BlockOrJunc','Year','TimeOfDay','Month','Day',
            'season_Fall,','season_Spring','season_Summer','season_Winter']

#scaler = preprocessing.StandardScaler()
#dfTrain = scaler.fit_transform(dfTrain)
#dfTrain = pd.DataFrame(dfTrain, columns=features)

#dfTest = scaler.fit_transform(dfTest)
#dfTest = pd.DataFrame(dfTest, columns=features)

#print "AFTER SCALAR:", dfTrain.head(7)
#----------------------------------------------------------------------------------

#-------------------------------NORMALIZING THE TRAIN AND TEST SET BY SUBTRACING MEAN---------------------------------------
#norm  = np.mean(dfTrain,axis=0)
#train_norm = dfTrain - norm
#test_norm = dfTest - norm
#----------------------------------------------------------------------------------
colsToDrop1 = ['Category_New','season_Spring','season_Fall','Year','X', 'IncidntNum']
colsToDrop2 = ['Category_New','Category','Location','PdId','season_Spring','season_Fall','Year','Day', 'IncidntNum','TimeOfDay']
# ---------------------------------------------TRAIN--------------------------

dfTrainLabel = dfTrain['Category_New']
#dfTrainLabel = np_utils.to_categorical(dfTrainLabel)

category = 'Category_New'
dfTrain = dfTrain.drop(colsToDrop2 ,axis=1)
#dfTrain['X'] = dfTrain['X'].abs()
#---------------------------------------------TEST----------------------
dfTestLabel = dfTest['Category_New']
#dfTestLabel = np_utils.to_categorical(dfTestLabel)

category = 'Category_New'
dfTest = dfTest.drop(colsToDrop2 ,axis=1)
#dfTest['X'] = dfTest['X'].abs()
#-------------------------------------------------------------------------

clf_gini = RandomForestClassifier(max_depth=13, n_estimators=700,max_features='sqrt',
                             random_state=10).fit(dfTrain, dfTrainLabel)
pred = clf_gini.predict_proba(dfTest)
print ("RANDON FOREST:", log_loss(dfTestLabel,pred))


#logistic = RandomForestClassifier(max_depth=13, n_estimators=100,max_features='log2',
#                             random_state=10)
#LogisticRegression(penalty='l2',C=0.6, tol=0.001,max_iter=400, solver="lbfgs", random_state=10)
#pca = PCA(n_components=10, svd_solver="full")
#pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
#pipe.fit(dfTrain, dfTrainLabel)
#pred = pipe.predict_proba(dfTest)
#print "PCA + LogReg:", log_loss(dfTestLabel,pred)

'''

=============XXXXXXX======================XXXXXXX================XXXXXX=======================
WORKING ONES:
#----------------------------------------------------------------------------------------------------
clf_gini = RandomForestClassifier(max_depth=13, n_estimators=700,max_features='sqrt',
                             random_state=10).fit(dfTrain, dfTrainLabel)
pred = clf_gini.predict_proba(dfTest)
print "RANDON FOREST:", log_loss(dfTestLabel,pred)

#----------------------------------------------------------------------------------------------------
lrModel = LogisticRegression(multi_class='multinomial',penalty='l1',max_iter=400, solver='saga').fit(dfTrain, dfTrainLabel)
pred = lrModel.predict_proba(dfTest)
print "LogReg:", log_loss(dfTestLabel,pred)
#----------------------------------------------------------------------------------------------------
multiNB = MultinomialNB(alpha=0.01).fit(dfTrain, dfTrainLabel)
pred = multiNB.predict_proba(dfTest)
print "Naive Bayes:", log_loss(dfTestLabel,pred)
#----------------------------------------------------------------------------------------------------

knn = KNeighborsClassifier(n_neighbors = 1000).fit(dfTrain, dfTrainLabel)
pred = knn.predict_proba(dfTest)
print ("KNN:", log_loss(dfTestLabel,pred))
#----------------------------------------------------------------------------------------------------



logistic = LogisticRegression()
pca = PCA(n_components=6, svd_solver="full")
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
pipe.fit(dfTrain, dfTrainLabel)
pred = pipe.predict_proba(dfTest)
print "PCA + LogReg:", log_loss(dfTestLabel,pred)



# Print the feature ranking
print("Feature ranking:")

for f in range(dfTrain.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
------------------------------------------------------------------------------------
LOGISTIC MULTINOMIAL REGRESSION
------------------------------------------------------------------------------------

lrModel = LogisticRegression(multi_class='multinomial',penalty='l1',max_iter=400, solver='saga').fit(dfTrain, dfTrainLabel)
pred = lrModel.predict_proba(dfTest)
print "LogReg:", log_loss(dfTestLabel,pred)
------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
SVC CLASSIFIER:

svcModel = SVC(C=1.0, class_weight=None, coef0=0.0, probability=True,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=300, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

svcModel.fit(dfTrain, dfTrainLabel)
pred = svcModel.predict_proba(dfTest)
print log_loss(dfTestLabel,pred)

------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
RANDOM FOREST CLASSIFIER:

TRAIN:

dfTrainLabel = dfTrain['Category_New']
category = 'Category_New'
colsToDrop = ['Category_New','season_Spring','season_Winter','Year','X']
dfTrain = dfTrain.drop(colsToDrop ,axis=1)

TEST:

dfTestLabel = dfTest['Category_New']
category = 'Category_New'
colsToDrop = ['Category_New','season_Spring','season_Winter','Year','X']
dfTest = dfTest.drop(colsToDrop ,axis=1)


clf_gini = RandomForestClassifier(max_depth=11, n_estimators=100,max_features='log2',
                             random_state=10).fit(dfTrain, dfTrainLabel)
pred = clf_gini.predict_proba(dfTest)
print "RANDON FOREST:", log_loss(dfTestLabel,pred)


------------------------------------------------------------------------------------
------------------------------------------------------------------------------------
DECISION TREE CLASSIFIER

clf_gini = DecisionTreeClassifier(criterion="gini", min_samples_leaf=20,max_depth=2, max_features="sqrt",
                                  min_samples_split=100,random_state=100)
clf_gini.fit(dfTrain, dfTrainLabel)
y_pred = clf_gini.predict_proba(dfTest)

print log_loss(dfTestLabel,y_pred)
-----------------------------------
-----------------------------------
FEATURE IMPORTANCE


importances = clf_gini.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(dfTrain.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
-----------------------------------
-----------------------------------
RANDOM FOREST WITH GRID SEARCH

clfrf = RandomForestClassifier(max_features='log2',random_state=10)
param_grid = {
    "max_depth": [2,4,6],
    "n_estimators": [10,20,30],
    "min_samples_leaf": [40,60,100]
}
CV_rfc = GridSearchCV(estimator=clfrf, param_grid=param_grid)
CV_rfc.fit(dfTrain, dfTrainLabel)
print CV_rfc.best_params_

log loss: 2.4874
--------------------------------------
----------------------------------
print "+++++++++++++++++++++++++++++++++++++++++++"
bdt_real = AdaBoostClassifier(n_estimators=24,random_state=True)
bdt_real.fit(dfTrain, dfTrainLabel)
y_pred2 = bdt_real.predict_proba(dfTest)
print log_loss(dfTestLabel,y_pred2)
print "+++++++++++++++++++++++++++++++++++++++++++"


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=4, min_samples_leaf=5, min_samples_split=2)
clf_gini.fit(dfTrain, dfTrainLabel)
y_pred = clf_gini.predict_proba(dfTest)


#print log_loss(dfTestLabel,y_pred)

bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")
bdt_discrete.fit(dfTrain, dfTrainLabel)
y_pred3 = clf_gini.predict_proba(dfTest)
print log_loss(dfTestLabel,y_pred3)



--------------------------------------
--------------------------------------------------

TO GENERATE COMPARATIVE PLOTS FOR BEFORE STANDARD SCALAR AND AFTER 

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')
sns.kdeplot(dfTrain['DayOfWeek'], ax=ax1)
sns.kdeplot(dfTrain['Hour'], ax=ax1)
sns.kdeplot(dfTrain['Month'], ax=ax1)

scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(dfTrain)
scaled_df = pd.DataFrame(scaled_df, columns=['DayOfWeek','PdDistrict','Category_New','X','Y','BlockOrJunc','Year','Hour','Month','Day',
            'season_Fall,','season_Spring','season_Summer','season_Winter'])

ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['DayOfWeek'], ax=ax2)
sns.kdeplot(scaled_df['Hour'], ax=ax2)
sns.kdeplot(scaled_df['Month'], ax=ax2)

plt.show()
-------------------------------------------------------

PLOTLY API KEY: 6sID9P0Z030qUtg7vxKX

-------------------------------------
KERAS:

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=11, activation='relu'))
    model.add(Dense(30, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=4, shuffle=True, random_state=seed)
results = cross_val_score(estimator, dfTrain, dfTrainLabel, cv=kfold)
print (results.mean())

------------------------------------


'''