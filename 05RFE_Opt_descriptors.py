import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import seaborn as sns
import os

path="./data/"
file='05Cor_descriptor.pkl'
df= pd.read_pickle(path+file)
print(df)

X_all=df.drop(df.columns[0:2], axis=1)
y_all=df["diffusion"]
for i in [X_all,y_all]:
    i.index = range(i.shape[0])
scaler.fit(X_all)
X = scaler.transform(X_all)


train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)  

X_train=train_data.drop(train_data.columns[0:2], axis=1)
y_train=train_data["diffusion"]
X_test=test_data.drop(test_data.columns[0:2], axis=1)
y_test=test_data["diffusion"]

# RF 模型超参数优化
def RF(n_estimator,max_depths,min_samples_split,min_samples_leaf):
    n_estimator = int(n_estimator)
    max_depths = int(max_depths)
    min_samples_split=int(np.round(min_samples_split))
    min_samples_leaf=int(np.round(min_samples_leaf))
    rfg = RandomForestRegressor(n_estimators = n_estimator, max_features='auto',random_state=1, max_depth = max_depths,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)
    rfg.fit(X_train,y_train.values.ravel())
    res = rfg.predict(X_test)
    print("Training set score: %f" % rfg.score(X_train,y_train))
    print("Test set score: %f" % rfg.score(X_test,y_test))
    error1=rfg.score(X_train,y_train)
    error2=rfg.score(X_test,y_test)
    error=error2
    return error

from bayes_opt import BayesianOptimization
pbounds = {"n_estimator": (10, 2000),"max_depths":(10,25),"min_samples_split":(2,10),"min_samples_leaf":(1,10)}
optimizer = BayesianOptimization(f=RF,pbounds=pbounds,random_state=2)
bo=optimizer.maximize(init_points=10,n_iter=500)

RF_par=optimizer.max.get('params')
print("====RF hyperparameter optimization results====")
print(RF_par)

n_estimator = int(RF_par.get('n_estimator'))
max_depths = int(RF_par.get('max_depths'))
min_samples_split=int(np.round(RF_par.get('min_samples_split')))
min_samples_leaf=int(np.round(RF_par.get('min_samples_leaf')))

rf = RandomForestRegressor(n_estimators = n_estimator, max_features='auto',random_state=1, max_depth = max_depths,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf)

def get_models():
    models = dict()
    for i in range(1, 43):
        rfe = RFE(estimator=rf, n_features_to_select=i)
        model =rf
        models[str(i)] = Pipeline(steps=[('s',rfe),('m',model)])
    return models

# 交叉验证
def evaluate_model(model, X, y):
    cv =RepeatedKFold(n_splits=10, n_repeats=1,random_state=1)
    scores = cross_val_score(model,X,y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
    scores_mse=cross_val_score(model,X,y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores,scores_mse

models = get_models()
results,MSE,names = list(), list(), list()


rfe = RFE(estimator=rf, n_features_to_select=12)
model = rf
pipeline = Pipeline(steps=[('s',rfe),('m',model)])

cv =RepeatedKFold(n_splits=10, n_repeats=1,random_state=1)
scores = cross_val_score(model, X, y_all, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
scores_mse=cross_val_score(model,  X, y_all,  scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
print('MSE: %.3f (%.3f)' % (mean(scores_mse), std(scores_mse)))
print('R2: %.3f (%.3f)' % (mean(scores), std(scores)))

selector =rfe.fit(X, y_all)

sele=selector.ranking_
feature_name=X_all.columns

Opt_desc=[]
for i in range (0,len(sele)):
    if sele[i]==1:
        Opt_desc.append(feature_name[i])
df1=df[Opt_desc]

opt_feature_name=df1.columns
print(opt_feature_name)

to_save_reset=df[["ID","diffusion"]].join(df1)
to_save = to_save_reset.reset_index(drop=True)

file2 = r"Opt_RFE_descriptor.csv"
to_save.to_csv(path+file2, index=False)
