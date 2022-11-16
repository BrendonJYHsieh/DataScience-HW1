import pandas as pd
import numpy as np
# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
# for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.impute import KNNImputer
from sklearn.utils import shuffle
## for explainer
from lime import lime_tabular
import csv
import random
from tqdm import tqdm
import sys

            
def trainPreprocess(dtf):

    dtf = dtf.rename(columns={"Attribute17": "Y"})
    dtf = dtf.replace({'Y': {'Yes': 1, 'No': 0}})

    # imputer = KNNImputer(n_neighbors=3, weights="distance")
    # for i in tqdm(dtf.columns[dtf.isnull().any()].tolist()):
    #     #dtf[i] = dtf[i].fillna(dtf[i].mean())
    #     dtf[i] = imputer.fit_transform(dtf[[i]])
    
    dtf.to_csv('trainSet.csv')

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(dtf.drop("Y", axis=1))
    dtf_scaled = pd.DataFrame(X, columns=dtf.drop("Y", axis=1).columns, index=dtf.index)
    dtf_scaled["Y"] = dtf["Y"]
    dtf = dtf_scaled
    
    return dtf

def testPreprocess(dtf):
    
    dtf = dtf.drop("Attribute1", axis=1)
    dtf = dtf.drop("Attribute8", axis=1)
    dtf = dtf.drop("Attribute10", axis=1)
    dtf = dtf.drop("Attribute14", axis=1)
    dtf = dtf.drop("Attribute16", axis=1)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(dtf)
    dtf = pd.DataFrame(X, columns=dtf.columns, index=dtf.index)
    
    return dtf
    
_dtf = pd.read_csv('train.csv') 


Y = _dtf.loc[_dtf['Attribute17'] == "Yes"]
Y = Y.drop("Attribute1", axis=1)
Y = Y.drop("Attribute8", axis=1)
Y = Y.drop("Attribute10", axis=1)
Y = Y.drop("Attribute14", axis=1)
Y = Y.drop("Attribute16", axis=1)
Y = Y.dropna(how='any')
Y = Y.reset_index(drop=True)

N = _dtf.loc[_dtf['Attribute17'] == "No"]
N = N.drop("Attribute1", axis=1)
N = N.drop("Attribute8", axis=1)
N = N.drop("Attribute10", axis=1)
N = N.drop("Attribute14", axis=1)
N = N.drop("Attribute16", axis=1)
N = N.dropna(how='any')
N = N.reset_index(drop=True)

print(len(Y.index),len(N.index))

remove_N = N.sample(n=len(N.index)-len(Y.index),random_state=random.randint(0,999999),axis=0)
N = N.drop(remove_N.index, axis=0)
N = N.reset_index(drop=True)

percent = 0.3

Y_more, Y_less = model_selection.train_test_split(Y, test_size=percent)
N_more, N_less = model_selection.train_test_split(N, test_size=percent)

trainSet = shuffle(trainPreprocess(pd.concat([Y_more,N_more])));
validSet = shuffle(trainPreprocess(pd.concat([Y_less,N_less])));

trainSet.to_csv('./data/inputProcessed/Attribute17_Y_train.csv',index=False)
validSet.to_csv('./data/inputProcessed/Attribute17_N_train.csv',index=False)


dtf_train = trainSet
dtf_valid = validSet
test_data = pd.read_csv('test.csv')

test_data = testPreprocess(test_data)


X = dtf_train.drop("Y", axis=1).values
y = dtf_train["Y"].values
feature_names = dtf_train.drop("Y", axis=1).columns.tolist()
## Importance
model = ensemble.RandomForestClassifier(n_estimators=1000,criterion="entropy", random_state=0)
model.fit(X,y)
importances = model.feature_importances_
## Put in a pandas dtf
dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":feature_names}).sort_values("IMPORTANCE", ascending=False)
dtf_importances['cumsum'] =  dtf_importances['IMPORTANCE'].cumsum(axis=0)
dtf_importances = dtf_importances.set_index("VARIABLE")
    
## Plot
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig.suptitle("Features Importance", fontsize=20)
ax[0].title.set_text('variables') 
dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot( kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")
ax[1].title.set_text('cumulative')
dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()


#for i in range(1,len(dtf_importances.index)):
X_names = dtf_importances.index
X_train = dtf_train[X_names].values
y_train = dtf_train["Y"].values
X_test = dtf_valid[X_names].values
y_test = dtf_valid["Y"].values

models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression
models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()

from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
    
    # Fit the classifier
    models[key].fit(X_train, y_train)
    # Make predictions
    predictions = models[key].predict(X_test)
    
    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

print(df_model)

ax = df_model.plot.barh()
ax.legend(
    ncol=len(models.keys()), 
    bbox_to_anchor=(0, 1), 
    loc='lower left', 
    prop={'size': 14}
)
plt.tight_layout()
# plt.show()



# # call model
# model = ensemble.GradientBoostingClassifier()
# ## define hyperparameters combinations to try
# param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],      #weighting factor for the corrections by new trees when added to the model
# 'n_estimators':[100,250,500,750,1000,1250,1500,1750],  #number of trees added to the model
# 'max_depth':[2,3,4,5,6,7],    #maximum depth of the tree
# 'min_samples_split':[2,4,6,8,10,20,40,60,100],    #sets the minimum number of samples to split
# 'min_samples_leaf':[1,3,5,7,9],     #the minimum number of samples to form a leaf
# 'max_features':[2,3,4,5,6,7],     #square root of features is usually a good starting point
# 'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}       #the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.
# random_search = model_selection.RandomizedSearchCV(model,param_distributions=param_dic, n_iter=1000,scoring="f1", n_jobs = -1,verbose=1).fit(X_train, y_train)
# print("Best Model parameters:", random_search.best_params_)
# print("Best Model mean accuracy:", random_search.best_score_)

# model = random_search.best_estimator_
# model.fit(X_train, y_train)
# ## test




# predicted_prob = model.predict_proba(dtf_valid[X_names].values)[:,1]
# predicted = model.predict(dtf_valid[X_names].values)

# ## Accuray e AUC
# accuracy = metrics.accuracy_score(y_test, predicted)
# auc = metrics.roc_auc_score(y_test, predicted_prob)
# print("Accuracy (overall correct predictions):",  round(accuracy,2))
# print("Auc:", round(auc,2))
    
# ## Precision e Recall
# recall = metrics.recall_score(y_test, predicted)
# precision = metrics.precision_score(y_test, predicted)
# print("Recall (all 1s predicted right):", round(recall,2))
# print("Precision (confidence when predicting a 1):", round(precision,2))
# print("Detail:")
# print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))


predicted_list = []
# predicted_ = model.predict(test_data[X_names].values)

model_list = ['Logistic Regression','Support Vector Machines']

for i in tqdm(range(0,10)):
    # model = random_search.best_estimator_
    # model.fit(X_train, y_train)
    # predicted_list.append(model.predict(test_data[X_names].values))

    
    for j in model_list:
        while True:
            Y_more, Y_less = model_selection.train_test_split(Y, test_size=percent)
            N_more, N_less = model_selection.train_test_split(N, test_size=percent)
            dtf_train = shuffle(trainPreprocess(pd.concat([Y_more,N_more])))
            dtf_valid = shuffle(trainPreprocess(pd.concat([Y_less,N_less])))
            X_train = dtf_train[X_names].values
            y_train = dtf_train["Y"].values
            X_test = dtf_valid[X_names].values
            y_test = dtf_valid["Y"].values
            models[j].fit(X_train, y_train)
            predictions = models[j].predict(X_test)
            accuracy = accuracy_score(predictions, y_test)
            precision= precision_score(predictions, y_test)
            recall = recall_score(predictions, y_test)

            if(accuracy>=0.82):
                print(accuracy,precision,recall)
                predicted_list.append(models[j].predict(test_data[X_names].values))
                break
    
ans = []
for i in tqdm(range(0,806)):
    result = []
    for j in predicted_list:
        result.append(j[i])
    if(result.count(0)>result.count(1)):
        ans.append([str(i)+'.0',str(0)])
    else:
        ans.append([str(i)+'.0',str(1)])
    print(i,result.count(0),result.count(1))
    
        
with open('ans.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['id', 'ans'])
    writer.writerows(ans)
        

# keys = range(0, len(predicted_))
# df_subm = pd.DataFrame({'id': keys, 'ans': predicted_})
# df_subm['id'] = df_subm["id"].astype("string")
# df_subm.to_csv('./data/ans_' + str(random_search.best_score_) + '_.csv', index=False)


# trainSet.to_csv('./data/split/trainSet_'+ str(random_search.best_score_) + '_.csv',index=False)
# validSet.to_csv('./data/split/validSet_'+ str(random_search.best_score_) + '_.csv',index=False)
