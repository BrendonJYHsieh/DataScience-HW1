# for data
import os
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
# for explainercl
from lime import lime_tabular
import csv
import random


def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < 9):
        return "cat"
    else:
        return "num"

def Preprocess(dtf,train=False):
    
    dtf = dtf.drop("Attribute1", axis=1)
    
    if(train):
        dtf = dtf.rename(columns={"Attribute17": "Y"})
        dtf = dtf.replace({'Y': {'Yes': 1, 'No': 0}})

    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    
    missingCol = ['Attribute3','Attribute4','Attribute5','Attribute6','Attribute7','Attribute9','Attribute11','Attribute12','Attribute13','Attribute15'];
    for i in missingCol:
        dtf[i] = imputer.fit_transform(dtf[[i]])
        
    
    dtf = dtf.dropna(axis=0).reset_index(drop=True)
    
    
    # dtf["Attribute3"] = dtf["Attribute3"].fillna(dtf["Attribute3"].mean())
    # dtf["Attribute4"] = dtf["Attribute4"].fillna(dtf["Attribute4"].mean())
    # dtf["Attribute5"] = dtf["Attribute5"].fillna(dtf["Attribute5"].mean())
    # dtf["Attribute6"] = dtf["Attribute6"].fillna(dtf["Attribute6"].mean())
    # dtf["Attribute7"] = dtf["Attribute7"].fillna(dtf["Attribute7"].mean())
    # dtf["Attribute9"] = dtf["Attribute9"].fillna(dtf["Attribute9"].mean())
    # dtf["Attribute11"] = dtf["Attribute11"].fillna(dtf["Attribute11"].mean())
    # dtf["Attribute12"] = dtf["Attribute12"].fillna(dtf["Attribute12"].mean())
    # dtf["Attribute13"] = dtf["Attribute13"].fillna(dtf["Attribute13"].mean())
    # dtf["Attribute15"] = dtf["Attribute15"].fillna(dtf["Attribute15"].mean())
    
    dummy = pd.get_dummies(
    dtf["Attribute8"],  prefix="Attribute8", drop_first=True)
    dtf = pd.concat([dtf, dummy], axis=1)
    dtf = dtf.drop("Attribute8", axis=1)
    dummy = pd.get_dummies(dtf["Attribute10"],
                        prefix="Attribute10", drop_first=True)
    dtf = pd.concat([dtf, dummy], axis=1)
    dtf = dtf.drop("Attribute10", axis=1)
    dummy = pd.get_dummies(dtf["Attribute14"],
                        prefix="Attribute14", drop_first=True)
    dtf = pd.concat([dtf, dummy], axis=1)
    dtf = dtf.drop("Attribute14", axis=1)
    dummy = pd.get_dummies(dtf["Attribute16"],
                        prefix="Attribute16", drop_first=True)
    dtf = pd.concat([dtf, dummy], axis=1)
    dtf = dtf.drop("Attribute16", axis=1)
    
    
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    if(train):
        X = scaler.fit_transform(dtf.drop("Y", axis=1))
        dtf_scaled = pd.DataFrame(X, columns=dtf.drop(
            "Y", axis=1).columns, index=dtf.index)
        dtf_scaled["Y"] = dtf["Y"]
        dtf = dtf_scaled
    else:
        
        X = scaler.fit_transform(dtf)
        dtf = pd.DataFrame(X, columns=dtf.columns, index=dtf.index)

    return dtf

dtf = pd.read_csv('train.csv')


indexes = []
count = 0

while count <= 10827:
    num = random.randrange(0,17104-count-1)
    if(dtf.iloc[[num]].values[0][16]=="No"):
        count = count+1
        dtf = dtf.drop(num, axis=0)
        dtf = dtf.reset_index(drop=True)

dtf = dtf.drop(indexes, axis=0)

dtf.to_csv('Processed_train.csv')

dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.2)

dtf_train = Preprocess(dtf,True)
dtf_test = Preprocess(dtf_test,True)

test_data = pd.read_csv('test.csv')
test_data = Preprocess(test_data,False)

#dtf_test = Preprocess(dtf_test)


# Data Encoding



#dtf_test.to_csv('Processed_test.csv')



# heatmap = dtf.isnull()

# for k,v in dic_cols.items():
#  if v == "num":
#    heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
#  else:
#    heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
# sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')
# plt.show()



X = dtf_train.drop("Y", axis=1).values
y = dtf_train["Y"].values
feature_names = dtf_train.drop("Y", axis=1).columns.tolist()
# Importance
model = ensemble.RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
model.fit(X, y)
importances = model.feature_importances_
# Put in a pandas dtf
dtf_importances = pd.DataFrame({"IMPORTANCE": importances,"VARIABLE": feature_names}).sort_values("IMPORTANCE",ascending=False)
dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
dtf_importances = dtf_importances.set_index("VARIABLE")

# Plot
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig.suptitle("Features Importance", fontsize=20)
ax[0].title.set_text('variables')

dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
    kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")
ax[1].title.set_text('cumulative')
dtf_importances[["cumsum"]].plot(kind="line", linewidth=4,
                                 legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)),
          xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
#plt.show()

#print(dtf_importances.index);

# print(dtf.drop("Y", axis=1).columns.tolist())

#dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=0.3)

results = []

#for i in range(1,len(dtf_importances.index)):
X_names = dtf_importances.index[:13]
X_train = dtf_train[X_names].values
y_train = dtf_train["Y"].values
X_test = dtf_test[X_names].values
y_test = dtf_test["Y"].values

# call model
model = ensemble.GradientBoostingClassifier()
## define hyperparameters combinations to try
param_dic = {'learning_rate':[0.01],      #weighting factor for the corrections by new trees when added to the model
'n_estimators':[500],  #number of trees added to the model
'max_depth':[2],    #maximum depth of the tree
'min_samples_split':[4],    #sets the minimum number of samples to split
'min_samples_leaf':[3],     #the minimum number of samples to form a leaf
'max_features':[2],     #square root of features is usually a good starting point
'subsample':[0.9]}       #the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.

random_search = model_selection.RandomizedSearchCV(model,param_distributions=param_dic, n_iter=1000,scoring="accuracy", n_jobs = -1,verbose=1).fit(X_train, y_train)
print("Best Model parameters:", random_search.best_params_)
print("Best Model mean accuracy:", random_search.best_score_)

print(random_search.best_params_['learning_rate'])

model = random_search.best_estimator_
#results.append([i,random_search.best_score_,param_dic['learning_rate'][0], param_dic['n_estimators'][0],param_dic['max_depth'][0],param_dic['min_samples_split'][0],param_dic['min_samples_leaf'][0],param_dic['max_features'][0],param_dic['subsample'][0]])
# # ## train
model.fit(X_train, y_train)
## test
predicted_prob = model.predict_proba(dtf_test[X_names].values)[:,1]
predicted = model.predict(dtf_test[X_names].values)





## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc,2))
    
## Precision e Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

classes = np.unique(y_test)
fig, ax = plt.subplots()
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
ax.set(xlabel="Pred", ylabel="True", title="Confusion matrix")
ax.set_yticklabels(labels=classes, rotation=0)
plt.show()




predicted_ = model.predict(test_data[X_names].values)

ans = []
for index, item in enumerate(predicted_):
    ans.append([str(index)+'.0',str(item)])

keys = range(0, len(predicted_))
df_subm = pd.DataFrame({'id': keys, 'ans': predicted_})
df_subm['id'] = df_subm["id"].astype("string")
df_subm.to_csv('ans.csv', index=False)

with open('ans.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['id', 'ans'])
    writer.writerows(ans)
    
    
    #writer.writerow(['i', 'accuracy', 'learning_rate', 'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'subsample'])
    #writer.writerows(results)

