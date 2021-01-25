import pandas as pd 
import glob
import os
path = "C:\\Users\\Saad\\Desktop\\Numba\\Preprocessed_data"

all_files = glob.glob(os.path.join(path, "*.csv"))
df_from_each_file = (pd.read_csv(f,index_col=0) for f in all_files[:])
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

def cleaner(df):
    df.drop(columns=["text","textSeq"],inplace=True)
    df["humor"] = df["humor"].astype(int)
    return df

dataFrame = cleaner(concatenated_df)

#
from sklearn.model_selection import train_test_split
X, X_test, y, y_test = train_test_split(dataFrame.iloc[:,1:].to_numpy(), dataFrame.iloc[:,0].to_numpy(), test_size=0.2, random_state=11)

"""
#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model_1 = KNeighborsClassifier(n_neighbors=7).fit(X, y)

#Support Vector Machine
from sklearn.svm import SVC
model_2 =  SVC(gamma='auto').fit(X, y)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_3 = DecisionTreeClassifier(random_state=7).fit(X,y)
"""
#Random Forests
from sklearn.ensemble import RandomForestClassifier
model_4 = RandomForestClassifier(max_depth=8, random_state=1123).fit(X, y)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model_5 = LogisticRegression(solver='lbfgs', max_iter=1000 , random_state=5).fit(X, y)

#xGboost
import xgboost as xgb
model_6=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model_6.fit(X, y)

#QST8
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
models = [model_4,model_5,model_6]
#models = [model_1,model_2,model_3,model_4,model_5]
#names = ["K-Nearest Neighbors", "Support Vector Machine", "Decision Tree","Random Forests","Logistic Regression"]
names = ["Random Forests","Logistic Regression","XGBoost"]

def getScores(model,X_test,y_test):
    yPred = model.predict(X_test)
    return (accuracy_score(y_test, yPred), 
            precision_score(y_test, yPred), 
            recall_score(y_test, yPred),
            f1_score(y_test,yPred))


def displayScore(scores):
      print("Accuracy: {} ".format(scores[0]))
      print("Precision: {} ".format(scores[1]))
      print("Recall: {} ".format(scores[2]))
      print("F1 Score: {} ".format(scores[3]))
      print("------------------------------")

def displayScores(models,names,X_test,y_test):
    for i in range(len(models)):
      print(names[i])
      scores = getScores(models[i],X_test,y_test)
      displayScore(scores)

displayScores(models,names,X_test,y_test)



from sklearn.ensemble import StackingClassifier
estimators = [ (names[i],models[i]) for i in range(len(models)-1) ]
stack_model = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression()).fit(X, y)

print("Stacking classifier")

displayScore(getScores(stack_model,X_test,y_test))

## Saving weights
import pickle
for i in range(len(models)):
    filename = 'weights\\' + names[i] +'.sav'
    pickle.dump(models[i], open(filename, 'wb'))