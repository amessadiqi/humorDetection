from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import pickle


class Scores:
    def __init__(self, dataset):
        self.dataset = dataset
        
        self.methods = []
        self.models = []

        self.prepare_training_set()

        self.fit_xgboost()
        self.fit_random_forest()
        self.fit_logistic_regression()


    def prepare_training_set(self):
        X, X_test, y, y_test = train_test_split(self.dataset.iloc[:,1:].to_numpy(), self.dataset.iloc[:,0].to_numpy(), test_size=0.2, random_state=11)
        
        self.X = X
        self.X_test = X_test
        self.y = y
        self.y_test = y_test


    def fit_xgboost(self):
        model = xgb.XGBClassifier(random_state=1,learning_rate=0.01)
        model.fit(self.X, self.y)

        self.methods.append("XGBoost")
        self.models.append(model)


    def fit_random_forest(self):
        model = RandomForestClassifier(max_depth=8, random_state=1123).fit(self.X, self.y)

        self.methods.append("Random Forests")
        self.models.append(model)


    def fit_logistic_regression(self):
        model = LogisticRegression(solver='lbfgs', max_iter=1000 , random_state=5).fit(self.X, self.y)

        self.methods.append("Logistic Regression")
        self.models.append(model)


    def getScores(self, model):
        yPred = model.predict(self.X_test)
        return (accuracy_score(self.y_test, yPred), 
                precision_score(self.y_test, yPred), 
                recall_score(self.y_test, yPred),
                f1_score(self.y_test,yPred))


    def displayScore(self, scores):
        print("Accuracy: {} ".format(scores[0]))
        print("Precision: {} ".format(scores[1]))
        print("Recall: {} ".format(scores[2]))
        print("F1 Score: {} ".format(scores[3]))
        print("------------------------------")


    def displayScores(self):
        for i in range(len(self.models)):
            print(self.methods[i])
            scores = self.getScores(self.models[i])
            self.displayScore(scores)


    def save_weight(self, dest):
        for i in range(len(self.models)):
            filename = dest + '/' + self.methods[i] +'.sav'
            pickle.dump(self.models[i], open(filename, 'wb'))


if __name__=='__main__':
    pass
