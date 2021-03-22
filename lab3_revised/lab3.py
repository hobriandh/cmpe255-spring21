import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
  
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        self.X_test = None
        self.y_test = None
        

    def define_feature(self, features):
        feature_cols = features
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y
    
    def train(self, features):
        # split X and y into training and testing sets
        X, y = self.define_feature(features)
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self, features):
        model = self.train(features)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)

    def find_correlation(self):
        #finding correlation
        correlation = self.pima.corr()
        f,ax=plt.subplots(figsize=(12,7))
        sns.heatmap(correlation,cmap='viridis',annot=True)
        plt.figure(1)
        plt.title("Correlation between features",weight='bold', fontsize=18)
        plt.show()
    
if __name__ == "__main__":
    # classifier that we are using
    classifier = DiabetesClassifier()

    # holds the baseline solution
    base_features = ['pregnant', 'insulin', 'bmi', 'age']
    result = classifier.predict(base_features)
    score = classifier.calculate_accuracy(result)
    con_matrix = classifier.confusion_matrix(result)

    # improving model by selecting all features
    all_features = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    result1 = classifier.predict(all_features)
    score1 = classifier.calculate_accuracy(result1)
    con_matrix1 = classifier.confusion_matrix(result1)

    # us the correlation graph to find most correlated features to the label
    classifier.find_correlation()

    # choosing the three highest correlated features
    specific_features = ['glucose', 'bmi', 'age']
    result2 = classifier.predict(specific_features)
    score2 = classifier.calculate_accuracy(result2)
    con_matrix2 = classifier.confusion_matrix(result2)

    # choosing the five highest correlated features
    # skipping pregnant because that is highly correlated with age
    # (since age is an important feature, being pregnant might just be an effect of older age)
    more_specific_features = ['glucose', 'insulin', 'bmi', 'pedigree', 'age']
    result3 = classifier.predict(more_specific_features)
    score3 = classifier.calculate_accuracy(result3)
    con_matrix3 = classifier.confusion_matrix(result3)

    print("| Experiment |      Accuracy      | Confusion Matrix  |                 Comment                               |")
    print("| Baseline   |", round(score, 16), "|", *con_matrix, "| Is the baseline solution given                        |")
    print("| Solution1  |", round(score1, 16), "|", *con_matrix1, "| All Features                                          |")
    print("| Solution2  |", round(score2, 16), "|", *con_matrix2, "| Choosing 3 specific features with high correlation    |")
    print("| Solution3  |", round(score3, 16), "|", *con_matrix3, "| Choosing 5 specific features with high correlation    |")