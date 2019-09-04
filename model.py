import pandas as pd
import numpy as np
import visuals as vs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,fbeta_score,make_scorer
from sklearn.model_selection import GridSearchCV
import pickle
import os
from sklearn.base import clone

filename = 'finalized_model.sav'



df = pd.read_csv('census.csv')

print(df.head())

n_records = df.shape[0]
n_greater_50k = sum(df['income']=='>50K')
n_at_most_50k = sum(df['income']=='<=50K')
greater_percent = n_greater_50k/n_records*100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

income_raw = df['income']
features_raw = df.drop('income', axis=1)

vs.distribution(df)

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
##vs.distribution(features_log_transformed, transformed = True)

scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
print(features_log_minmax_transform.head(n = 5))

features_final = pd.get_dummies(features_log_minmax_transform)

income = income_raw.map({'>50K':1,'<=50K':0})
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
print(encoded)

TP = np.sum(income)
FP = income.count() - TP
TN = 0
FN = 0
accuracy = float(TP)/(TP+FP)
recall = float(TP)/(TP+FN)
precision = accuracy

fscore = (1+0.5**2)*(precision*recall)/(0.5**2*precision+recall)


X_train, X_test, y_train, y_test = train_test_split(features_final,income,test_size = 0.2,random_state = 0)

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):

    results={}

    start = time()
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time()

    results['train_time'] = end-start

    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()

    results['pred_time'] = end-start

    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)

    results['acc_test'] = accuracy_score(y_test,predictions_test)

    results['f_train'] = fbeta_score(y_train[:300],predictions_train,0.5)

    results['f_test'] = fbeta_score(y_test,predictions_test,0.5)

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

if (os.path.exists('finalized_model.sav')):
    best_clf = pickle.load(open(filename, 'rb'))
    best_predictions = best_clf.predict(X_test)

else:
    clf_A = GradientBoostingClassifier(random_state=42)
    clf_B = RandomForestClassifier(random_state=42)
    clf_C = LogisticRegression(random_state=42)

    samples_100 = len(y_train)
    samples_10 = int(0.10 * len(y_train))
    samples_1 = int(0.01 * len(y_train))

    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

    # Run metrics visualization for the three supervised learning models chosen
    ##vs.evaluate(results, accuracy, fscore)

    clf = GradientBoostingClassifier(random_state=42)

    parameters = {'n_estimators': [100, 300, 500], 'learning_rate': [0.1, 1, 1.3]}

    scorer = make_scorer(fbeta_score, beta=0.5)

    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    # Make predictions using the unoptimized and model
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))

    pickle.dump(best_clf, open(filename, 'wb'))

importances = best_clf.feature_importances_

vs.feature_plot(importances, X_train, y_train)

X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

clf = (clone(best_clf)).fit(X_train_reduced, y_train)
reduced_predictions = clf.predict(X_test_reduced)

print ("Final Model trained on full data\n------")
print ("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print ("\nFinal Model trained on reduced data\n------")
print ("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


