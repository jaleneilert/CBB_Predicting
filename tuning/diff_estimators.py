import proj_setup
from sklearn.metrics import accuracy_score, average_precision_score
from xgboost import XGBClassifier, callback, plot_tree, to_graphviz
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import os

cbb_df = proj_setup.read('./data/cbb.csv', './data/cbb25.csv')

cbb_df = proj_setup.encode_postseason(cbb_df)

cbb_df = proj_setup.encode_conference(cbb_df)

X, Xtest = proj_setup.split_data(cbb_df)

scale_weight = 2843/680

y = X['POSTSEASON']
X = X.drop('POSTSEASON', axis = 1)

ytest = Xtest['POSTSEASON']
Xtest = Xtest.drop('POSTSEASON', axis = 1)

accuracy = []

avg_accuracies = []

precisions = []

avg_precisions = []

# different num of estimators
diff_n_estimators = range(10, 400, 10)

for ne in diff_n_estimators:
  for i in  range(7):

    Xtrain, ytrain, Xvalid, yvalid = proj_setup.init_window(X, y, i)

    # --- START OF MODEL PROGRAM ---

    # setting is binary/logistic as we are looking for 0 being not making tournament and 1 making tournament

    clf = XGBClassifier(objective='binary:logistic',
                        base_score=0.192,
                        eval_metric='aucpr',
                        learning_rate=0.25,
                        n_estimators=ne,
                        scale_pos_weight=scale_weight)

    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xvalid)

    precisions.append(average_precision_score(yvalid, ypred))

    accuracy.append(accuracy_score(yvalid, ypred))

  print(accuracy)
  print(np.mean(accuracy))
  avg_accuracies.append(np.mean(accuracy))
  accuracy.clear()
  print(f"Average precision score: {np.mean(precisions)}")
  avg_precisions.append(np.mean(precisions))
  precisions.clear()


print(avg_accuracies)

print(avg_precisions)
plt.plot(diff_n_estimators, avg_precisions)
plt.xlabel('n_estimators')
plt.ylabel('Average Precision Score')
plt.title('XG Boost Average Precision Score w/ different n_estimators')
#plt.xticks(learning_rates)
plt.show()



