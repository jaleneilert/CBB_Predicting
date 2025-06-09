import proj_setup
from sklearn.metrics import accuracy_score, root_mean_squared_error, ConfusionMatrixDisplay
from xgboost import XGBClassifier, callback
import matplotlib.pyplot as plt
import numpy as np

cbb_df = proj_setup.read('./data/cbb.csv', './data/cbb25.csv')

cbb_df = proj_setup.encode_postseason(cbb_df)

cbb_df = proj_setup.encode_conference(cbb_df)

X, Xtest = proj_setup.split_data(cbb_df)

y = X['POSTSEASON']
X = X.drop('POSTSEASON', axis = 1)

ytest = Xtest['POSTSEASON']
Xtest = Xtest.drop('POSTSEASON', axis = 1)

accuracy = []

avg_accuracies = []

# test 10 different seeds to test model stability
seeds = range(10)
# 7 windows

for seed in seeds:
  for i in  range(7):

    Xtrain, ytrain, Xvalid, yvalid = proj_setup.init_window(X, y, i)

    # --- START OF MODEL PROGRAM ---

    # setting is binary/logistic as we are looking for 0 being not making tournament and 1 making tournament
    clf = XGBClassifier(objective='binary:logistic',
                        eval_metric='aucpr',
                        random_state=seed)

    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xvalid)

    accuracy.append(accuracy_score(yvalid, ypred))

    if seed == 0:
      ConfusionMatrixDisplay.from_predictions(yvalid, ypred)
      plt.show()

  print(accuracy)
  print(np.mean(accuracy))
  avg_accuracies.append(np.mean(accuracy))
  accuracy.clear()

print(avg_accuracies)
print(np.mean(avg_accuracies))
print(f"The avg validation accuracy is {np.mean(avg_accuracies)}")


clf = XGBClassifier(objective='binary:logistic',
                    eval_metric='aucpr')

clf.fit(X, y)

ypred_t = clf.predict(Xtest)
accuracy_t = accuracy_score(ytest, ypred_t)
print(f"The test accuracy is {accuracy_t}")

