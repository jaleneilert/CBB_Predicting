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

# best parameters found through grid search in hyper_params.py
best_params = { 'eta' : 0.4,
                'gamma': 0.8,
                'max_delta_step': 3,
                'min_child_weight': 1,
                'num_parallel_tree': 10,
               }
# test 10 different seeds to test model stability
seeds = range(10)
# 7 windows

for seed in seeds:
  for i in  range(7):

    Xtrain, ytrain, Xvalid, yvalid = proj_setup.init_window(X, y, i)

    # --- START OF MODEL PROGRAM ---

    # setting is binary/logistic as we are looking for 0 being not making tournament and 1 making tournament
    es = callback.EarlyStopping(rounds=10, save_best=True)
    em = callback.EvaluationMonitor(show_stdv=True)

    clf = XGBClassifier(objective='binary:logistic',
                        base_score=0.192,
                        eval_metric='aucpr',
                        random_state=seed,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        **best_params,
                        callbacks= [es, em])

    clf.fit(Xtrain, ytrain, eval_set=[(Xvalid, yvalid)])

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
