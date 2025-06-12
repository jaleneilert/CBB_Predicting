import proj_setup
from sklearn.metrics import accuracy_score, root_mean_squared_error, ConfusionMatrixDisplay
from xgboost import XGBClassifier, callback, plot_tree, to_graphviz
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import os
import time
import pickle

t0 = time.time()
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

# best parameters found through grid search in hyper_params.py

''' 92%
best_params = { 'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 1,
                'colsample_bytree': 0.8,
                'max_delta_step': 2
               }
'''

# 92.4%
best_params = { 'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 1,
                'colsample_bytree': 0.8,
                'max_delta_step': 2,
                'reg_lambda': 1,
                'min_split_loss': 0.2
               }

# test 10 different seeds to test model stability
seeds = range(10)
# 7 windows

for seed in seeds:
  for i in  range(7):

    Xtrain, ytrain, Xvalid, yvalid = proj_setup.init_window(X, y, i)

    # --- START OF MODEL PROGRAM ---

    # setting is binary/logistic as we are looking for 0 being not making tournament and 1 making tournament
    clf = XGBClassifier(objective='binary:logistic',
                        base_score=0.192,
                        eval_metric='aucpr',
                        random_state=seed,
                        scale_pos_weight=scale_weight,
                        learning_rate = 0.25,
                        n_estimators=60,
                        **best_params)

    clf.fit(Xtrain, ytrain)

    ypred = clf.predict(Xvalid)

    accuracy.append(accuracy_score(yvalid, ypred))

    '''
        if seed == 0:
      #ConfusionMatrixDisplay.from_predictions(yvalid, ypred)
      #plt.show()
      # get first three trees
      for num_tree in range(3):
          tree_dot = to_graphviz(clf, num_trees=num_tree)
          # Save the dot file
          dot_file_path = f"xgboost_tree_{num_tree}.dot"
          tree_dot.save(dot_file_path)
          # Convert dot file to png and display
          with open(dot_file_path) as f:
              dot_graph = f.read()
          # Use graphviz to display the tree
          graph = graphviz.Source(dot_graph)
          graph.render(f"xgboost_tree_{num_tree}")
          # Optionally, visualize the graph directly
          graph
    '''

  #print(accuracy)
  #print(np.mean(accuracy))
  avg_accuracies.append(np.mean(accuracy))
  accuracy.clear()

#print(avg_accuracies)
print(f"The avg validation accuracy is {np.mean(avg_accuracies)}")

clf = XGBClassifier(objective='binary:logistic',
                    base_score=0.192,
                    eval_metric='aucpr',
                    scale_pos_weight=scale_weight,
                    learning_rate = 0.25,
                    n_estimators=60,
                    **best_params)

clf.fit(X, y)

ypred_t = clf.predict(Xtest)
accuracy_t = accuracy_score(ytest, ypred_t)
print(f"The test accuracy is {accuracy_t}")

t1 = time.time()
print(f"Run time: {t1-t0}s")

with open('clf.pkl', 'wb') as f:
   pickle.dump(clf, f)
