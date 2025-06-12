# College Basketball March Madness Predictor (w/ XGBoost) 🏀
![Static Badge](https://img.shields.io/badge/Conda-24.11.3-green)
![Static Badge](https://img.shields.io/badge/scikit_learn-1.4.2-blue)
![Static Badge](https://img.shields.io/badge/xgboost-2.1.4-blue)
![Static Badge](https://img.shields.io/badge/pandas-2.2.2-blue)
![Static Badge](https://img.shields.io/badge/matplotlib-3.8.4-blue)
![Static Badge](https://img.shields.io/badge/graphviz-0.20.1-blue)

This project is aimed at predicting which teams will receive a tournament bid with given stats such as their 3PT%, FG%, WAB, etc. After optimizing hyperparameters, the model was able to achieve a **92.42% test accuracy** and **91.50% average validation accuracy**.

<img src = "https://github.com/user-attachments/assets/f4abd1d7-2e7a-4913-8e06-a5fa02697fc6" width = 475px height = 300px>

📸: [K-State Athletic Release](https://www.ksal.com/k-state-moves-on-to-elite-8-with-instant-classic-over-michigan-state/)

## XGBoost 
XGBoost is a machine learning algorithm that utilizes gradient boosting with decision trees. The algorithm differs from random forest as it constructs trees sequentially instead of multiple in parallel.

For classification, logistic loss is used when calculating the gradient to see the direction of error and its magnitude. This allows the algorithm to move in the direction where the loss is minimized. These errors guide the creation of new trees that come after them. 

![image](https://github.com/user-attachments/assets/418b0fb1-3577-4095-877e-c8c32f4933ba)

This is the first tree the model outputs. The leaf nodes have a value that is a probability but is in the form of log(odds). We can convert this by using the logistic function:
```math
Probability = \frac{e^{log(odds)}}{1 + e^{log(odds)}}
```

```math
Probability\ of\ left\ most\ leaf = \frac{e^{0.331378371}}{1 + e^{0.331378371}} = 0.58
```
The tree gives us a predicted probability of 0.58 for a team that has 6 or more wins below the bubble, less than 17 wins, and allowed FG% of 45.6%. This would mean the model thinks the team is likely middle of the pack, fighting for a final pick. With the team's lacking resume, future trees will likely further reduce the overall prediction (being in favor of the team not making the tournament). 

<img src =https://github.com/user-attachments/assets/affc8d13-4e81-4c94-bc58-a248cc6dfe52 width = 1000px height = 425px>

This is the third graph that has some more refined findings. 

A team that has 2 wins or more below the bubble, less than 16 wins, has an allowed 3P% of 30.2%, and less than 37.2% offensive 3P% has a 0.43 predicted probability of making the tournament. This individual tree prediction is low but in combination with other trees, will most likely have this team not making the tournament as well.


## Dataset
College Basketball stats from 2013 - 2025 (not incluing the 2020 season as there was no post season). 

Data outlines different statistical metrics over the basketball season and results in the postseason.

https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?resource=download

### Modifications
In the data folder is a modified version of the dataset

- Added the results of the 2025 postseason

- RANK, 3PR, and 3PD were removed as they don't appear in the main cbb.csv with years 2024 and prior.

- Removed SEED in code (NCAA seed in tournament) as this would tell us whether the team made the tournament.

- Removed TEAM in code to better generalize which stats are more predictive of a tournament bid

### Encoding
The project utilizes binary encoding for the post-season one-hot encoding for the conference (found in *proj_setup*). 

- *encode_postseason(cbb_df)* function encodes all tournament misses 'NA', 'N/A', 'nan' to 0 and the rest (making the tournament) to 1. 

<img src = "https://github.com/user-attachments/assets/b6805915-88e2-4377-94d3-1baf48dce7f2" width = 400px height = 300px>

- *encode_conference(cbb_df, keep_team:bool = False)* adds a new column for my defined power five conferences and one other for all other conferences (Mid-Major). A one is placed in that column of that team's conference and a 0 in all other conference columns.

<img src = "https://github.com/user-attachments/assets/242de795-2bb5-4586-85d0-d20c0dcd983c" width = 400px height = 300px>

## Training/Validation
Training and validation utilize a 7-window sliding setup (3 years train, 1 validate) to see how the model generalizes over time.

The for loops seen in *default_params* and *optimized modules* define the number of windows that the program will make. 
```python
for i in range(7)
```
I pass in i from the for loop to increment the range of years in the 'years' array to create the desired window. Then I check the dataset's year column to see which ones correspond to the newly defined 'training_years' and 'validation_year'. This creates a boolean array (1 if its is data from one of the years we want 0 if not). From there I grab the indices of the training and validation sets. Using loc on the original dataframe creates new dataframes for the newly defined training and validation window.

*proj_setup.py*
```python
def init_window(X, y, i):
  # need to account for missing 2021 data due to no postseason

    years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]

    training_years = years[i: i + 3]
    validation_year = years[i + 3]

    # bool array telling us which data in X have the years we want
    train_m = X['YEAR'].isin(training_years)
    validation_m = X['YEAR'] == validation_year

    train_indices = np.where(train_m)[0]
    valid_indices = np.where(validation_m)[0]

    Xtrain = X.loc[train_indices]
    ytrain = y.loc[train_indices]
    Xvalid = X.loc[valid_indices]
    yvalid = y.loc[valid_indices]

    return  Xtrain, ytrain, Xvalid, yvalid

```

<img src = "https://github.com/user-attachments/assets/2a857f3f-a834-4435-a107-eac58053dcf1" width = 500px height = 200px>

### Hyperparameter Optimization
#### Learning Rate *(learning_rates.py)*

Learing rate is how fast the model can learn. I tested different values ranging from 0 to 1, but found that between 0 and 0.3 was where the model had the best precision.  

![image](https://github.com/user-attachments/assets/054b7acc-d287-4c1b-bb69-8234e7d2f4b4)

From the simulation, the best precision score of 0.6678 was achieved with 0.25, so this became the model's new learning rate.

#### Number of Estimators *(diff_estimators.py)*

Having the learning rate, we can now tune the number of estimators. The number of estimators (n_estimators) in XGBoost determines the number of trees that the model will produce. Increasing this number can lead to overfitting and be computationally expensive. Therefore, I am looking for the best precision with the lowest n_estimators I can get.

![image](https://github.com/user-attachments/assets/c70a305d-bd8e-4456-98f3-89840a48669d)

From the simulation, we can see the model achieves its highest precision between 60 and 100 and then falls of slowly after. For this reason, I went with 60 estimators to achieve the balance of precision and lowering model variance.

#### Important Hyperparameters *(hyper_params.py)*

- max_depth (default:6) - The max depth of the decision tree
- min_child_weight (default: 1) - Minimum sum of instance weights needed to create a new child node.
- subsample (default: 1) - Fraction of training samples used to create a tree.
- cosample_by_tree (default: 1) - Fraction of features used to create a tree.

These parameters were tuned using the new learning_rate and n_estimators with GridSearchCV. I defined different values for each of the different hyperparameters. The program iterates through all of them and returns the combination with the best accuracy. 

*Note*: GridSearchCV requires a splitter in its function. I redefined the init_window to a class CustomSlidingWindow in which its split function performs the same task, but only returns the indices of each window. 

## Conclusion
The optimized model was able to achieve a better training accuracy while also being almost a second faster in run time. Using XGBoost for this project was a good choice and I can see why people would use it over algorithms like random forest.

<img src = "https://github.com/user-attachments/assets/8c7424fd-bfc9-4f03-aa86-0167d6ffac5b" width = 360px height = 180px>
<img src = "https://github.com/user-attachments/assets/a659237e-2b45-4b4f-a7e0-e041cb30d21b" width = 360px height = 180px>

Resources for more research:

Statquest series: https://www.youtube.com/watch?v=OtD8wVaFm6E&t=1260s

Gradient Boosting: https://www.youtube.com/watch?v=en2bmeB4QUo&t=553s

Important Model Parameters: https://xgboosting.com/most-important-xgboost-hyperparameters-to-tune/






