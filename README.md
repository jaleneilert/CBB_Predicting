# College Basketball March Madness Predictor (w/ XGBoost) 🏀
![Static Badge](https://img.shields.io/badge/Status-In_progress-yellow)
![Static Badge](https://img.shields.io/badge/Conda-24.11.3-green)
![Static Badge](https://img.shields.io/badge/scikit_learn-1.4.2-blue)
![Static Badge](https://img.shields.io/badge/xgboost-2.1.4-blue)
![Static Badge](https://img.shields.io/badge/pandas-2.2.2-blue)
![Static Badge](https://img.shields.io/badge/matplotlib-3.8.4-blue)
![Static Badge](https://img.shields.io/badge/graphviz-0.20.1-blue)

This project is aimed at predicting which teams will receive a tournament bid with given stats such as their 3PT%, FT%, FG%, etc.

<img src = "https://github.com/user-attachments/assets/f4abd1d7-2e7a-4913-8e06-a5fa02697fc6" width = 475px height = 300px>

📸: [K-State Athletic Release](https://www.ksal.com/k-state-moves-on-to-elite-8-with-instant-classic-over-michigan-state/)

## Dataset
College Basketball stats from 2013 - 2025 (not incluing the 2020 season as there was no post season). 

Data outlines different statistical metrics over the basketball season and results in postseason.

https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset?resource=download

### Modifications
In the data folder is a modified version of the dataset

- Added the results of the 2025 postseason

- RANK, 3PR, and 3PD were removed as they don't appear in the main cbb.csv with years 2024 and prior.

- Removed SEED in code (NCAA seed in tournament) as this would tell us whether the team made the tournament.

- Removed TEAM in code to better generalize which stats are more predictive of a tournament bid

### Encoding
The project utilizes binary encoding for the post-season one-hot encoding for the conference (found in *proj_setup*). 

The *encode_postseason(cbb_df)* function encodes all tournament misses 'NA', 'N/A', 'nan' to 0 and the rest (making the tournament) to 1. 
<img src = "https://github.com/user-attachments/assets/b6805915-88e2-4377-94d3-1baf48dce7f2" width = 400px height = 300px>

The *encode_conference(cbb_df, keep_team:bool = False)* adds a new column for my defined power five conferences and one other for all other conferences (Mid-Major). A one is placed in that column of that team's conference and a 0 in all other conference columns.

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


