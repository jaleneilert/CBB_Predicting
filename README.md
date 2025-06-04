# College Basketball March Madness Predictor (w/ XGBoost) 🏀
![Static Badge](https://img.shields.io/badge/Status-In_progress-yellow)

This project is aimed at predicting which teams will make the tournament in a given year. Training and validation utilize a 7-window sliding setup (3 years train, 1 validate) to see how the model generalizes over time.

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
