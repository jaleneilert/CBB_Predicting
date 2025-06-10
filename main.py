from xgboost import XGBClassifier
import proj_setup
import pickle
import sys

Xtest, ytest = proj_setup.main_setup()

with open('clf.pkl', 'rb') as f:
    loaded_model=pickle.load(f)

print("College Basketball March Madness Predictor \n\n")
yr = int(input("Type year to test (2024 or 2025):"))
if yr == 2025:
    indices = Xtest['YEAR'] == 2025
    Xtest = Xtest[indices]
    ytest = ytest[indices]
elif yr == 2024:
    indices = Xtest['YEAR'] == 2024
    Xtest = Xtest[indices]
    ytest = ytest[indices]
else:
    print("Please enter a valid year")
    running = 0
    sys.exit()

team = input("Type team to test (ex:Kansas_St. ... Iowa_St.:")
if team == 'N.C._St.' or team == 'North_Carolina_St.' or team == 'N.C._St.':
    index = Xtest['TEAM'] == 'N.C._St.'
    Xtest = Xtest[index]
    ytest = ytest[index]
    ytest = ytest.astype(int)
elif Xtest[Xtest['TEAM'] == team].empty == False:
    index = Xtest['TEAM'] == team
    Xtest = Xtest[index]
    ytest = ytest[index]
    ytest = ytest.astype(int)
else:
    print("Please enter a valid team")
    sys.exit()

# PRINT TEAM INFO
print(f"\nTeam: {Xtest['TEAM'].item()}")
print(f"Games Played: {Xtest['G'].item()}")
print(f"Wins: {Xtest['W'].item()}")
print(f"Wins above bubble: {Xtest['WAB'].item()}")
print(f"Power rating: {Xtest['BARTHAG'].item()}")
print(f"2 PT Percentage: {Xtest['2P_O'].item()}%")
print(f"3 PT Percentage: {Xtest['3P_O'].item()}%")
print(f"Tempo: {Xtest['ADJ_T'].item()}\n")

Xtest = Xtest.drop('TEAM', axis = 1)

pred = loaded_model.predict(Xtest)

if pred == 0:
    print("Model predicted team would miss the tournament :/")
else:
    print("Model predicted team would make the tournament!!")

if ytest.item() == 0:
    print("Actual: missed the tournament :/")
else:
    print("Actual: made the tournament :)")
