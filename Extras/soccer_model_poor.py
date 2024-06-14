

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

data = pd.read_csv('PremierLeague.csv')

data.dropna(inplace=True)

categorical_columns = ['HomeTeam', 'AwayTeam', 'FullTimeResult', 'HalfTimeResult', 'Referee']

label_encoders = {}

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column + 'Encoded'] = label_encoders[column].fit_transform(data[column])

average_stats = data.groupby('HomeTeam').agg({
    'FullTimeHomeTeamGoals': 'mean',
    'FullTimeAwayTeamGoals': 'mean',
    'HalfTimeHomeTeamGoals': 'mean',
    'HalfTimeAwayTeamGoals': 'mean',
    'HomeTeamShots': 'mean',
    'AwayTeamShots': 'mean',
    'HomeTeamShotsOnTarget': 'mean',
    'AwayTeamShotsOnTarget': 'mean',
    'HomeTeamCorners': 'mean',
    'AwayTeamCorners': 'mean',
    'HomeTeamFouls': 'mean',
    'AwayTeamFouls': 'mean',
    'HomeTeamYellowCards': 'mean',
    'AwayTeamYellowCards': 'mean',
    'HomeTeamRedCards': 'mean',
    'AwayTeamRedCards': 'mean'
}).reset_index()

average_stats.columns = [
    'Team', 'AvgFullTimeHomeTeamGoals', 'AvgFullTimeAwayTeamGoals', 'AvgHalfTimeHomeTeamGoals',
    'AvgHalfTimeAwayTeamGoals', 'AvgHomeTeamShots', 'AvgAwayTeamShots', 'AvgHomeTeamShotsOnTarget',
    'AvgAwayTeamShotsOnTarget', 'AvgHomeTeamCorners', 'AvgAwayTeamCorners', 'AvgHomeTeamFouls',
    'AvgAwayTeamFouls', 'AvgHomeTeamYellowCards', 'AvgAwayTeamYellowCards', 'AvgHomeTeamRedCards',
    'AvgAwayTeamRedCards'
]

data = data.merge(average_stats, how='left', left_on='HomeTeam', right_on='Team').drop('Team', axis=1)
data = data.merge(average_stats, how='left', left_on='AwayTeam', right_on='Team', suffixes=('_Home', '_Away')).drop('Team', axis=1)

data['Home_Away_Goal_Diff'] = data['AvgFullTimeHomeTeamGoals_Home'] - data['AvgFullTimeAwayTeamGoals_Away']
data['Home_Away_Shots_Diff'] = data['AvgHomeTeamShots_Home'] - data['AvgAwayTeamShots_Away']
data['Home_Away_ShotsOnTarget_Diff'] = data['AvgHomeTeamShotsOnTarget_Home'] - data['AvgAwayTeamShotsOnTarget_Away']
data['Home_Away_Corners_Diff'] = data['AvgHomeTeamCorners_Home'] - data['AvgAwayTeamCorners_Away']
data['Home_Away_Fouls_Diff'] = data['AvgHomeTeamFouls_Home'] - data['AvgAwayTeamFouls_Away']
data['Home_Away_YellowCards_Diff'] = data['AvgHomeTeamYellowCards_Home'] - data['AvgAwayTeamYellowCards_Away']
data['Home_Away_RedCards_Diff'] = data['AvgHomeTeamRedCards_Home'] - data['AvgAwayTeamRedCards_Away']

x = data[[
    'HomeTeamEncoded', 'AwayTeamEncoded', 'RefereeEncoded', 'B365HomeTeam', 'B365Draw', 'B365AwayTeam',
    'B365Over2.5Goals', 'B365Under2.5Goals', 'MarketMaxHomeTeam', 'MarketMaxDraw', 'MarketMaxAwayTeam',
    'MarketAvgHomeTeam', 'MarketAvgDraw', 'MarketAvgAwayTeam', 'MarketMaxOver2.5Goals', 'MarketMaxUnder2.5Goals',
    'MarketAvgOver2.5Goals', 'MarketAvgUnder2.5Goals', 'AvgFullTimeHomeTeamGoals_Home', 'AvgFullTimeAwayTeamGoals_Home',
    'AvgHalfTimeHomeTeamGoals_Home', 'AvgHalfTimeAwayTeamGoals_Home', 'AvgHomeTeamShots_Home', 'AvgAwayTeamShots_Home',
    'AvgHomeTeamShotsOnTarget_Home', 'AvgAwayTeamShotsOnTarget_Home', 'AvgHomeTeamCorners_Home', 'AvgAwayTeamCorners_Home',
    'AvgHomeTeamFouls_Home', 'AvgAwayTeamFouls_Home', 'AvgHomeTeamYellowCards_Home', 'AvgAwayTeamYellowCards_Home',
    'AvgHomeTeamRedCards_Home', 'AvgAwayTeamRedCards_Home', 'AvgFullTimeHomeTeamGoals_Away', 'AvgFullTimeAwayTeamGoals_Away',
    'AvgHalfTimeHomeTeamGoals_Away', 'AvgHalfTimeAwayTeamGoals_Away', 'AvgHomeTeamShots_Away', 'AvgAwayTeamShots_Away',
    'AvgHomeTeamShotsOnTarget_Away', 'AvgAwayTeamShotsOnTarget_Away', 'AvgHomeTeamCorners_Away', 'AvgAwayTeamCorners_Away',
    'AvgHomeTeamFouls_Away', 'AvgAwayTeamFouls_Away', 'AvgHomeTeamYellowCards_Away', 'AvgAwayTeamYellowCards_Away',
    'AvgHomeTeamRedCards_Away', 'AvgAwayTeamRedCards_Away', 'Home_Away_Goal_Diff', 'Home_Away_Shots_Diff',
    'Home_Away_ShotsOnTarget_Diff', 'Home_Away_Corners_Diff', 'Home_Away_Fouls_Diff', 'Home_Away_YellowCards_Diff',
    'Home_Away_RedCards_Diff'
]]

y = data['FullTimeResultEncoded']

x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=92001)

rf = RandomForestClassifier(random_state=92002)
gb = GradientBoostingClassifier(random_state=92002)
lr = LogisticRegression(max_iter=5000)

ensemble_model = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')
ensemble_model.fit(x_Train, y_Train)

y_Prediction = ensemble_model.predict(x_Test)

accuracy = accuracy_score(y_Test, y_Prediction)
mse = mean_squared_error(y_Test, y_Prediction)
mae = mean_absolute_error(y_Test, y_Prediction)

print(f'Accuracy: {accuracy}')
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')