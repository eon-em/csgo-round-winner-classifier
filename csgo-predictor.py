# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.compose
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, KBinsDiscretizer
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Read csgo df
df = pd.read_csv('https://raw.githubusercontent.com/Theu011/CSGOClassification/main/csgo_round_snapshots.csv')

# Drop columns that has a mean or std of less than 5%
for column in df.select_dtypes(include=['float64']):
  if(df[column].mean() < 0.05) and (df[column].std() < 0.05):
    #print(column,"\t\tMean:", df[column].mean(),"\t\tStd:", df[column].std())
    df.drop(columns=column,axis=1, inplace=True)

# Columns Names
features = df.columns
# Df data
X = df.loc[:, features[0:-1]]
# Column to be predicted
y = df.loc[:, features[-1]]

column_transformer = sklearn.compose.ColumnTransformer(transformers=[
    ("map", OrdinalEncoder(), [3]),
    ("bomb_planted", OneHotEncoder(drop="first"), [4]),
    ("ct_players_alive", KBinsDiscretizer(n_bins=3),[14]),
    ("t_players_alive", KBinsDiscretizer(n_bins=3),[15])
], remainder='passthrough')
sc = StandardScaler()

X = column_transformer.fit_transform(X)
X = sc.fit_transform(X)

# Split 75% of the df into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21, train_size=0.75)
training_data, test_data = train_test_split(df, train_size=0.75)

# bomb_planted_games = training_data.groupby('round_winner')['bomb_planted'].value_counts(sort='True')
# bomb_planted_games = bomb_planted_games.unstack().transpose()

# bomb_planted_games.plot(kind='bar', color = ['green', 'red'], title="Partidas com bomba plantada x Partidas sem bomba plantada.")
# plt.show()

rf = RandomForestClassifier(random_state=21,max_depth=200,n_estimators=200, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# print(rf)
print("Accuracy: ", accuracy_score(y_test, rf_pred))
print("F1: ", f1_score(y_test, rf_pred, average='weighted'))