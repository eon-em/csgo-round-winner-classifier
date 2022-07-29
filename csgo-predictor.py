# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.compose
from sklearn import svm, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
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

# --------RANDOM FOREST---------------

rf = RandomForestClassifier(random_state=21,max_depth=200,n_estimators=200, n_jobs=-1)
rf_pred = rf.fit(X_train, y_train).predict(X_test)
print("RANDOM FOREST:")
print("Accuracy: ", accuracy_score(y_test, rf_pred))
print("F1: ", f1_score(y_test, rf_pred, average='weighted'))

# --------GAUSSIAN NAIVE BAYES---------------
gnb = GaussianNB()
gnb_pred = gnb.fit(X_train, y_train).predict(X_test)
print("GAUSSIAN NAIVE BAYES:")
print("Accuracy: ", accuracy_score(y_test, gnb_pred))
print("F1: ", f1_score(y_test, gnb_pred, average='weighted'))

# --------SUPPORT VECTOR MACHINE---------------
# clf = svm.SVC()
# svm_pred = clf.fit(X_train, y_train).predict(X_test)
# print("SUPPORT VECTOR MACHINE:")
# print("Accuracy: ", accuracy_score(y_test, svm_pred))
# print("F1: ", f1_score(y_test, svm_pred, average='weighted'))

# --------DECISION TREE---------------
dt = tree.DecisionTreeClassifier()
dt_pred = dt.fit(X_train, y_train).predict(X_test)
print("DECISION TREE:")
print("Accuracy: ", accuracy_score(y_test, dt_pred))
print("F1: ", f1_score(y_test, dt_pred, average='weighted'))

# PLOTAR A ARVORE DE DECISAO?
# tree.plot_tree(dt_pred)

# --------K-NEAREST NEIGHBORS---------------
knn = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
knn_pred = knn.fit(X_train, y_train).predict(X_test)
print("K-NEAREST NEIGHBORS:")
print("Accuracy: ", accuracy_score(y_test, knn_pred))
print("F1: ", f1_score(y_test, knn_pred, average='weighted'))

# Print Confusion Matrixes

randomForestCM = ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, cmap=plt.cm.Blues, normalize='true', colorbar=False)
randomForestCM.ax_.set_title('Random Forest')
print(randomForestCM)
plt.show()

naiveBayesCM = ConfusionMatrixDisplay.from_predictions(y_test, gnb_pred, cmap=plt.cm.Blues, normalize='true', colorbar=False)
naiveBayesCM.ax_.set_title('Naive Bayes (Gaussian)')
print(naiveBayesCM)
plt.show()

decisionTreeCM = ConfusionMatrixDisplay.from_predictions(y_test, dt_pred, cmap=plt.cm.Blues, normalize='true', colorbar=False)
decisionTreeCM.ax_.set_title('Decision Tree')
print(decisionTreeCM)
plt.show()

knnCM = ConfusionMatrixDisplay.from_predictions(y_test, knn_pred, cmap=plt.cm.Blues, normalize='true', colorbar=False)
knnCM.ax_.set_title('K-Nearest Neighbors')
print(knnCM)
plt.show()