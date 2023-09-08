# Tyler Tobin
# Football prediction

#---------- Imports ----------

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import sklearn.linear_model
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
import scipy
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn import gaussian_process
from sklearn import naive_bayes
from sklearn import svm
from sklearn import tree
from sklearn import discriminant_analysis

#-------- Dataset loading ---------
dataset = pd.read_csv("prem22_23.csv")

features = ['home_team', 'away_team', 'round', 'day', 'referee']

X_raw = dataset[features]
#y_raw = dataset['shots_on_target']
#y_raw = dataset['cards']

y_raw = dataset['result']
y_raw = y_raw.map({'a': 0, 'd': 1, 'h': 2})

#------------- Pre-processing ------------

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.15, shuffle=True, random_state=0)

X_train_num = X_train_raw.select_dtypes(include=np.number)
X_train_cat = X_train_raw.select_dtypes(exclude=np.number)

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

numeric_imputer.fit(X_train_num)
categorical_imputer.fit(X_train_cat)

np.set_printoptions(threshold=np.inf)

X_train_num_imp = numeric_imputer.transform(X_train_num)
X_train_cat_imp = categorical_imputer.transform(X_train_cat)

X_test_num = X_test_raw.select_dtypes(include=np.number)
X_test_cat = X_test_raw.select_dtypes(exclude=np.number)

X_test_num_imp = numeric_imputer.transform(X_test_num)
X_test_cat_imp = categorical_imputer.transform(X_test_cat)

X_train_cat_home = X_train_cat_imp[:, 0].reshape(-1, 1)
X_test_cat_home = X_test_cat_imp[:, 0].reshape(-1, 1)

X_train_cat_away = X_train_cat_imp[:, 1].reshape(-1, 1)
X_test_cat_away = X_test_cat_imp[:, 1].reshape(-1, 1)

X_train_cat_day = X_train_cat_imp[:, 2].reshape(-1, 1)
X_test_cat_day = X_test_cat_imp[:, 2].reshape(-1, 1)

X_train_cat_ref = X_train_cat_imp[:, 3].reshape(-1, 1)
X_test_cat_ref = X_test_cat_imp[:, 3].reshape(-1, 1)

encoder1 = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[['crystal_palace', 'fulham', 'tottenham', 'newcastle', 'leeds', 'bournemouth', 'everton', 'leicester', 'man_utd', 'west_ham', 'aston_villa', 'man_city', 'southampton', 'wolves', 'arsenal', 'brighton', 'brentford', 'nottingham', 'chelsea', 'liverpool']])
encoder2 = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[['crystal_palace', 'fulham', 'tottenham', 'newcastle', 'leeds', 'bournemouth', 'everton', 'leicester', 'man_utd', 'west_ham', 'aston_villa', 'man_city', 'southampton', 'wolves', 'arsenal', 'brighton', 'brentford', 'nottingham', 'chelsea', 'liverpool']])
encoder3 = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']])
encoder4 = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[['anthony_taylor', 'andy_madley', 'andre_marriner', 'simon_hooper', 'robert_jones', 'peter_bankes', 'craig_pawson', 'jarred_gillett', 'paul_tierney', 'michael_oliver', 'david_coote', 'tony_harrington', 'john_brooks', 'darren_england', 'graham_scott', 'stuart_attwell', 'michael_salisbury', 'thomas_bramall', 'chris_kavanagh', 'robert_madley', 'darren_bond', 'tim_robinson']])

X_onehot_cat_train_home = encoder1.fit_transform(X_train_cat_home)
X_onehot_cat_train_away = encoder2.fit_transform(X_train_cat_away)
X_onehot_cat_train_day = encoder3.fit_transform(X_train_cat_day)
X_onehot_cat_train_ref = encoder4.fit_transform(X_train_cat_ref)

X_onehot_cat_test_home = encoder1.transform(X_test_cat_home)
X_onehot_cat_test_away = encoder2.transform(X_test_cat_away)
X_onehot_cat_test_day = encoder3.transform(X_test_cat_day)
X_onehot_cat_test_ref = encoder4.transform(X_test_cat_ref)

X_train_cat_imp = np.concatenate([X_onehot_cat_train_home, X_onehot_cat_train_away, X_onehot_cat_train_day, X_onehot_cat_train_ref], axis=1).astype(int)

X_test_cat_imp = np.concatenate([X_onehot_cat_test_home, X_onehot_cat_test_away, X_onehot_cat_test_day, X_onehot_cat_test_ref], axis=1).astype(int)

np.set_printoptions(precision=2, suppress=True)

X_train = np.concatenate([X_train_cat_imp, X_train_num_imp], axis=1)
X_test = np.concatenate([X_test_cat_imp, X_test_num_imp], axis=1)

base_clf = DecisionTreeClassifier(max_depth=2)

modelList = [ExtraTreesClassifier(n_estimators=100),
             LogisticRegression(random_state=0, max_iter=1000),
             MLPClassifier(solver='lbfgs', alpha=5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=1200),
             SGDClassifier(loss='log_loss', penalty="l2", max_iter=100),
             XGBClassifier(n_estimators=100, learning_rate=0.0001, max_depth=5),
             LGBMClassifier(num_leaves=31, learning_rate=0.001, n_estimators=100),
             GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=2),
             RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
             KNeighborsClassifier(n_neighbors=21),
             GaussianNB(priors=[0.26, 0.28, 0.46]),
             ensemble.AdaBoostClassifier(base_estimator=base_clf, n_estimators=100, learning_rate=0.1, algorithm='SAMME.R'),
             ensemble.BaggingClassifier(),
             ensemble.ExtraTreesClassifier(),
             ensemble.RandomForestClassifier(),
             ensemble.GradientBoostingClassifier(),
             gaussian_process.GaussianProcessClassifier(),
             naive_bayes.BernoulliNB(),
             naive_bayes.GaussianNB(),
             svm.SVC(probability=True),
             svm.NuSVC(probability=True),
             tree.DecisionTreeClassifier(),
             tree.ExtraTreeClassifier(),
             discriminant_analysis.LinearDiscriminantAnalysis(),
             discriminant_analysis.QuadraticDiscriminantAnalysis()

             ]
num = 0

highScore = 0

for model in modelList:
   print("------------------------------------------")
   num = num + 1
   print("model: ", num)
   model = model

   model.fit(X_train, y_train)

   y_pred_train = model.predict(X_train)
   y_pred = model.predict(X_test)

   score = model.score(X_test, y_test)
   if score > highScore:
       highScore = score
   print("score:", score)

   #----------- Prediction and evaluation ----------

   # get the start time
   st = time.time()
   test_prediction = [[ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  35]]

   prediction = model.predict(test_prediction)
   pred_probability = model.predict_proba(test_prediction)
   print(prediction[0])
   print("prediction confidence: ", pred_probability)
   prediction = model.predict(X_test)
   ps = precision_score(y_test, prediction, average='weighted', zero_division=0)
   recall = recall_score(y_test, prediction, average='macro', zero_division=0)
   et = time.time()
   elapsed_time = et - st

   print(elapsed_time, " - First Prediction took")
   print('precision: ',ps)
   print('recall: ',recall)
   # evaluate performance on testing data
   score = model.score(X_test, y_test)
   print(f'Test accuracy: {score}')

print()
print("Highest accuracy:", int((round(highScore, 2))*100), "%")
