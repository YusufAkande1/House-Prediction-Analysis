import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("House_Prediction_Clean.csv")

df['PriceCategory'] = df['HousePrice'].apply(lambda x: 1 if x > df['HousePrice'].mean() else 0)
X = df.drop(['HousePrice', 'PriceCategory'], axis=1)
y = df['PriceCategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X = pd.get_dummies(X, drop_first=True)

#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

#EVALUATE MODELS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Function to evaluate
def evaluate(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("------------------------")
#Run evaluation
print("Logistic Regression")
evaluate(y_test, y_pred_lr)

print("Decision Tree")
evaluate(y_test, y_pred_dt)

print("Random Forest")
evaluate(y_test, y_pred_rf)

#HYPERPARAMETER TUNING (GRID SEARCH)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

#Predict again
y_pred_best = best_model.predict(X_test)
evaluate(y_test, y_pred_best)

