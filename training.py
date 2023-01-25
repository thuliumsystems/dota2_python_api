# from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

url = "https://bet-dota2.fly.dev/api/training"
data = requests.get(url).json()

all = data["all"]

X = []
mean_decrease_in_impurity_score = []
# permutation_score = []
# linear_regression_score = []
# logistic_regression_score = []
dt_regressor_score = []
xgboost_score = []

for t in all:
    X.append(t[:-1])

y = [last for *_, last in all]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

feature_names = [
    "assists",
    "deaths",
    "gpm",
    "hero_dmg",
    "hero_healing",
    "xpm",
    "kda",
    "kills",
    "neutral_kills",
    "last_hits",
    "lvl",
    "net_worth",
    "tf_participation",
    "tower_dmge",
    "kpm",
    "observer_uses",
    "sentry_uses",
    "lane_efficiency",
]

forest = RandomForestClassifier(random_state=0)
forest.fit(X, y)

# Feature importance based on mean decrease in impurity
mean_decrease_in_impurity = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
forest_importances_impurity = pd.Series(mean_decrease_in_impurity, index=feature_names)

for i, v in enumerate(mean_decrease_in_impurity):
    mean_decrease_in_impurity_score.append(v)
# Feature importance based on mean decrease in impurity

# Feature importance based on feature permutation
# permutation = permutation_importance(
#     forest, X_test, y_test, n_repeats=10, random_state=0, n_jobs=2
# )
# forest_importances_permutation = pd.Series(
#     permutation.importances_mean, index=feature_names
# )

# for i, v in enumerate(permutation.importances_mean):
#     permutation_score.append(v)
# Feature importance based on feature permutation

# Linear Regression Feature Importance
# model = LinearRegression()
# model.fit(X, y)
# linear_regression = model.coef_

# for i, v in enumerate(linear_regression):
#     linear_regression_score.append(v)
# Linear Regression Feature Importance

# Logistic Regression Feature Importance
# model = LogisticRegression()
# model.fit(X, y)
# logistic_regression = model.coef_[0]

# for i, v in enumerate(logistic_regression):
#     logistic_regression_score.append(v)
# Logistic Regression Feature Importance

# Regression Feature Importance
model = DecisionTreeRegressor()
model.fit(X, y)
dt_regressor = model.feature_importances_

for i, v in enumerate(dt_regressor):
    dt_regressor_score.append(v)
# Regression Feature Importance

# XGBoost Feature Importance
model = XGBRegressor()
model.fit(X, y)
xgboost_importance = model.feature_importances_

for i, v in enumerate(xgboost_importance):
    xgboost_score.append(v)
# XGBoost Feature Importance

df = pd.DataFrame(
    {
        "features": feature_names,
        "impurity": np.array(mean_decrease_in_impurity_score),
        # "permutation": np.array(permutation_score),
        # "linear_regression": np.array(linear_regression_score),
        # "logistic_regression": np.array(logistic_regression_score),
        "dt_regressor": np.array(dt_regressor_score),
        "xgboost": np.array(xgboost_score),
    }
)

df.plot(
    x="features",
    y=["impurity", "dt_regressor", "xgboost"],
    kind="bar",
)

plt.show()
