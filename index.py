from flask import Flask, jsonify, request
import requests
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)


@app.route("/teams/", methods=["POST"])
def result():

    players = request.json

    url = "http://localhost:3000/api/training"
    data = requests.get(url).json()
    training_core = data["core"]
    training_supp4 = data["supp4"]
    training_supp5 = data["supp5"]
    training_all = data["all"]

    X_core = []
    X_supp4 = []
    X_supp5 = []
    X_all = []

    for t in training_core:
        X_core.append(t[:-1])

    for t in training_supp4:
        X_supp4.append(t[:-1])

    for t in training_supp5:
        X_supp5.append(t[:-1])
        
    for t in training_all:
        X_all.append(t[:-1])

    y_core = [last for *_, last in training_core]
    y_supp4 = [last for *_, last in training_supp4]
    y_supp5 = [last for *_, last in training_supp5]
    y_all = [last for *_, last in training_all]

    percent_rad_p1 = len(players["rad"]["p1"]) / len(training_core)
    percent_rad_p2 = len(players["rad"]["p2"]) / len(training_core)
    percent_rad_p3 = len(players["rad"]["p3"]) / len(training_core)
    percent_rad_p4 = len(players["rad"]["p4"]) / len(training_supp4)
    percent_rad_p5 = len(players["rad"]["p5"]) / len(training_supp5)

    percent_dir_p1 = len(players["dir"]["p1"]) / len(training_core)
    percent_dir_p2 = len(players["dir"]["p2"]) / len(training_core)
    percent_dir_p3 = len(players["dir"]["p3"]) / len(training_core)
    percent_dir_p4 = len(players["dir"]["p4"]) / len(training_supp4)
    percent_dir_p5 = len(players["dir"]["p5"]) / len(training_supp5)

    # Rad P1
    X_train, X_test, y_train, y_test = train_test_split(
        X_core, y_core, test_size=percent_rad_p1
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)
    rad_p1 = metrics.accuracy_score(y_test, y_pred)

    # Rad P2
    X_train, X_test, y_train, y_test = train_test_split(
        X_core, y_core, test_size=percent_rad_p2
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p2"])
    rad_p2 = metrics.accuracy_score(y_test, y_pred)

    # Rad P3
    X_train, X_test, y_train, y_test = train_test_split(
        X_core, y_core, test_size=percent_rad_p3
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p3"])
    rad_p3 = metrics.accuracy_score(y_test, y_pred)

    # Rad P4
    X_train, X_test, y_train, y_test = train_test_split(
        X_supp4, y_supp4, test_size=percent_rad_p4
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p4"])
    rad_p4 = metrics.accuracy_score(y_test, y_pred)

    # Rad P5
    X_train, X_test, y_train, y_test = train_test_split(
        X_supp5, y_supp5, test_size=percent_rad_p5
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p5"])
    rad_p5 = metrics.accuracy_score(y_test, y_pred)

    # Dir P1
    X_train, X_test, y_train, y_test = train_test_split(
        X_core, y_core, test_size=percent_dir_p1
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p1"])
    dir_p1 = metrics.accuracy_score(y_test, y_pred)

    # Dir P2
    X_train, X_test, y_train, y_test = train_test_split(
        X_core, y_core, test_size=percent_dir_p2
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p2"])
    dir_p2 = metrics.accuracy_score(y_test, y_pred)

    # Dir P3
    X_train, X_test, y_train, y_test = train_test_split(
        X_core, y_core, test_size=percent_dir_p3
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p3"])
    dir_p3 = metrics.accuracy_score(y_test, y_pred)

    # Dir P4
    X_train, X_test, y_train, y_test = train_test_split(
        X_supp4, y_supp4, test_size=percent_dir_p4
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p4"])
    dir_p4 = metrics.accuracy_score(y_test, y_pred)

    # Dir P5
    X_train, X_test, y_train, y_test = train_test_split(
        X_supp5, y_supp5, test_size=percent_dir_p5
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p5"])
    dir_p5 = metrics.accuracy_score(y_test, y_pred)

    return jsonify(
        {
            "dt": {
                "rad": {
                    "p1": rad_p1,
                    "p2": rad_p2,
                    "p3": rad_p3,
                    "p4": rad_p4,
                    "p5": rad_p5,
                },
                "dir": {
                    "p1": dir_p1,
                    "p2": dir_p2,
                    "p3": dir_p3,
                    "p4": dir_p4,
                    "p5": dir_p5,
                },
            },
            "rf": {
                "rad": {
                    "p1": ""
                }
            }
        }
    )
