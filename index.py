from flask import Flask, jsonify, request
import requests
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model

app = Flask(__name__)


@app.route("/predict/", methods=["POST"])
def result():

    players = request.json

    url = "https://bet-dota2.fly.dev/api/training"
    data = requests.get(url).json()
    training_data = data["all"]

    # return jsonify(players)

    X = []

    for t in training_data:
        X.append(t[:-1])

    y = [last for *_, last in training_data]

    # Rad P1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["rad"]["p1"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p1"])
    rad_dt_p1 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["rad"]["p1"])
    rad_lr_p1 = logr.score(players["rad"]["p1"], y_test)

    # Rad P2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["rad"]["p2"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p2"])
    rad_dt_p2 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["rad"]["p2"])
    rad_lr_p2 = logr.score(players["rad"]["p2"], y_test)

    # Rad P3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["rad"]["p3"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p3"])
    rad_dt_p3 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["rad"]["p3"])
    rad_lr_p3 = logr.score(players["rad"]["p3"], y_test)

    # Rad P4
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["rad"]["p4"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p4"])
    rad_dt_p4 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["rad"]["p4"])
    rad_lr_p4 = logr.score(players["rad"]["p4"], y_test)

    # Rad P5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["rad"]["p5"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["rad"]["p5"])
    rad_dt_p5 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["rad"]["p5"])
    rad_lr_p5 = logr.score(players["rad"]["p5"], y_test)

    # Dir P1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["dir"]["p1"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p1"])
    dir_dt_p1 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["dir"]["p1"])
    dir_lr_p1 = logr.score(players["dir"]["p1"], y_test)

    # # Dir P2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["dir"]["p2"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p2"])
    dir_dt_p2 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["dir"]["p2"])
    dir_lr_p2 = logr.score(players["dir"]["p2"], y_test)

    # Dir P3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["dir"]["p3"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p3"])
    dir_dt_p3 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["dir"]["p3"])
    dir_lr_p3 = logr.score(players["dir"]["p3"], y_test)

    # Dir P4
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["dir"]["p4"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p4"])
    dir_dt_p4 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["dir"]["p4"])
    dir_lr_p4 = logr.score(players["dir"]["p4"], y_test)

    # Dir P5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(players["dir"]["p5"]), random_state=0
    )
    dtree = DecisionTreeClassifier(max_depth=3)
    dtree = dtree.fit(X_train, y_train)
    y_pred = dtree.predict(players["dir"]["p5"])
    dir_dt_p5 = metrics.accuracy_score(y_test, y_pred)

    logr = linear_model.LogisticRegression(solver="liblinear", random_state=0).fit(
        X_train, y_train
    )
    y_pred = logr.predict(players["dir"]["p5"])
    dir_lr_p5 = logr.score(players["dir"]["p5"], y_test)

    return jsonify(
        {
            "dt": {
                "rad": [
                    float("%.*f" % (2, rad_dt_p1)),
                    float("%.*f" % (2, rad_dt_p2)),
                    float("%.*f" % (2, rad_dt_p3)),
                    float("%.*f" % (2, rad_dt_p4)),
                    float("%.*f" % (2, rad_dt_p5)),
                ],
                "dir": [
                    float("%.*f" % (2, dir_dt_p1)),
                    float("%.*f" % (2, dir_dt_p2)),
                    float("%.*f" % (2, dir_dt_p3)),
                    float("%.*f" % (2, dir_dt_p4)),
                    float("%.*f" % (2, dir_dt_p5)),
                ],
            },
            "lr": {
                "rad": [
                    float("%.*f" % (2, rad_lr_p1)),
                    float("%.*f" % (2, rad_lr_p2)),
                    float("%.*f" % (2, rad_lr_p3)),
                    float("%.*f" % (2, rad_lr_p4)),
                    float("%.*f" % (2, rad_lr_p5)),
                ],
                "dir": [
                    float("%.*f" % (2, dir_lr_p1)),
                    float("%.*f" % (2, dir_lr_p2)),
                    float("%.*f" % (2, dir_lr_p3)),
                    float("%.*f" % (2, dir_lr_p4)),
                    float("%.*f" % (2, dir_lr_p5)),
                ],
            },
        }
    )
