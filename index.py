from flask import Flask, jsonify, request
import requests
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


def split_data(X, y, data):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=len(data), random_state=0, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(data)
    return X_train, X_test, y_train, y_test


def train_dt_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def train_lr_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_rf_model(X_train, y_train):
    model = RandomForestClassifier(max_features=18)
    model.fit(X_train, y_train)
    return model


def train_gb_model(X_train, y_train):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model


@app.route("/predict/", methods=["POST"])
def result():

    players = request.json

    # url = "http://localhost:3000/api/training"
    url = "https://bet-dota2.fly.dev/api/training"
    data = requests.get(url).json()
    training_data = data["all"]

    # return jsonify(players)

    X = []

    for t in training_data:
        X.append(t[:-1])

    y = [last for *_, last in training_data]

    # Rad P1
    X_train, X_test, y_train, y_test = split_data(X, y, players["rad"]["p1"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    rad_dt_p1 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    rad_lr_p1 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    rad_rf_p1 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    rad_gb_p1 = accuracy_score(y_test, y_pred)
    # print(classification_report(y_test, gb.predict(X_test)))

    # Rad P2
    X_train, X_test, y_train, y_test = split_data(X, y, players["rad"]["p2"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    rad_dt_p2 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    rad_lr_p2 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    rad_rf_p2 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    rad_gb_p2 = accuracy_score(y_test, y_pred)
    
    # Rad P3
    X_train, X_test, y_train, y_test = split_data(X, y, players["rad"]["p3"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    rad_dt_p3 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    rad_lr_p3 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    rad_rf_p3 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    rad_gb_p3 = accuracy_score(y_test, y_pred)
    
    # Rad P4
    X_train, X_test, y_train, y_test = split_data(X, y, players["rad"]["p4"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    rad_dt_p4 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    rad_lr_p4 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    rad_rf_p4 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    rad_gb_p4 = accuracy_score(y_test, y_pred)
    
    # Rad P5
    X_train, X_test, y_train, y_test = split_data(X, y, players["rad"]["p5"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    rad_dt_p5 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    rad_lr_p5 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    rad_rf_p5 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    rad_gb_p5 = accuracy_score(y_test, y_pred)
    
    # Dir P1
    X_train, X_test, y_train, y_test = split_data(X, y, players["dir"]["p1"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    dir_dt_p1 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    dir_lr_p1 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    dir_rf_p1 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    dir_gb_p1 = accuracy_score(y_test, y_pred)

    # Dir P2
    X_train, X_test, y_train, y_test = split_data(X, y, players["dir"]["p2"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    dir_dt_p2 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    dir_lr_p2 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    dir_rf_p2 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    dir_gb_p2 = accuracy_score(y_test, y_pred)
    
    # Dir P3
    X_train, X_test, y_train, y_test = split_data(X, y, players["dir"]["p3"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    dir_dt_p3 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    dir_lr_p3 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    dir_rf_p3 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    dir_gb_p3 = accuracy_score(y_test, y_pred)
    
    # Dir P4
    X_train, X_test, y_train, y_test = split_data(X, y, players["dir"]["p4"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    dir_dt_p4 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    dir_lr_p4 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    dir_rf_p4 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    dir_gb_p4 = accuracy_score(y_test, y_pred)
    
    # Dir P5
    X_train, X_test, y_train, y_test = split_data(X, y, players["dir"]["p5"])
    dt = train_dt_model(X_train, y_train)
    y_pred = dt.predict(X_test)
    dir_dt_p5 = accuracy_score(y_test, y_pred)

    lr = train_lr_model(X_train, y_train)
    y_pred = lr.predict(X_test)
    dir_lr_p5 = accuracy_score(y_test, y_pred)

    rf = train_rf_model(X_train, y_train)
    y_pred = rf.predict(X_test)
    dir_rf_p5 = accuracy_score(y_test, y_pred)

    gb = train_gb_model(X_train, y_train)
    y_pred = gb.predict(X_test)
    dir_gb_p5 = accuracy_score(y_test, y_pred)

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
            "rf": {
                "rad": [
                    float("%.*f" % (2, rad_rf_p1)),
                    float("%.*f" % (2, rad_rf_p2)),
                    float("%.*f" % (2, rad_rf_p3)),
                    float("%.*f" % (2, rad_rf_p4)),
                    float("%.*f" % (2, rad_rf_p5)),
                ],
                "dir": [
                    float("%.*f" % (2, dir_rf_p1)),
                    float("%.*f" % (2, dir_rf_p2)),
                    float("%.*f" % (2, dir_rf_p3)),
                    float("%.*f" % (2, dir_rf_p4)),
                    float("%.*f" % (2, dir_rf_p5)),
                ],
            },
            "gb": {
                "rad": [
                    float("%.*f" % (2, rad_gb_p1)),
                    float("%.*f" % (2, rad_gb_p2)),
                    float("%.*f" % (2, rad_gb_p3)),
                    float("%.*f" % (2, rad_gb_p4)),
                    float("%.*f" % (2, rad_gb_p5)),
                ],
                "dir": [
                    float("%.*f" % (2, dir_gb_p1)),
                    float("%.*f" % (2, dir_gb_p2)),
                    float("%.*f" % (2, dir_gb_p3)),
                    float("%.*f" % (2, dir_gb_p4)),
                    float("%.*f" % (2, dir_gb_p5)),
                ],
            },
        }
    )
