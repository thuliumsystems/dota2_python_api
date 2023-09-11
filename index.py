from flask import Flask, jsonify, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle

app = Flask(__name__)


@app.route("/predict/", methods=["POST"])
def result():
    data = request.get_json()

    all = result_all(data["teams_avg_all"])
    as_kda = result_as_kda(data["teams_avg_as_kda"])
    de_as_kda_gpm = result_de_as_kda_gpm(data["teams_avg_de_as_kda_gpm"])

    return jsonify({"all": all, "as_kda": as_kda, "de_as_kda_gpm": de_as_kda_gpm})


def result_all(teams_avg_all):
    rad = 0
    dir = 0

    rf = pickle.load(open("saved/all/rf.sav", "rb"))
    rf = rf.predict_proba([teams_avg_all])
    rad += rf[0][1]
    dir += rf[0][0]

    dt = pickle.load(open("saved/all/dt.sav", "rb"))
    dt = dt.predict_proba([teams_avg_all])
    rad += dt[0][1]
    dir += dt[0][0]

    lr = pickle.load(open("saved/all/lr.sav", "rb"))
    lr = lr.predict_proba([teams_avg_all])
    rad += lr[0][1]
    dir += lr[0][0]

    gb = pickle.load(open("saved/all/gb.sav", "rb"))
    gb = gb.predict_proba([teams_avg_all])
    rad += gb[0][1]
    dir += gb[0][0]

    gnb = pickle.load(open("saved/all/gnb.sav", "rb"))
    gnb = gnb.predict_proba([teams_avg_all])
    rad += gnb[0][1]
    dir += gnb[0][0]

    knn = pickle.load(open("saved/all/knn.sav", "rb"))
    knn = knn.predict_proba([teams_avg_all])
    rad += knn[0][1]
    dir += knn[0][0]

    ada = pickle.load(open("saved/all/ada.sav", "rb"))
    ada = ada.predict_proba([teams_avg_all])
    rad += ada[0][1]
    dir += ada[0][0]

    mpl = pickle.load(open("saved/all/mpl.sav", "rb"))
    mpl = mpl.predict_proba([teams_avg_all])
    rad += mpl[0][1]
    dir += mpl[0][0]

    svc = pickle.load(open("saved/all/svc.sav", "rb"))
    svc = svc.predict_proba([teams_avg_all])
    rad += svc[0][1]
    dir += svc[0][0]

    qda = pickle.load(open("saved/all/qda.sav", "rb"))
    qda = qda.predict_proba([teams_avg_all])
    rad += qda[0][1]
    dir += qda[0][0]

    return [
        round(rad / 10, 2),
        round(dir / 10, 2),
        [round(rf[0][1], 2), round(rf[0][0], 2)],
        [round(dt[0][1], 2), round(dt[0][0], 2)],
        [round(lr[0][1], 2), round(lr[0][0], 2)],
        [round(gb[0][1], 2), round(gb[0][0], 2)],
        [round(gnb[0][1], 2), round(gnb[0][0], 2)],
        [round(knn[0][1], 2), round(knn[0][0], 2)],
        [round(ada[0][1], 2), round(ada[0][0], 2)],
        [round(mpl[0][1], 2), round(mpl[0][0], 2)],
        [round(svc[0][1], 2), round(svc[0][0], 2)],
        [round(qda[0][1], 2), round(qda[0][0], 2)],
    ]


def result_as_kda(teams_avg_as_kda):
    rad = 0
    dir = 0

    rf = pickle.load(open("saved/as_kda/rf.sav", "rb"))
    rf = rf.predict_proba([teams_avg_as_kda])
    rad += rf[0][1]
    dir += rf[0][0]

    dt = pickle.load(open("saved/as_kda/dt.sav", "rb"))
    dt = dt.predict_proba([teams_avg_as_kda])
    rad += dt[0][1]
    dir += dt[0][0]

    lr = pickle.load(open("saved/as_kda/lr.sav", "rb"))
    lr = lr.predict_proba([teams_avg_as_kda])
    rad += lr[0][1]
    dir += lr[0][0]

    gb = pickle.load(open("saved/as_kda/gb.sav", "rb"))
    gb = gb.predict_proba([teams_avg_as_kda])
    rad += gb[0][1]
    dir += gb[0][0]

    gnb = pickle.load(open("saved/as_kda/gnb.sav", "rb"))
    gnb = gnb.predict_proba([teams_avg_as_kda])
    rad += gnb[0][1]
    dir += gnb[0][0]

    knn = pickle.load(open("saved/as_kda/knn.sav", "rb"))
    knn = knn.predict_proba([teams_avg_as_kda])
    rad += knn[0][1]
    dir += knn[0][0]

    ada = pickle.load(open("saved/as_kda/ada.sav", "rb"))
    ada = ada.predict_proba([teams_avg_as_kda])
    rad += ada[0][1]
    dir += ada[0][0]

    mpl = pickle.load(open("saved/as_kda/mpl.sav", "rb"))
    mpl = mpl.predict_proba([teams_avg_as_kda])
    rad += mpl[0][1]
    dir += mpl[0][0]

    svc = pickle.load(open("saved/as_kda/svc.sav", "rb"))
    svc = svc.predict_proba([teams_avg_as_kda])
    rad += svc[0][1]
    dir += svc[0][0]

    qda = pickle.load(open("saved/as_kda/qda.sav", "rb"))
    qda = qda.predict_proba([teams_avg_as_kda])
    rad += qda[0][1]
    dir += qda[0][0]

    return [
        round(rad / 10, 2),
        round(dir / 10, 2),
        [round(rf[0][1], 2), round(rf[0][0], 2)],
        [round(dt[0][1], 2), round(dt[0][0], 2)],
        [round(lr[0][1], 2), round(lr[0][0], 2)],
        [round(gb[0][1], 2), round(gb[0][0], 2)],
        [round(gnb[0][1], 2), round(gnb[0][0], 2)],
        [round(knn[0][1], 2), round(knn[0][0], 2)],
        [round(ada[0][1], 2), round(ada[0][0], 2)],
        [round(mpl[0][1], 2), round(mpl[0][0], 2)],
        [round(svc[0][1], 2), round(svc[0][0], 2)],
        [round(qda[0][1], 2), round(qda[0][0], 2)],
    ]


def result_de_as_kda_gpm(teams_avg_de_as_kda_gpm):
    rad = 0
    dir = 0

    rf = pickle.load(open("saved/de_as_kda_gpm/rf.sav", "rb"))
    rf = rf.predict_proba([teams_avg_de_as_kda_gpm])
    rad += rf[0][1]
    dir += rf[0][0]

    dt = pickle.load(open("saved/de_as_kda_gpm/dt.sav", "rb"))
    dt = dt.predict_proba([teams_avg_de_as_kda_gpm])
    rad += dt[0][1]
    dir += dt[0][0]

    lr = pickle.load(open("saved/de_as_kda_gpm/lr.sav", "rb"))
    lr = lr.predict_proba([teams_avg_de_as_kda_gpm])
    rad += lr[0][1]
    dir += lr[0][0]

    gb = pickle.load(open("saved/de_as_kda_gpm/gb.sav", "rb"))
    gb = gb.predict_proba([teams_avg_de_as_kda_gpm])
    rad += gb[0][1]
    dir += gb[0][0]

    gnb = pickle.load(open("saved/de_as_kda_gpm/gnb.sav", "rb"))
    gnb = gnb.predict_proba([teams_avg_de_as_kda_gpm])
    rad += gnb[0][1]
    dir += gnb[0][0]

    knn = pickle.load(open("saved/de_as_kda_gpm/knn.sav", "rb"))
    knn = knn.predict_proba([teams_avg_de_as_kda_gpm])
    rad += knn[0][1]
    dir += knn[0][0]

    ada = pickle.load(open("saved/de_as_kda_gpm/ada.sav", "rb"))
    ada = ada.predict_proba([teams_avg_de_as_kda_gpm])
    rad += ada[0][1]
    dir += ada[0][0]

    mpl = pickle.load(open("saved/de_as_kda_gpm/mpl.sav", "rb"))
    mpl = mpl.predict_proba([teams_avg_de_as_kda_gpm])
    rad += mpl[0][1]
    dir += mpl[0][0]

    svc = pickle.load(open("saved/de_as_kda_gpm/svc.sav", "rb"))
    svc = svc.predict_proba([teams_avg_de_as_kda_gpm])
    rad += svc[0][1]
    dir += svc[0][0]

    qda = pickle.load(open("saved/de_as_kda_gpm/qda.sav", "rb"))
    qda = qda.predict_proba([teams_avg_de_as_kda_gpm])
    rad += qda[0][1]
    dir += qda[0][0]

    return [
        round(rad / 10, 2),
        round(dir / 10, 2),
        [round(rf[0][1], 2), round(rf[0][0], 2)],
        [round(dt[0][1], 2), round(dt[0][0], 2)],
        [round(lr[0][1], 2), round(lr[0][0], 2)],
        [round(gb[0][1], 2), round(gb[0][0], 2)],
        [round(gnb[0][1], 2), round(gnb[0][0], 2)],
        [round(knn[0][1], 2), round(knn[0][0], 2)],
        [round(ada[0][1], 2), round(ada[0][0], 2)],
        [round(mpl[0][1], 2), round(mpl[0][0], 2)],
        [round(svc[0][1], 2), round(svc[0][0], 2)],
        [round(qda[0][1], 2), round(qda[0][0], 2)],
    ]


def train_rf_model(X_train, y_train):
    model = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=200)
    model.fit(X_train, y_train)
    return model


def train_dt_model(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=4, min_samples_split=10
    )
    model.fit(X_train, y_train)
    return model


def train_ada_model(X_train, y_train):
    model = AdaBoostClassifier(learning_rate=0.1, n_estimators=200)
    model.fit(X_train, y_train)
    return model


def train_lr_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_gb_model(X_train, y_train):
    model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=100)
    model.fit(X_train, y_train)
    return model


def train_gnb_model(X_train, y_train):
    model = GaussianNB(var_smoothing=1e-09)
    model.fit(X_train, y_train)
    return model


def train_knn_model(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train, y_train)
    return model


def train_mpl_model(X_train, y_train):
    model = MLPClassifier()
    model.fit(X_train, y_train)
    return model


def train_svc_model(X_train, y_train):
    model = SVC(C=1, gamma=0.001, kernel="rbf", probability=True)
    model.fit(X_train, y_train)
    return model


def train_qda_model(X_train, y_train):
    model = QuadraticDiscriminantAnalysis(reg_param=0.1)
    model.fit(X_train, y_train)
    return model
