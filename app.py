from flask import Flask, jsonify, request
import pickle
import numpy as np

app = Flask(__name__)


@app.route("/predict/", methods=["POST"])
def result():
    data = request.get_json()

    kda = result_kda(data["teams_avg_kda"])
    as_kda = result_as_kda(data["teams_avg_as_kda"])
    de_as_kda_gpm = result_de_as_kda_gpm(data["teams_avg_de_as_kda_gpm"])
    all = result_all(data["teams_avg_all"])

    return jsonify(
        {"kda": kda, "as_kda": as_kda, "de_as_kda_gpm": de_as_kda_gpm, "all": all}
    )


def result_kda(teams_avg_kda):
    rad = 0
    dir = 0

    scaler = pickle.load(open("saved/kda/scaler.pkl", "rb"))
    scaler = scaler.transform([teams_avg_kda])

    rf = pickle.load(open("saved/kda/rf.pkl", "rb"))
    rf = rf.predict_proba(list(scaler))
    rad += rf[0][1]
    dir += rf[0][0]

    dt = pickle.load(open("saved/kda/dt.pkl", "rb"))
    dt = dt.predict_proba(list(scaler))
    rad += dt[0][1]
    dir += dt[0][0]

    lr = pickle.load(open("saved/kda/lr.pkl", "rb"))
    lr = lr.predict_proba(list(scaler))
    rad += lr[0][1]
    dir += lr[0][0]

    gb = pickle.load(open("saved/kda/gb.pkl", "rb"))
    gb = gb.predict_proba(list(scaler))
    rad += gb[0][1]
    dir += gb[0][0]

    gnb = pickle.load(open("saved/kda/gnb.pkl", "rb"))
    gnb = gnb.predict_proba(list(scaler))
    rad += gnb[0][1]
    dir += gnb[0][0]

    knn = pickle.load(open("saved/kda/knn.pkl", "rb"))
    knn = knn.predict_proba(list(scaler))
    rad += knn[0][1]
    dir += knn[0][0]

    ada = pickle.load(open("saved/kda/ada.pkl", "rb"))
    ada = ada.predict_proba(list(scaler))
    rad += ada[0][1]
    dir += ada[0][0]

    mpl = pickle.load(open("saved/kda/mpl.pkl", "rb"))
    mpl = mpl.predict_proba(list(scaler))
    rad += mpl[0][1]
    dir += mpl[0][0]

    svc = pickle.load(open("saved/kda/svc.pkl", "rb"))
    svc = svc.predict_proba(list(scaler))
    rad += svc[0][1]
    dir += svc[0][0]

    qda = pickle.load(open("saved/kda/qda.pkl", "rb"))
    qda = qda.predict_proba(list(scaler))
    rad += qda[0][1]
    dir += qda[0][0]

    vc = pickle.load(open("saved/kda/vc.pkl", "rb"))
    vc = vc.predict_proba(list(scaler))

    return [
        [round(rad / 10, 2), round(dir / 10, 2)],
        [round(vc[0][1], 2), round(vc[0][0], 2)],
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
    
    scaler = pickle.load(open("saved/as_kda/scaler.pkl", "rb"))
    scaler = scaler.transform([teams_avg_as_kda])

    rf = pickle.load(open("saved/as_kda/rf.pkl", "rb"))
    rf = rf.predict_proba(list(scaler))
    rad += rf[0][1]
    dir += rf[0][0]

    dt = pickle.load(open("saved/as_kda/dt.pkl", "rb"))
    dt = dt.predict_proba(list(scaler))
    rad += dt[0][1]
    dir += dt[0][0]

    lr = pickle.load(open("saved/as_kda/lr.pkl", "rb"))
    lr = lr.predict_proba(list(scaler))
    rad += lr[0][1]
    dir += lr[0][0]

    gb = pickle.load(open("saved/as_kda/gb.pkl", "rb"))
    gb = gb.predict_proba(list(scaler))
    rad += gb[0][1]
    dir += gb[0][0]

    gnb = pickle.load(open("saved/as_kda/gnb.pkl", "rb"))
    gnb = gnb.predict_proba(list(scaler))
    rad += gnb[0][1]
    dir += gnb[0][0]

    knn = pickle.load(open("saved/as_kda/knn.pkl", "rb"))
    knn = knn.predict_proba(list(scaler))
    rad += knn[0][1]
    dir += knn[0][0]

    ada = pickle.load(open("saved/as_kda/ada.pkl", "rb"))
    ada = ada.predict_proba(list(scaler))
    rad += ada[0][1]
    dir += ada[0][0]

    mpl = pickle.load(open("saved/as_kda/mpl.pkl", "rb"))
    mpl = mpl.predict_proba(list(scaler))
    rad += mpl[0][1]
    dir += mpl[0][0]

    svc = pickle.load(open("saved/as_kda/svc.pkl", "rb"))
    svc = svc.predict_proba(list(scaler))
    rad += svc[0][1]
    dir += svc[0][0]

    qda = pickle.load(open("saved/as_kda/qda.pkl", "rb"))
    qda = qda.predict_proba(list(scaler))
    rad += qda[0][1]
    dir += qda[0][0]

    vc = pickle.load(open("saved/as_kda/vc.pkl", "rb"))
    vc = vc.predict_proba(list(scaler))

    return [
        [round(rad / 10, 2), round(dir / 10, 2)],
        [round(vc[0][1], 2), round(vc[0][0], 2)],
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
    
    scaler = pickle.load(open("saved/de_as_kda_gpm/scaler.pkl", "rb"))
    scaler = scaler.transform([teams_avg_de_as_kda_gpm])

    rf = pickle.load(open("saved/de_as_kda_gpm/rf.pkl", "rb"))
    rf = rf.predict_proba(list(scaler))
    rad += rf[0][1]
    dir += rf[0][0]

    dt = pickle.load(open("saved/de_as_kda_gpm/dt.pkl", "rb"))
    dt = dt.predict_proba(list(scaler))
    rad += dt[0][1]
    dir += dt[0][0]

    lr = pickle.load(open("saved/de_as_kda_gpm/lr.pkl", "rb"))
    lr = lr.predict_proba(list(scaler))
    rad += lr[0][1]
    dir += lr[0][0]

    gb = pickle.load(open("saved/de_as_kda_gpm/gb.pkl", "rb"))
    gb = gb.predict_proba(list(scaler))
    rad += gb[0][1]
    dir += gb[0][0]

    gnb = pickle.load(open("saved/de_as_kda_gpm/gnb.pkl", "rb"))
    gnb = gnb.predict_proba(list(scaler))
    rad += gnb[0][1]
    dir += gnb[0][0]

    knn = pickle.load(open("saved/de_as_kda_gpm/knn.pkl", "rb"))
    knn = knn.predict_proba(list(scaler))
    rad += knn[0][1]
    dir += knn[0][0]

    ada = pickle.load(open("saved/de_as_kda_gpm/ada.pkl", "rb"))
    ada = ada.predict_proba(list(scaler))
    rad += ada[0][1]
    dir += ada[0][0]

    mpl = pickle.load(open("saved/de_as_kda_gpm/mpl.pkl", "rb"))
    mpl = mpl.predict_proba(list(scaler))
    rad += mpl[0][1]
    dir += mpl[0][0]

    svc = pickle.load(open("saved/de_as_kda_gpm/svc.pkl", "rb"))
    svc = svc.predict_proba(list(scaler))
    rad += svc[0][1]
    dir += svc[0][0]

    qda = pickle.load(open("saved/de_as_kda_gpm/qda.pkl", "rb"))
    qda = qda.predict_proba(list(scaler))
    rad += qda[0][1]
    dir += qda[0][0]

    vc = pickle.load(open("saved/de_as_kda_gpm/vc.pkl", "rb"))
    vc = vc.predict_proba(list(scaler))

    return [
        [round(rad / 10, 2), round(dir / 10, 2)],
        [round(vc[0][1], 2), round(vc[0][0], 2)],
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


def result_all(teams_avg_all):
    rad = 0
    dir = 0
    
    scaler = pickle.load(open("saved/all/scaler.pkl", "rb"))
    scaler = scaler.transform([teams_avg_all])

    rf = pickle.load(open("saved/all/rf.pkl", "rb"))
    rf = rf.predict_proba(list(scaler))
    rad += rf[0][1]
    dir += rf[0][0]

    dt = pickle.load(open("saved/all/dt.pkl", "rb"))
    dt = dt.predict_proba(list(scaler))
    rad += dt[0][1]
    dir += dt[0][0]

    lr = pickle.load(open("saved/all/lr.pkl", "rb"))
    lr = lr.predict_proba(list(scaler))
    rad += lr[0][1]
    dir += lr[0][0]

    gb = pickle.load(open("saved/all/gb.pkl", "rb"))
    gb = gb.predict_proba(list(scaler))
    rad += gb[0][1]
    dir += gb[0][0]

    gnb = pickle.load(open("saved/all/gnb.pkl", "rb"))
    gnb = gnb.predict_proba(list(scaler))
    rad += gnb[0][1]
    dir += gnb[0][0]

    knn = pickle.load(open("saved/all/knn.pkl", "rb"))
    knn = knn.predict_proba(list(scaler))
    rad += knn[0][1]
    dir += knn[0][0]

    ada = pickle.load(open("saved/all/ada.pkl", "rb"))
    ada = ada.predict_proba(list(scaler))
    rad += ada[0][1]
    dir += ada[0][0]

    mpl = pickle.load(open("saved/all/mpl.pkl", "rb"))
    mpl = mpl.predict_proba(list(scaler))
    rad += mpl[0][1]
    dir += mpl[0][0]

    svc = pickle.load(open("saved/all/svc.pkl", "rb"))
    svc = svc.predict_proba(list(scaler))
    rad += svc[0][1]
    dir += svc[0][0]

    qda = pickle.load(open("saved/all/qda.pkl", "rb"))
    qda = qda.predict_proba(list(scaler))
    rad += qda[0][1]
    dir += qda[0][0]

    vc = pickle.load(open("saved/all/vc.pkl", "rb"))
    vc = vc.predict_proba(list(scaler))

    return [
        [round(rad / 10, 2), round(dir / 10, 2)],
        [round(vc[0][1], 2), round(vc[0][0], 2)],
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
