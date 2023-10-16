from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import json

with open("data/kda.json") as user_file_kda:
    file_contents_kda = user_file_kda.read()
data_kda = json.loads(file_contents_kda)

with open("data/as_kda.json") as user_file_as_kda:
    file_contents_as_kda = user_file_as_kda.read()
data_as_kda = json.loads(file_contents_as_kda)

with open("data/de_as_kda_gpm.json") as user_file_de_as_kda_gpm:
    file_contents_de_as_kda_gpm = user_file_de_as_kda_gpm.read()
data_de_as_kda_gpm = json.loads(file_contents_de_as_kda_gpm)

with open("data/all.json") as user_file_all:
    file_contents_all = user_file_all.read()
data_all = json.loads(file_contents_all)


def training_kda():
    X_train, X_test, y_train, y_test = split_data(data_kda, "kda")

    rf = train_rf_model(X_train, y_train)
    pickle.dump(rf, open("saved/kda/rf.pkl", "wb"))

    dt = train_dt_model(X_train, y_train)
    pickle.dump(dt, open("saved/kda/dt.pkl", "wb"))

    lr = train_lr_model(X_train, y_train)
    pickle.dump(lr, open("saved/kda/lr.pkl", "wb"))

    gb = train_gb_model(X_train, y_train)
    pickle.dump(gb, open("saved/kda/gb.pkl", "wb"))

    gnb = train_gnb_model(X_train, y_train)
    pickle.dump(gnb, open("saved/kda/gnb.pkl", "wb"))

    knn = train_knn_model(X_train, y_train)
    pickle.dump(knn, open("saved/kda/knn.pkl", "wb"))

    ada = train_ada_model(X_train, y_train)
    pickle.dump(ada, open("saved/kda/ada.pkl", "wb"))

    mpl = train_mpl_model(X_train, y_train)
    pickle.dump(mpl, open("saved/kda/mpl.pkl", "wb"))

    svc = train_svc_model(X_train, y_train)
    pickle.dump(svc, open("saved/kda/svc.pkl", "wb"))

    qda = train_qda_model(X_train, y_train)
    pickle.dump(qda, open("saved/kda/qda.pkl", "wb"))

    estimators = [
        ("rf", rf),
        ("dt", dt),
        ("lr", lr),
        ("gb", gb),
        ("gnb", gnb),
        ("knn", knn),
        ("ada", ada),
        ("mpl", mpl),
        ("svc", svc),
        ("qda", qda),
    ]
    vc = train_vc(X_train, y_train, estimators)
    pickle.dump(vc, open("saved/kda/vc.pkl", "wb"))


def training_as_kda():
    X_train, X_test, y_train, y_test = split_data(data_as_kda, "as_kda")

    rf = train_rf_model(X_train, y_train)
    pickle.dump(rf, open("saved/as_kda/rf.pkl", "wb"))

    dt = train_dt_model(X_train, y_train)
    pickle.dump(dt, open("saved/as_kda/dt.pkl", "wb"))

    lr = train_lr_model(X_train, y_train)
    pickle.dump(lr, open("saved/as_kda/lr.pkl", "wb"))

    gb = train_gb_model(X_train, y_train)
    pickle.dump(gb, open("saved/as_kda/gb.pkl", "wb"))

    gnb = train_gnb_model(X_train, y_train)
    pickle.dump(gnb, open("saved/as_kda/gnb.pkl", "wb"))

    knn = train_knn_model(X_train, y_train)
    pickle.dump(knn, open("saved/as_kda/knn.pkl", "wb"))

    ada = train_ada_model(X_train, y_train)
    pickle.dump(ada, open("saved/as_kda/ada.pkl", "wb"))

    mpl = train_mpl_model(X_train, y_train)
    pickle.dump(mpl, open("saved/as_kda/mpl.pkl", "wb"))

    svc = train_svc_model(X_train, y_train)
    pickle.dump(svc, open("saved/as_kda/svc.pkl", "wb"))

    qda = train_qda_model(X_train, y_train)
    pickle.dump(qda, open("saved/as_kda/qda.pkl", "wb"))

    estimators = [
        ("rf", rf),
        ("dt", dt),
        ("lr", lr),
        ("ada", ada),
        ("gb", gb),
        ("gnb", gnb),
        ("knn", knn),
        ("mpl", mpl),
        ("svc", svc),
        ("qda", qda),
    ]
    vc = train_vc(X_train, y_train, estimators)
    pickle.dump(vc, open("saved/as_kda/vc.pkl", "wb"))


def training_de_as_kda_gpm():
    X_train, X_test, y_train, y_test = split_data(data_de_as_kda_gpm, "de_as_kda_gpm")

    rf = train_rf_model(X_train, y_train)
    pickle.dump(rf, open("saved/de_as_kda_gpm/rf.pkl", "wb"))

    dt = train_dt_model(X_train, y_train)
    pickle.dump(dt, open("saved/de_as_kda_gpm/dt.pkl", "wb"))

    lr = train_lr_model(X_train, y_train)
    pickle.dump(lr, open("saved/de_as_kda_gpm/lr.pkl", "wb"))

    gb = train_gb_model(X_train, y_train)
    pickle.dump(gb, open("saved/de_as_kda_gpm/gb.pkl", "wb"))

    gnb = train_gnb_model(X_train, y_train)
    pickle.dump(gnb, open("saved/de_as_kda_gpm/gnb.pkl", "wb"))

    knn = train_knn_model(X_train, y_train)
    pickle.dump(knn, open("saved/de_as_kda_gpm/knn.pkl", "wb"))

    ada = train_ada_model(X_train, y_train)
    pickle.dump(ada, open("saved/de_as_kda_gpm/ada.pkl", "wb"))

    mpl = train_mpl_model(X_train, y_train)
    pickle.dump(mpl, open("saved/de_as_kda_gpm/mpl.pkl", "wb"))

    svc = train_svc_model(X_train, y_train)
    pickle.dump(svc, open("saved/de_as_kda_gpm/svc.pkl", "wb"))

    qda = train_qda_model(X_train, y_train)
    pickle.dump(qda, open("saved/de_as_kda_gpm/qda.pkl", "wb"))

    estimators = [
        ("rf", rf),
        ("dt", dt),
        ("lr", lr),
        ("ada", ada),
        ("gb", gb),
        ("gnb", gnb),
        ("knn", knn),
        ("mpl", mpl),
        ("svc", svc),
        ("qda", qda),
    ]
    vc = train_vc(X_train, y_train, estimators)
    pickle.dump(vc, open("saved/de_as_kda_gpm/vc.pkl", "wb"))


def training_all():
    X_train, X_test, y_train, y_test = split_data(data_all, "all")

    rf = train_rf_model(X_train, y_train)
    pickle.dump(rf, open("saved/all/rf.pkl", "wb"))

    dt = train_dt_model(X_train, y_train)
    pickle.dump(dt, open("saved/all/dt.pkl", "wb"))

    lr = train_lr_model(X_train, y_train)
    pickle.dump(lr, open("saved/all/lr.pkl", "wb"))

    gb = train_gb_model(X_train, y_train)
    pickle.dump(gb, open("saved/all/gb.pkl", "wb"))

    gnb = train_gnb_model(X_train, y_train)
    pickle.dump(gnb, open("saved/all/gnb.pkl", "wb"))

    knn = train_knn_model(X_train, y_train)
    pickle.dump(knn, open("saved/all/knn.pkl", "wb"))

    ada = train_ada_model(X_train, y_train)
    pickle.dump(ada, open("saved/all/ada.pkl", "wb"))

    mpl = train_mpl_model(X_train, y_train)
    pickle.dump(mpl, open("saved/all/mpl.pkl", "wb"))

    svc = train_svc_model(X_train, y_train)
    pickle.dump(svc, open("saved/all/svc.pkl", "wb"))

    qda = train_qda_model(X_train, y_train)
    pickle.dump(qda, open("saved/all/qda.pkl", "wb"))

    estimators = [
        ("rf", rf),
        ("dt", dt),
        ("lr", lr),
        ("ada", ada),
        ("gb", gb),
        ("gnb", gnb),
        ("knn", knn),
        ("mpl", mpl),
        ("svc", svc),
        ("qda", qda),
    ]
    vc = train_vc(X_train, y_train, estimators)
    pickle.dump(vc, open("saved/all/vc.pkl", "wb"))


def split_data(X_t, type):
    X = []

    for t in X_t:
        X.append(t[:-1])

    y = [last for *_, last in X_t]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    pickle.dump(scaler, open("saved/{}/scaler.pkl".format(type), "wb"))
    
    return X_train, X_test, y_train, y_test


def train_rf_model(X_train, y_train):
    model = RandomForestClassifier(max_depth=10, min_samples_split=5, n_estimators=200)
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
    model = LogisticRegression(max_iter=50000)
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
    model = MLPClassifier(max_iter=50000)
    model.fit(X_train, y_train)
    return model


def train_svc_model(X_train, y_train):
    model = SVC(C=1, gamma=0.001, kernel="rbf", probability=True, max_iter=50000)
    model.fit(X_train, y_train)
    return model


def train_qda_model(X_train, y_train):
    model = QuadraticDiscriminantAnalysis(reg_param=0.1)
    model.fit(X_train, y_train)
    return model


def train_vc(X_train, y_train, estimators):
    vc = VotingClassifier(estimators=estimators, voting="soft")
    vc = vc.fit(X_train, y_train)
    return vc


training_kda()
training_as_kda()
training_de_as_kda_gpm()
training_all()
