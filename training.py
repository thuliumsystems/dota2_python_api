from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import json

with open("data/all_features.json") as user_file:
    file_contents = user_file.read()
data_all = json.loads(file_contents)

with open("data/de_as_kda_gpm.json") as user_file:
    file_contents = user_file.read()
data_de_as_kda_gpm = json.loads(file_contents)

with open("data/as_kda.json") as user_file:
    file_contents = user_file.read()
data_as_kda = json.loads(file_contents)


def training_all():
    X_train, X_test, y_train, y_test = split_data(data_all)

    rf = train_rf_model(X_train, y_train)
    pickle.dump(rf, open("saved/all/rf.sav", "wb"))

    dt = train_dt_model(X_train, y_train)
    pickle.dump(dt, open("saved/all/dt.sav", "wb"))

    lr = train_lr_model(X_train, y_train)
    pickle.dump(lr, open("saved/all/lr.sav", "wb"))

    gb = train_gb_model(X_train, y_train)
    pickle.dump(gb, open("saved/all/gb.sav", "wb"))

    gnb = train_gnb_model(X_train, y_train)
    pickle.dump(gnb, open("saved/all/gnb.sav", "wb"))

    knn = train_knn_model(X_train, y_train)
    pickle.dump(knn, open("saved/all/knn.sav", "wb"))

    ada = train_ada_model(X_train, y_train)
    pickle.dump(ada, open("saved/all/ada.sav", "wb"))

    mpl = train_mpl_model(X_train, y_train)
    pickle.dump(mpl, open("saved/all/mpl.sav", "wb"))

    svc = train_svc_model(X_train, y_train)
    pickle.dump(svc, open("saved/all/svc.sav", "wb"))

    qda = train_qda_model(X_train, y_train)
    pickle.dump(qda, open("saved/all/qda.sav", "wb"))


def training_as_kda():
    X_train, X_test, y_train, y_test = split_data(data_as_kda)

    rf = train_rf_model(X_train, y_train)
    pickle.dump(rf, open("saved/as_kda/rf.sav", "wb"))

    dt = train_dt_model(X_train, y_train)
    pickle.dump(dt, open("saved/as_kda/dt.sav", "wb"))

    lr = train_lr_model(X_train, y_train)
    pickle.dump(lr, open("saved/as_kda/lr.sav", "wb"))

    gb = train_gb_model(X_train, y_train)
    pickle.dump(gb, open("saved/as_kda/gb.sav", "wb"))

    gnb = train_gnb_model(X_train, y_train)
    pickle.dump(gnb, open("saved/as_kda/gnb.sav", "wb"))

    knn = train_knn_model(X_train, y_train)
    pickle.dump(knn, open("saved/as_kda/knn.sav", "wb"))

    ada = train_ada_model(X_train, y_train)
    pickle.dump(ada, open("saved/as_kda/ada.sav", "wb"))

    mpl = train_mpl_model(X_train, y_train)
    pickle.dump(mpl, open("saved/as_kda/mpl.sav", "wb"))

    svc = train_svc_model(X_train, y_train)
    pickle.dump(svc, open("saved/as_kda/svc.sav", "wb"))

    qda = train_qda_model(X_train, y_train)
    pickle.dump(qda, open("saved/as_kda/qda.sav", "wb"))


def training_de_as_kda_gpm():
    X_train, X_test, y_train, y_test = split_data(data_de_as_kda_gpm)

    rf = train_rf_model(X_train, y_train)
    pickle.dump(rf, open("saved/de_as_kda_gpm/rf.sav", "wb"))

    dt = train_dt_model(X_train, y_train)
    pickle.dump(dt, open("saved/de_as_kda_gpm/dt.sav", "wb"))

    lr = train_lr_model(X_train, y_train)
    pickle.dump(lr, open("saved/de_as_kda_gpm/lr.sav", "wb"))

    gb = train_gb_model(X_train, y_train)
    pickle.dump(gb, open("saved/de_as_kda_gpm/gb.sav", "wb"))

    gnb = train_gnb_model(X_train, y_train)
    pickle.dump(gnb, open("saved/de_as_kda_gpm/gnb.sav", "wb"))

    knn = train_knn_model(X_train, y_train)
    pickle.dump(knn, open("saved/de_as_kda_gpm/knn.sav", "wb"))

    ada = train_ada_model(X_train, y_train)
    pickle.dump(ada, open("saved/de_as_kda_gpm/ada.sav", "wb"))

    mpl = train_mpl_model(X_train, y_train)
    pickle.dump(mpl, open("saved/de_as_kda_gpm/mpl.sav", "wb"))

    svc = train_svc_model(X_train, y_train)
    pickle.dump(svc, open("saved/de_as_kda_gpm/svc.sav", "wb"))

    qda = train_qda_model(X_train, y_train)
    pickle.dump(qda, open("saved/de_as_kda_gpm/qda.sav", "wb"))


def split_data(X_t):
    X = []

    for t in X_t:
        X.append(t[:-1])

    y = [last for *_, last in X_t]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=100, stratify=y
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


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


training_all()
training_as_kda()
training_de_as_kda_gpm()
