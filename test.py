from flask import Flask, jsonify
import pickle

app = Flask(__name__)


@app.route("/", methods=["GET"])
def result():
    
    rf = pickle.load(open("saved/as_kda/rf.sav", "rb"))
    print(str(rf))
    return jsonify({"a": rf})