from flask import Flask
from sklearn.neural_network import MLPClassifier
from pymongo import MongoClient

import os

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/start_game_session')
# Takes a network_id and creates a game session. Returns the game session
def start_game_session():
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X, y)
    return str(clf.predict_proba([[2., 2.], [1., 2.]]))
    
@app.route('/make_decision')
# Takes a network_id and a set of input. Queries the given network for a decision, logs the decision for later annotation, and returns the decision
def make_decision():
    db = getRutsuboDB()
    counter = {"count": 1}
    counter_id = db.counters.insert_one(counter).inserted_id
    return str(counter_id)

@app.route('/end_game_session')
# Takes a session_id and a result. Annotes all decisions made during the session with those results. Maybe use the data to train the network
def end_game_session():
    return 'It worked! Finally!!'

@app.route('/create_network')
# Takes a network_id and creates a network. Does not overwrite existing networks
def create_network():
    return 'create_network'


@app.route('/train_network')
# Takes an existing set of training data and a network_id, bootstraps the network
# ONE DAY: Make it asynchronous
def train_network():
    return "train_network"

def getRutsuboDB():
    client = MongoClient(os.environ['OPENSHIFT_MONGODB_DB_HOST'], int(os.environ['OPENSHIFT_MONGODB_DB_PORT']))
    db = client.rutsubo
    db.authenticate(os.environ['OPENSHIFT_MONGODB_DB_USERNAME'], os.environ['OPENSHIFT_MONGODB_DB_PASSWORD'], os.environ['OPENSHIFT_APP_NAME'])
    return db


if __name__ == '__main__':
    app.run()