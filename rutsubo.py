from flask import Flask
from flask import request
from sklearn.neural_network import MLPClassifier
from pymongo import MongoClient
import traceback

import os
import pickle
import json

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
    try:
        network_id = request.args['network_id']
        inputs = json.loads(request.args['input'])
        choices = json.loads(request.args['choices'])
        db = get_rutsubo_db()
        network = load_network(db, network_id)
        app.logger.error(inputs)
        
        output = network.predict(inputs)
        app.logger.error(str(output))
        return str(output)
    except Exception as e:
        app.logger.error(traceback.print_exc())
        return str("[-1]")


@app.route('/end_game_session')
# Takes a session_id and a result. Annotes all decisions made during the session with those results. Maybe use the data to train the network
def end_game_session():
    network_id = request.args['network_id']
    db = get_rutsubo_db()
    network = db.networks.find_one({'network_id': network_id})
    return str(network)


@app.route('/create_network', methods=['POST'])
# Takes a network_id and a set of network parameters and creates a network. Does not overwrite existing networks
def create_network():
    try:
        network_id = request.form['network_id']
        network = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 20), random_state=1)
        db = get_rutsubo_db()
        was_created = store_network(db, network_id, network, False)
    except Exception as e:
        app.logger.error(traceback.print_exc())
    return "success"


@app.route('/load_training_data', methods=['POST'])
# Takes an array of annotated training data and stores it for training
def load_training_data():
    try:
        raw_decisions = request.form['data']
        decisions = json.loads(raw_decisions)
        for decision in decisions:
            to_insert = {'input': decision['input'],
                         'output': decision['output'],
                         'agent': decision['agent'],
                         'annotation': decision['annotation']}
            db = get_rutsubo_db()
            db.decisions.insert_one(to_insert)
    except Exception as e:
        app.logger.error(traceback.print_exc())
    return "Success"


@app.route('/train_network', methods=['POST'])
def train_network():
    try:
        network_id = request.form['network_id']
        db = get_rutsubo_db()
        network = load_network(db, network_id)
        if network == None:
            app.logger.error("No network with network_id " + str(network_id))
            return "No network with network_id " + str(network_id)
        
        raw_decisions = request.form['data']
        decisions = json.loads(raw_decisions)
        inputs = [d['input'] for d in decisions]
        outputs = [d['output'] for d in decisions]
        train = network.fit(inputs, outputs)
    
        store_network(db, network_id, network)
    except Exception as e:
        app.logger.error(traceback.print_exc())
    return "success"


# Returns an MLPClassifier
def load_network(db, network_id):
    raw_network = db.networks.find_one({'network_id': network_id})
    if raw_network == None:
        return None
    return pickle.loads(raw_network['network_pickle'])


# Returns true if the network was inserted and false if it was updated
def store_network(db, network_id, network, overwrite=True):
    try:
        if db.networks.find_one({'network_id': network_id}) != None:
            app.logger.error("woo! " + str(overwrite))
            if overwrite: 
                db.networks.update_one({"network_id": network_id},
                    {"$set": {"network_pickle": pickle.dumps(network)}})
                app.logger.error("YEAH! " + str(overwrite))
                return True
        else:
            db.networks.insert_one({'network_pickle': pickle.dumps(network),
                                        'network_id': network_id})
    except Exception as e:
        app.logger.error(traceback.print_exc())
    return False

def get_rutsubo_db():
    client = MongoClient(os.environ['OPENSHIFT_MONGODB_DB_HOST'], int(os.environ['OPENSHIFT_MONGODB_DB_PORT']))
    db = client.rutsubo
    db.authenticate(os.environ['OPENSHIFT_MONGODB_DB_USERNAME'], os.environ['OPENSHIFT_MONGODB_DB_PASSWORD'], os.environ['OPENSHIFT_APP_NAME'])
    return db


if __name__ == '__main__':
    app.run()