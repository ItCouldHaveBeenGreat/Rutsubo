import time
from flask import Flask
from flask import request
from sklearn.neural_network import MLPClassifier
import traceback

import os
import pickle
import json

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/make_decision')
# Takes a network_id and a set of input. Queries the given network and returns the decision
def make_decision():
    try:
        network_id = request.args['network_id']
        inputs = [json.loads(request.args['input'])] # extra brackets because inputs must be 2d
        choices = json.loads(request.args['choices'])
        network = load_network(network_id)

        output = network.predict(inputs)
        app.logger.error(str(network.predict_proba(inputs)))
        return str(output)
    except Exception as e:
        app.logger.error(traceback.print_exc())
        return str("[-1]")


@app.route('/create_network', methods=['POST'])
# Takes a network_id and a set of network parameters and creates a network.
def create_network():
    try:
        network_id = request.form['network_id']
        network = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(347, 347), random_state=1)
        store_network(network_id, network, False)
    except Exception as e:
        app.logger.error(traceback.print_exc())
    return "success"


@app.route('/load_training_data', methods=['POST'])
# Takes an array of annotated training data and stores it for training
def load_training_data():
    return "Not implemented"


@app.route('/train_network', methods=['POST'])
def train_network():
    try:
        network_id = request.form['network_id']
        network = load_network(network_id)
        if network == None:
            app.logger.error("No network with network_id " + str(network_id))
            return "No network with network_id " + str(network_id)
        
        raw_decisions = request.form['data']
        decisions = json.loads(raw_decisions)
        inputs = [d['input'] for d in decisions]
        outputs = [d['output'] for d in decisions]

        classes = []
        for x in range(0, 31):
            classes.append(x) 
        for x in range(0, 6):
            for y in range(1, 31):
                classes.append(x * 100 + y)

        network.partial_fit(inputs, outputs, classes)
    
        store_network(network_id, network, True)
    except Exception as e:
        app.logger.error(traceback.print_exc())
    return "success"


# Caches models between use
# network_id : { network, ttl }
model_cache = {}
TTL = 60 * 1000;

# Returns an MLPClassifier
def load_network(network_id):
    if network_id in model_cache:
        cache_line = model_cache[network_id]
        if cache_line['ttl'] > time.time():
            return cache_line['network']

    with open(get_network_path(network_id), 'r') as model_file:
        network = pickle.load(model_file)
        model_cache[network_id] = {"network": network, "ttl": time.time() + TTL}
        app.logger.info("Refreshed cache for " + network_id)
        return network;

def store_network(network_id, network, overwrite=False):
    if os.path.exists(get_network_path(network_id)):
        app.logger.info('Path ' + get_network_path(network_id) + ' already exists')
        if not overwrite:
            return
    with open(get_network_path(network_id), 'wb') as model_file:
        pickle.dump(network, model_file)

def get_network_path(network_id):
    return "Rutsubo/models/" + network_id + ".model"


if __name__ == '__main__':
    app.run()
