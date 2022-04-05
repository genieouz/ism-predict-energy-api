from flask import Flask, request
from urlextract import URLExtract
import numpy as np
import pickle
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
extractor = URLExtract()

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['POST'])
def index():
    data_form = request.get_json(force=True)
    return predict(data_form)

def predict(data_form):
    int_features = [int(x) for x in data_form.values()]
    final_features  = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    return { "result": output }