import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS


# load the model and tfidf
print('Loading model...')

svc = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

print('Model loaded!')

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "to predict, post to /predict"


@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    prediction = svc.predict(tfidf.transform([review]))
    response = jsonify({'prediction': prediction[0]})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

