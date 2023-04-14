import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from flask import Flask, request, jsonify
from flask_cors import CORS


print('Loading dataset...')

df_review_imb = pd.read_csv('dataset.csv')

print('Splitting dataset...')

train, test = train_test_split(df_review_imb, test_size=0.33, random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

print('Vectorizing dataset...')


tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)
# also fit the test_x_vector
test_x_vector = tfidf.transform(test_x)

print('Training model...')

svc = SVC(C=1,kernel='linear')
svc.fit(train_x_vector, train_y)

pickle.dump(svc, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

print('Model trained!')

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET'])
def predict():
    review = request.args.get('review')
    if review is None:
        return jsonify({'error': 'No review provided'})
    else :
        prediction = svc.predict(tfidf.transform([review]))
        response = jsonify({'prediction': prediction[0]})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

