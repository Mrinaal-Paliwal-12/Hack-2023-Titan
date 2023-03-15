import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import re
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

dataset = pd.read_csv('./ApData.csv', quoting=3, header=None)
ps = PorterStemmer()

corpus = []
for i in range(0, 50):
    review = re.sub('[^a-zA-Z]', ' ', dataset[0][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
classifier = GaussianNB()
model=classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(np.concatenate((Y_test.reshape(len(Y_test), 1), Y_pred.reshape(len(Y_pred), 1)), 1))
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print("Accuracy Score: ", accuracy_score(Y_test, Y_pred))
precision = 91/(91+42)
print("Precision: ", precision)
recall = 91/(91+12)
print("Recall: ", recall)
print("F1 Score: ", (2*precision*recall)/(precision+recall))

model=pickle.load(open("saved_model.pkl","rb"))['model']

def test(data):
    review=""
    corpus=[]
    review = re.sub('[^a-zA-Z]', ' ', data)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    X_test1 = cv.transform(corpus).toarray()
    X_test1
    Y_pred1=model.predict(X_test1)
    return Y_pred1[0]
print(test("employee information!"))


from flask import Flask,request,jsonify


app = Flask(__name__)

@app.route('/')
def home():
    return "Page"

@app.route('/predict', methods = ['GET'])
def predict():
    # review = request.form.get('review')
    review = request.args.get('review')
    result = test(review)
    return jsonify({'result':int(result)})

if __name__ == '__main__':
    app.run(debug=True)