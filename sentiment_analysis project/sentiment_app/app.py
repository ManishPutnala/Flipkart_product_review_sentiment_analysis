from flask import Flask, render_template, request
import re 
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib

app = Flask(__name__)
model = joblib.load("model/SVC.pkl")
stop_words = set(stopwords.words('english'))

def text_process(text):
    text = str(text)
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '',text)
    clean_text = clean_text.lower()
    tokens = word_tokenize(clean_text)    
    stop_words = set(stopwords.words('english'))
    clean_text = [token for token in tokens if token not in stop_words]
    return " ".join(clean_text)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/prediction', methods=['POST'])
def prediction():
    review = request.form['review']
    final_text = text_process(review)
    prediction = model.predict([final_text])
    sentiment = "Positive" if prediction[0] == 'Positive' else "Negative"
    return render_template("output.html", prediction=prediction, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
