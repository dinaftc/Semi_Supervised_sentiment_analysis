from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import demoji  # Import the demoji library
import re
import string

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('BestModel.joblib')

# Initialize the demoji library
demoji.download_codes()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text_input']

        # Clean and prepare the input text
        cleaned_text = remove_mult_spaces(clean_hashtags(strip_all_entities(clean_text_from_emojis(text_input))))

        # Use the trained model to predict the sentiment
        predicted_sentiment = model.predict([cleaned_text])  # Fix variable name

        reverse_mapping = {
            -1: "none",
            0: "negative",
            1: "neutral",  # Fix typo in "neutral"
            2: "positive"
        }

        # Map the predicted sentiment back to the original labels
        predicted_sentiment_label = reverse_mapping.get(predicted_sentiment[0], "Unknown")

        # Make predictions using the pre-trained model (you might not need this line)
        # prediction = model.predict([cleaned_text])

        return render_template('result.html', prediction=predicted_sentiment_label, text_input=text_input)

def clean_text_from_emojis(text):
    return demoji.replace(text, '')

def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return new_tweet2

def remove_mult_spaces(text):
    return re.sub("\s\s+", " ", text)

if __name__ == '__main__':
    app.run(debug=True)
