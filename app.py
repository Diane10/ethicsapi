# import sklearn
# print(sklearn.__version__)

# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# import joblib
# # Later, to load the model and vectorizer for predictions
# print("start predicting!!")
# loaded_rf_model = joblib.load('random_forest_model.pkl')
# loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# # New text data
# new_text = ['When I go camping with my family, I deserve to borrow sunscreen from my brother because I let him borrow some bug spray']

# # Preprocess the new text data
# preprocessed_new_text = [preprocess_text(text) for text in new_text]

# # Transform the preprocessed text using the loaded TF-IDF vectorizer
# X_new = loaded_tfidf_vectorizer.transform(preprocessed_new_text)

# # Make predictions using the loaded Random Forest classifier
# new_predictions = loaded_rf_model.predict(X_new)

# # Output predictions
# for text, prediction in zip(new_text, new_predictions):
#     print(f"Text: {text} - Predicted Label: {prediction}")

from flask import Flask, jsonify, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load the pre-trained model and vectorizer
loaded_rf_model = joblib.load('random_forest_model.pkl')
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercasing
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]  # Lemmatization
    tokens = [token for token in tokens if token not in stop_words]  # Removing stop words
    return ' '.join(tokens)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    new_text = data['text']

    # Preprocess the new text data
    preprocessed_new_text = [preprocess_text(text) for text in new_text]

    # Transform the preprocessed text using the loaded TF-IDF vectorizer
    X_new = loaded_tfidf_vectorizer.transform(preprocessed_new_text)

    # Make predictions using the loaded Random Forest classifier
    new_predictions = loaded_rf_model.predict(X_new)

    # Output predictions
    predictions = []
    for text, prediction in zip(new_text, new_predictions):
        predictions.append({'Text': text, 'Predicted_Label': prediction})

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(port=5000, debug=True,host="0.0.0.0")
