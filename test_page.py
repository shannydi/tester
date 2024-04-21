
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the vectorizer and model (adjust paths as necessary)
# vectorizer = joblib.load('tfidf_vectorizer.pkl') 
# model = joblib.load('sentiment_model_np.pkl')

vectorizer = joblib.load(r'C:\Users\Dianne\Desktop\DT\nlp_env\tfidf_vectorizer.pkl') 
model = joblib.load(r'C:\Users\Dianne\Desktop\DT\nlp_env\sentiment_model_np.pkl')



def predict_sentiment(text):
    """Function to predict sentiment from text input using loaded model and vectorizer."""
    transformed_text = vectorizer.transform([text])
    sentiment = model.predict(transformed_text)
    sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    return sentiment_map[sentiment[0]]

def main():
    # Streamlit page configuration
    st.title('Sentiment Analysis App')
    st.write('This app uses a machine learning model to predict sentiment from text.')

    # User text input
    user_input = st.text_area("Enter text here to analyze sentiment:", height=150)
    if st.button("Predict Sentiment"):
        if user_input:
            # Predict sentiment
            result = predict_sentiment(user_input)
            # Display the prediction
            st.success(f"The predicted sentiment is: {result}")
        else:
            st.error("Please enter some text to analyze.")

if __name__ == "__main__":
    main()