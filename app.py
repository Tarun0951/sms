import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from twilio.rest import Client  # Import Twilio

# Twilio credentials
TWILIO_ACCOUNT_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''
RECIPIENT_PHONE_NUMBER = ''

# Load the model
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('tfidf (1).pkl')

# Define the stop words
stop_words = set(ENGLISH_STOP_WORDS)

def preprocess(inp):
    inp = inp.lower()  # Convert to lowercase
    inp = ' '.join(inp.split())  # Replace multiple spaces with a single space
    inp = ' '.join([word for word in inp.split() if word not in stop_words])  # Tokenize the sentence
    ps = PorterStemmer()
    inp = ' '.join([ps.stem(i) for i in inp.split()])  # Stemming
    return inp  # Return the processed text

def detect_suicide(input_text):
    processed_text = preprocess(input_text)
    predict = model.predict(vectorizer.transform([processed_text]).toarray())  # Convert to a dense numpy array
    return predict[0]



# Initialize the Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_sms(message):
    # Send an SMS
    message = client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    return message.sid

st.title("Suicide Detection and SMS Notification")

# Define two empty lists to store user and chatbot messages
user_messages = []
chatbot_messages = []

# Define a text input widget for the user to enter messages
user_input = st.text_area("User Message", key="user_input")

# Define a button to send the user message and detect suicide
if st.button("Send", key="send_button"):
    user_messages.append(user_input)

    # Perform suicide detection
    result = detect_suicide(user_input)
    if result == 'suicide':
        message = "Possible suicide risk detected. Seek help immediately."
        chatbot_messages.append(f"Chatbot: {message}")
        # Send an SMS alert
        send_sms(message)
    else:
        chatbot_messages.append("Chatbot: No suicide risk detected.")

# Display the chat interface
st.subheader("Chat:")
for user_msg, chatbot_msg in zip(user_messages, chatbot_messages):
    st.markdown(f"User: {user_msg}")
    st.markdown(chatbot_msg)



  


