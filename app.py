import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        
counter = 0

def main():
    global counter
    st.title("Healthcare Chatbot by Vaishnavi Tripathi")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("Welcome to the Healthcare Chatbot! Please type your health-related question or concern below, and I will do my best to assist you.")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User  Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:

            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Take care and stay healthy!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f":User  {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("This Healthcare Chatbot is designed to assist users with health-related inquiries. It was developed by Vaishnavi Tripathi, utilizing Natural Language Processing (NLP) techniques and Logistic Regression to understand and respond to user input based on predefined intents.")

        st.subheader("Project Overview:")

        st.write("""
        The project consists of two main components:
        1. **NLP Techniques**: The chatbot is trained using NLP methods and a Logistic Regression algorithm to interpret user queries related to health.
        2. **Streamlit Interface**: The user-friendly interface allows individuals to interact with the chatbot seamlessly.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset comprises a collection of labeled intents and responses relevant to healthcare. 
        - **Intents**: Categories of user inquiries (e.g., "symptoms", "medications", "appointments").
        - **Entities**: Specific details extracted from user input (e.g., "What are the symptoms of flu?", "What medication should I take for a headache?").
        - **Text**: The actual user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit, featuring a text input box for users to submit their health-related questions and a chat window to display responses. The interface leverages the trained model to provide accurate and helpful answers.")

        st.subheader("Conclusion:")

        st.write("This project showcases a healthcare chatbot capable of understanding and responding to user inquiries based on intents. It employs NLP and Logistic Regression for training, while the interactive interface is developed using Streamlit. Future enhancements could include expanding the dataset and integrating more advanced NLP techniques or machine learning models.")

if __name__ == '__main__':
    main()