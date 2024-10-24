import streamlit as st
import requests

# Set the FastAPI endpoint
API_URL = "http://localhost:8000/qa"

# Streamlit app
st.title("Q&A Chatbot")
st.write("Hi there 👋🏽")

# Input field for the question
question = st.text_input("Enter your question:")

# Button to get a response
if st.button("Get Response"):
    if question:
        try:
            # Send a POST request to the FastAPI endpoint
            response = requests.post(API_URL, json={"text": question})
            response_data = response.json()
            
            # Display the response
            st.write(f"**Response:** {response_data['response']}")
            st.write(f"_(Predicted Intent: {response_data['intent']})_")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the API: {e}")
    else:
        st.warning("Please enter a question.")

# Option to continue asking questions
st.write("Feel free to ask more questions!")