dataset = {
    "greetings": {
        "patterns": ["Hi", "Hello", "Hey"],
        "responses": ["Hello! How can I assist you?", "Hi there! How can I help you today?"]
    },
    "menu": {
        "patterns": ["What's on the menu?", "What dishes do you offer?", "Show me the menu"],
        "responses": ["We have a variety of dishes available. What type of cuisine are you interested in?"]
    },
    "delivery": {
        "patterns": ["Do you offer delivery?", "Can you deliver food?", "Is delivery available?"],
        "responses": ["Yes, we offer delivery services. Please provide your address so we can check if it's within our delivery range."]
    },
    "order": {
        "patterns": ["How can I place an order?", "I want to order food", "Can I make an order?"],
        "responses": ["To place an order, you can either call our hotline or use our mobile app. Which method would you prefer?"]
    },
    "payment": {
        "patterns": ["What payment methods do you accept?", "Can I pay by credit card?", "Do you accept cash?"],
        "responses": ["We accept both cash and credit card payments. You can choose your preferred method at the time of delivery."]
    },
    "thank_you": {
        "patterns": ["Thank you", "Thanks a lot", "Appreciate it"],
        "responses": ["You're welcome! If you have any more questions, feel free to ask."]
    },
    "goodbye": {
        "patterns": ["Goodbye", "Bye", "See you later"],
        "responses": ["Goodbye! Have a great day."]
    },
    "safari": {
        "patterns": ["Tell me about safaris", "What safaris do you offer?", "Can you provide safari details?" , " What are the prices for the safari packages?" , "What safety measures should you take when going on a safari?"],
        "responses": ["We offer exciting safari experiences in various national parks and reserves. Our safaris include guided tours, wildlife viewing, and accommodation. Which specific safari are you interested in?", 
                      "The prices for the safari packages vary depending on the package. For example, the all-inclusive packages start at 2,000 per person, the partial-inclusive packages start at 1,500 per person, the private packages start at 5,000 per person, and the group packages start at 3,000 per person." , "Keep be by your guide's side,Follow your guide's directions,Pay attention your surroundings."]
    },
    "foods": {
        "patterns": ["What food options do you have?", "Tell me about your food offerings", "What cuisines are available?" , "Can I get assistance with meal planning?" , "What should I keep out of the room?"],
        "responses": [
            "We have a diverse range of food options, including local and international cuisines. Our menu features appetizers, main courses, desserts, and beverages. Let me know if you have any dietary preferences or restrictions." , 
            "Yes, you can get guidance with food planning. You can utilize the hotel's website service to request assistance." , 
            " Smoking ,Drinking alcohol,Eating in bed"],
    },
    "rooms": {
        "patterns": [
            "Tell me about your rooms", 
            "What types of rooms do you offer?", 
            "Can you describe the accommodations?" 
            ,"What is the minimum age limit for the rooms?" , 
            "Can you help me with my food planning? "],
        "responses": ["We provide comfortable and luxurious rooms with modern amenities. Our rooms include standard, deluxe, and suite options. Each room is designed to offer a relaxing and enjoyable stay. How many guests will be staying, and for what dates?" , 
                      " According on the kind of room, different age limits apply. Standard rooms, for instance, are available to guests of all ages, whereas deluxe rooms, suites, and presidential suites are only available to guests who are 12 years old or older." , 
                      "Can you help me with my food planning? "],
    },
    "activities": {
        "patterns": ["What activities are available?", 
                     "What can I do during my stay?",
                       "Tell me about the recreational options",
                       "What are the hotel's cancellation guidelines for activities?" , 
                       "What time periods do the activities at the hotel operate?"],
        "responses": ["We offer a variety of activities for our guests, including swimming, spa treatments, nature walks, and cultural experiences. You can also explore nearby attractions and participate in guided tours. Let us know your interests, and we can provide more information." , "According on the activity, different cancellation procedures apply to hotel activities. The fitness center may be cancelled up to 24 hours in advance for a full refund, and yoga sessions can be cancelled up to 12 hours in advance for a full refund, however the pool and hot tub are non-refundable." , "The response varies based on the activity, the hotel's operating hours. For instance, the fitness facility is open from 6am to 10pm, the swimming pool is accessible from 7am to 8am, and yoga lessons are available from 6pm to 7pm, Monday through Friday. "],
    }
}

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

def get_reply(intent):
    responses = dataset[intent]['responses']
    random_response = random.choice(responses)
    return random_response

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load model, tokenizer, and label encoder
model = tf.keras.models.load_model('my_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as ecn_file:
    le = pickle.load(ecn_file)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    sentence = message['sentence']
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=11)
    prediction = model.predict(padded_sequence)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    



    response = {
        'prediction': {
            'sentence': sentence,
            'intent': predicted_label[0],
            "reply" : get_reply(predicted_label[0])
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=1385)
