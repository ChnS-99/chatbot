import numpy as np
import pickle
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.stem import WordNetLemmatizer
import json
import nltk

nltk.download('punkt_tab')
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('python-deep-learning-chatbot/chatbot_model.h5')

# Load the words and classes from the training process
with open('python-deep-learning-chatbot/words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('python-deep-learning-chatbot/classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Load the intents for testing
with open('/home/arjuna/python-deep-learning-chatbot2/python-deep-learning-chatbot/job_intents.json', encoding='utf-8') as data_file:
    intents = json.load(data_file)

# Function to preprocess input text into a bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"Found in bag: {w}")
    return np.array(bag)

# Prepare test data
test_x = []
test_y = []
true_labels = []
predicted_labels = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        test_x.append(bow(pattern, words, show_details=False))
        test_y.append(classes.index(intent['tag']))
        true_labels.append(intent['tag'])

# Convert test data to numpy arrays
test_x = np.array(test_x)
test_y = np.array(test_y)

# Make predictions
predictions = model.predict(test_x)
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted class indices to class labels
predicted_labels = [classes[i] for i in predicted_classes]

# Print classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=classes))

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels, labels=classes))