import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load and preprocess data
with open('job_intents.json', encoding='utf-8') as data_file:
    intents = json.load(data_file)

words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Tokenize and process the data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Initialize training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [0] * len(words)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    for word in pattern_words:
        if word in words:
            bag[words.index(word)] = 1

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

# Convert to numpy array
training = np.array(training, dtype=object)

# Split into train_x and train_y
train_x = np.array(list(training[:, 0]), dtype=float)
train_y = np.array(list(training[:, 1]), dtype=float)

print("Training data created")

# Create and compile model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=700, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model created")
