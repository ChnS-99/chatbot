# Chatbot using Deep Learning

Simple Chatbot using Deep Learning & Flask

### Initial Setup

Run the below command to clone the repo and create virtual environment
```
git clone https://github.com/ChnS-99/chatbot.git
cd chatbot
python3 -m venv venv
source venv/bin/activate
```

Run the below command to install required packages
```
pip install bs4
pip install nltk
pip install numpy
pip install keras
pip install tensorflow
pip install flask
```

### Dataset

Run the below file to scrape web and crate a json file. Feel free to change the "url" and "data" according to requirements
```
python3 dataset_utils.py
```

### Training

Run the below command to train and create the model
```
python3 chatbot.py
```

### Deploying

Run the below command to deploy the model on web using Flask
```
python app.py
```

### Reference

[Contextual-chatbot](https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077)