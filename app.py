#   Python 3.7.8rc1
#
#   This will need to be done in order to solve compatability issues
#
#   python\lib\site-packages\chatterbot\tagging
#
#   change
#   self.nlp = spacy.load(self.language.ISO_639_1.lower())
#
#   to
#
#   if self.language.ISO_639_1.lower() == 'en':
#            self.nlp = spacy.load('en_core_web_sm')
#            self.nlp = spacy.load(self.language.ISO_639_1.lower())
#
#   spacy              2.3.5
#   ChatterBot         1.0.8
#
#   This can all be installed through pip commands
#
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import Flask, render_template, request

app = Flask(__name__)

# Create a chatbot
chatbot = ChatBot('My Chatbot')

# Create a language trainer
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot
trainer.train("chatterbot.corpus.english")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get-response")
def get_response():
    user_input = request.args.get('msg')
    response = chatbot.get_response(user_input)
    return str(response)

if __name__ == "__main__":
    app.run(debug=True)
