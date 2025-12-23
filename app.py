from flask import Flask, render_template, request, jsonify
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

# Sample knowledge base
knowledge_base = {
    "hello": "Hi there! How can I assist you today?",
    "how are you": "I am an AI chatbot, ready to help you!",
    "what is ai": "AI stands for Artificial Intelligence, enabling machines to simulate human intelligence.",
    "thank you": "You're welcome! Happy to help."
}
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get", methods=["POST"])
def chatbot_response():
    user_text = request.form["msg"].lower()
    # Tokenization & Lemmatization
    tokens = nltk.word_tokenize(user_text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
             # Compute cosine similarity with knowledge base
    corpus = list(knowledge_base.keys())
    vectorizer = TfidfVectorizer().fit_transform(corpus + [' '.join(tokens)])
    similarity = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    index = similarity.argmax()



    response = knowledge_base[corpus[index]] if similarity[0][index] > 0 else "I'm not sure about that. Could you please elaborate?"
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
