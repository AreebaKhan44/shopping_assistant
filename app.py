from flask import Flask, request, jsonify
from rag_bot import ask_question

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question not provided"}), 400
    
    answer = ask_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
