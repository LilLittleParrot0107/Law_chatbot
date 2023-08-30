from BM25 import BM25
import flask
base_qa = BM25()
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/get_answer',methods = ['POST'])
def get_answer():
    data = request.json
    question = data.get('question')
    if question:
        answer = base_qa.my_search(question)[0]
        return jsonify({"answer":answer})
    else:
        return jsonify({"error": "Question field is missing in the request data."}), 400

if __name__ == '__main__':
    app.run(debug=True)