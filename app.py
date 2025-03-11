from flask import Flask, request, jsonify
import pickle

# Load the pre-trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Text Classification API!"

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()  # Get the incoming JSON data
        text = data['text']  # Extract the text from the data

        # Preprocess and vectorize the text
        X_new = vectorizer.transform([text])

        # Predict the class using the model
        prediction = model.predict(X_new)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
