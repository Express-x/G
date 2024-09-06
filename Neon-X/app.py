from flask import Flask, render_template, request, jsonify
from groq import Groq
import json


app = Flask(__name__)

# Create a Groq client with the API key
client = Groq(api_key="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")

# Define a function to handle user input and generate a response using the Groq chat model

def generate_response(user_input):
    # Create a new chat completion request
    chat_completion = client.chat.completions.create(
        messages=[
            {"role":"system","content": "You are a smart coder assistant"},
            {"role": "user", "content": user_input}
        ],
        model="llama-3.1-70b-versatile",
        temperature=0.5,
        max_tokens=8100,
        top_p=1
    )
    # Return the generated response
    return chat_completion.choices[0].message.content


# Define a route for the chat interface
@app.route("/chat", methods=["GET", "POST"])

def chat():
    if request.method == "POST":
        # Get the user input from the form
        user_input = request.form["user_input"]
        # Generate a response using the Groq chat model
        response = generate_response(user_input)
        # Return the response as JSON
        return jsonify({"response": response})
    # Render the chat interface template
    return render_template("chat.html")

# Define a route for the model configuration interface
@app.route("/model_config", methods=["GET", "POST"])

def model_config():
    if request.method == "POST":
        # Get the model configuration from the form
        system_instruction = request.form["system_instruction"]
        model_name = request.form["model_name"]
        api_key = request.form["api_key"]
        # Save the model configuration
        with open("model_config.json", "w") as f:
            json.dump({
                "system_instruction": system_instruction,
                "model_name": model_name,
                "api_key": api_key
            }, f)
        # Return a success message
        return "Model configuration saved successfully"
    # Render the model configuration interface template
    return render_template("model_config.html")


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)