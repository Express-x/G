from flask import Flask, request, jsonify, render_template

from groq import Groq
import json

app = Flask(__name__)

# Initialize Groq client with API key
client = Groq(api_key="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")

# Load system instructions from JSON file
with open("instruct_and_Ipex_tool.json", "r") as f:
    system_instructions = json.load(f)

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form.get("message")
        if user_input:
            # Generate response using Groq
            response = generate_response(user_input)
            return jsonify({"response": response})
        else:
            return jsonify({"error": "No message provided"})
    else:
        return render_template("index.html")

def generate_response(user_input):
    """Generates a response using the Groq model."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_instructions["system_instructions"],
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stop=None,
        stream=False,
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    app.run(debug=True)