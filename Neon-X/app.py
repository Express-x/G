from flask import Flask, render_template, request
from groq import Groq

app = Flask(__name__, template_folder="templates")

client = Groq(api_key="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")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input")
    system_instruction = "you're a smart coder assistant you goal is to write codes in very concise and software engineer perspective"
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ],
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1
    )
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    app.run(debug=True)