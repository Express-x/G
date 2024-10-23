import asyncio
import json
from groq import AsyncGroq # Added import statement
from Reasoners.models import process_tasks, chat_completion_async, handle_chat_instance #Import necessary functions

# Initialize the Groq client with the API key
client = AsyncGroq(api_key="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")

async def main(tasks):
    model = "llama-3.2-90b-Text-Preview"
    try:
        with open('Reasoners/instruction.json', 'r') as f:
            instructions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        instructions = ["You are a helpful assistant."] * len(tasks)

    results = await process_tasks(tasks, model, instructions)
    return results


if __name__ == "__main__":
    tasks = [f"Task {i+1}: What is the meaning of life?" for i in range(3)]  # Example tasks
    results = asyncio.run(main(tasks))
    print("All instances completed. Results:")
    for i, result in enumerate(results):
        print(f"Task {i+1}: {result}")
