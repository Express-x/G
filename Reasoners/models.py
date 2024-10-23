import asyncio
import json
from groq import AsyncGroq

# Initialize the Groq client with the API key (can be passed as a parameter for better modularity)
client = AsyncGroq(api_key="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")

# Asynchronous function to get a chat completion from the Groq API.  Handles potential errors.
async def chat_completion_async(messages, model, temperature=0.5, max_tokens=8000, top_p=1, stop=None):
    try:
        completion = await client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        # The model's response is in completion.choices[0].message.content
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


# Asynchronous function to handle a single chat instance.  Includes system instruction.
async def handle_chat_instance(messages, model, instance_id, system_instruction):
    messages.insert(0, {"role": "system", "content": system_instruction})
    result = await chat_completion_async(messages, model)
    print(f"Instance {instance_id}: {result}")
    return result


# Asynchronous function to process multiple tasks concurrently.  Handles empty task lists.
async def process_tasks(tasks, model, instructions):
    if not tasks:
        return []
    
    num_instances = len(tasks)
    task_results = []
    # Asynchronous function to process a single task
    async def process_single_task(task, instruction, instance_id):
        messages = [{"role": "user", "content": task}]
        result = await handle_chat_instance(messages, model, instance_id, instruction)
        return result

    task_list = []
    for i, task in enumerate(tasks):
        task_list.append(process_single_task(task, instructions[i % len(instructions)], i+1))

    task_results = await asyncio.gather(*task_list)
    return task_results
