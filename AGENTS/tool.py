import json
import os
import re

# Define a function to append tasks to the dashboard database
def append_task(task_details):
    # Assuming the dashboard database is a JSON file
    db_file = "dashboard_db.json"
    if os.path.exists(db_file):
        with open(db_file, "r") as f:
            db_data = json.load(f)
    else:
        db_data = []

    # Append the new task to the database
    db_data.append(task_details)

    with open(db_file, "w") as f:
        json.dump(db_data, f)

    return "Task appended successfully"

# Define a function to create and write to a file
def create_and_write_file(file_name, action, append_code=None, target_code=None):
    if action == "initial":
        # Create a new file and write the initial code
        with open(file_name, "w") as f:
            f.write(append_code)
        return f"File {file_name} created and initial code written"
    elif action == "refactor":
        # Refactor the code in the file
        with open(file_name, "r") as f:
            file_content = f.read()
        # Use regular expression to find the target code
        if target_code:
            target_code = re.sub(r"\n", "", target_code)
            target_code = re.sub(r"\s+", " ", target_code)
        pattern = re.compile(re.escape(target_code), re.IGNORECASE)
        refactored_content = pattern.sub(append_code, file_content)
        with open(file_name, "w") as f:
            f.write(refactored_content)
        return f"Code in file {file_name} refactored"
    elif action == "new_feature":
        # Append a new feature to the file
        with open(file_name, "r") as f:
            file_content = f.read()
        # Use regular expression to find the target code
        if target_code:
            target_code = re.sub(r"\n", "", target_code)
            target_code = re.sub(r"\s+", " ", target_code)
        pattern = re.compile(re.escape(target_code), re.IGNORECASE)
        new_feature_content = pattern.sub(target_code + "\n\n" + append_code, file_content)
        with open(file_name, "w") as f:
            f.write(new_feature_content)
        return f"New feature appended to file {file_name}"

# Define a function to ask the user for clarification
def ask_user_clarification(question):
    # Assuming the user's response will be captured and processed further
    return f"Question posed to user: {question}"

# Define a function to process the AI agent's response
def process_response(response):
    if "dashboard_of_plans" in response:
        # Append tasks to the dashboard database
        task_details = response["dashboard_of_plans"]
        return append_task(task_details)
    elif "create_and_write" in response:
        # Create and write to a file
        file_name = response["create_and_write"]
        action = response["action"]
        append_code = response.get("append_code")
        target_code = response.get("target_code")
        # Restructure the append_code and target_code to remove newlines and add \n\n
        if append_code:
            append_code = re.sub(r"\n", "", append_code)
            append_code = re.sub(r"\s+", " ", append_code)
            append_code = append_code.replace(" ", "\n\n")
        if target_code:
            target_code = re.sub(r"\n", "", target_code)
            target_code = re.sub(r"\s+", " ", target_code)
        return create_and_write_file(file_name, action, append_code, target_code)
    elif "question_user" in response:
        # Ask the user for clarification
        question = response["question_user"]
        return ask_user_clarification(question)
    else:
        return "Invalid response for tool call"