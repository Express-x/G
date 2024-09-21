import re
import os


def track_agent_responses(responses):
    """
    Tracks agent responses to identify tool use/calls for tools.
    """
    tool_calls = []
    for response in responses:
        # Check for file creation/writing requests
        create_write_match = re.search(r"<\\\\\\\\$create_or_write_to_file\\\\\\\\$:(.+?)\\\\\\\\|\\\\\\\\|\\\\\\\\|(.+?)\\\\\\\\|\\\\\\\\|\\\\\\\\|(.+?)>", response)
        if create_write_match:
            tool_calls.append({
                "tool": "create_or_write_to_file",
                "file_name": create_write_match.group(1),
                "content_type": create_write_match.group(2),  # e.g., 'Code', 'Text', etc.
                "content": create_write_match.group(3),
            })

        # Check for refactoring requests
        refactor_match = re.search(r"<\\\\\\\\$create_or_write_to_file\\\\\\\\$:(.+?)\\\\\\\\|\\\\\\\\|\\\\\\\\|Refactor\\\\\\\\|\\\\\\\\|\\\\\\\\|(.+?)>(.+?)<\\\\\\\\$End_code\\\\\\\\$>", response)
        if refactor_match:
            tool_calls.append({
                "tool": "refactor_code",
                "file_name": refactor_match.group(1),
                "new_code": refactor_match.group(2),
                "old_code": refactor_match.group(3),
            })

        # Add more checks for other tool calls as needed
        # Check for extra feature requests
        extra_feature_match = re.search(r"<\\\\\\\\$create_or_write_to_file\\\\\\\\$:(.+?)\\\\\\\\|\\\\\\\\|\\\\\\\\|Extra_Feature\\\\\\\\|\\\\\\\\|\\\\\\\\|(.+?)>(.+?)<\\\\\\\\$End_code\\\\\\\\$>", response)
        if extra_feature_match:
            tool_calls.append({
                "tool": "add_extra_feature",
                "file_name": extra_feature_match.group(1),
                "new_code": extra_feature_match.group(2),
                "code_to_insert_after": extra_feature_match.group(3),  # Code to insert the extra feature after
            })

    return tool_calls


def create_or_write_file(file_name, content_type, content):
    """
    Creates or writes to a file with the given name and content.
    """
    try:
        with open(file_name, "w") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error creating/writing to file: {e}")
        return False


def refactor_code(file_name, new_code, old_code):
    """
    Refactors code in the specified file by replacing old code with new code.
    """
    try:
        with open(file_name, "r") as f:
            content = f.read()
        new_content = content.replace(old_code, new_code)
        with open(file_name, "w") as f:
            f.write(new_content)
        return True
    except Exception as e:
        print(f"Error refactoring code: {e}")
        return False


def add_extra_feature(file_name, new_code, code_to_insert_after):
    """
    Adds an extra feature to the specified file by placing the new code after the specified code.
    """
    try:
        with open(file_name, "r") as f:
            content = f.read()
        if code_to_insert_after in content:
            parts = content.split(code_to_insert_after)
            new_content = parts[0] + code_to_insert_after + new_code + parts[1]
            with open(file_name, "w") as f:
                f.write(new_content)
            return True
        else:
            print(f"Error adding extra feature: Code to insert after not found in file {file_name}")
            return False
    except Exception as e:
        print(f"Error adding extra feature: {e}")
        return False


def append_task(task):
    """
    Appends a task to the developer agent's task list.
    """
    with open("developer_tasks.txt", "a") as f:
        f.write(task + "\\\\\\\\\\\\\\\\\\\\\\")


def get_next_task():
    """
    Gets the next task from the developer agent's task list.
    """
    with open("developer_tasks.txt", "r") as f:
        tasks = f.readlines()
    if tasks:
        next_task = tasks.pop(0).strip()
        # Update the task list file
        with open("developer_tasks.txt", "w") as f:
            f.writelines(tasks)
        return next_task
    else:
        return None
