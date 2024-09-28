import tkinter as tk
from tkinter import filedialog, Text, ttk, scrolledtext, messagebox
import os
import subprocess
import re
import sys

root = tk.Tk()
root.title("Python IDE")
root.geometry("1200x800")

output = ""

def run_script():
    global output
    try:
        code = editor.get("1.0", tk.END)
        process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()
        console.config(state="normal")
        console.delete("1.0", tk.END)
        if output:
            console.insert(tk.END, output)
        if error:
            console.insert(tk.END, error)
            # Error highlighting (basic example)
            try:
                line_number = int(re.search(r"line (\\\\\\\\d+)", error).group(1))
                editor.tag_add("error", f"{line_number}.0", f"{line_number}.end")
                editor.tag_configure("error", background="lightcoral")
            except:
                pass # Handle cases where line number cannot be extracted
        console.config(state="disabled")
    except Exception as e:
        console.config(state="normal")
        console.delete("1.0", tk.END)
        console.insert(tk.END, f"Error: {str(e)}")
        console.config(state="disabled")

# Menu Bar
menubar = tk.Menu(root)

# File Menu
filemenu = tk.Menu(menubar, tearoff=0)
def new_file():
    editor.delete("1.0", tk.END)
def open_file():
    file_path = filedialog.askopenfilename(defaultextension=".py", filetypes=[("Python Files", "*.py"), ("All Files", "*")])
    if file_path:
        try:
            with open(file_path, "r") as file:
                editor.delete("1.0", tk.END)
                editor.insert(tk.END, file.read())
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {file_path}")
def save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".py", filetypes=[("Python Files", "*.py"), ("All Files", "*")])
    if file_path:
        try:
            with open(file_path, "w") as file:
                code = editor.get("1.0", tk.END)
                file.write(code)
        except Exception as e:
            messagebox.showerror("Error", f"Error saving file: {str(e)}")
filemenu.add_command(label="New", command=new_file)
filemenu.add_command(label="Open", command=open_file)
filemenu.add_command(label="Save", command=save_file)
filemenu.add_command(label="Save As", command=save_file)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

# Edit Menu
editmenu = tk.Menu(menubar, tearoff=0)
def find_replace():
    find_dialog = tk.Toplevel(root)
    find_dialog.title("Find and Replace")
    find_label = tk.Label(find_dialog, text="Find:")
    find_label.grid(row=0, column=0)
    find_entry = tk.Entry(find_dialog)
    find_entry.grid(row=0, column=1)
    replace_label = tk.Label(find_dialog, text="Replace:")
    replace_label.grid(row=1, column=0)
    replace_entry = tk.Entry(find_dialog)
    replace_entry.grid(row=1, column=1)
    def find():
        text = find_entry.get()
        editor.tag_remove("found", "1.0", tk.END)
        start_index = "1.0"
        while True:
            index = editor.search(text, start_index, tk.END)
            if not index:
                break
            editor.tag_add("found", index, f"{index}+{len(text)}c")
            start_index = f"{index}+{len(text)}c"
    def replace():
        find()
        text = find_entry.get()
        replace_text = replace_entry.get()
        editor.tag_remove("found", "1.0", tk.END) #remove previous highlight
        start_index = "1.0"
        while True:
            index = editor.search(text, start_index, tk.END)
            if not index:
                break
            editor.replace(index, f"{index}+{len(text)}c", replace_text)
            start_index = f"{index}+{len(replace_text)}c"
    def replace_all():
        text = find_entry.get()
        replace_text = replace_entry.get()
        editor.tag_remove("found", "1.0", tk.END) #remove highlight
        start_index = "1.0"
        while True:
            index = editor.search(text, start_index, tk.END)
            if not index:
                break
            editor.replace(index, f"{index}+{len(text)}c", replace_text)
            start_index = f"{index}+{len(replace_text)}c"
    find_button = tk.Button(find_dialog, text="Find", command=find)
    find_button.grid(row=2, column=0)
    replace_button = tk.Button(find_dialog, text="Replace", command=replace)
    replace_button.grid(row=2, column=1)
    replace_all_button = tk.Button(find_dialog, text="Replace All", command=replace_all)
    replace_all_button.grid(row=3, column=0, columnspan=2)
    editor.tag_configure("found", background="yellow")

editmenu.add_command(label="Cut", command=lambda: editor.event_generate("<<Cut>>"))
editmenu.add_command(label="Copy", command=lambda: editor.event_generate("<<Copy>>"))
editmenu.add_command(label="Paste", command=lambda: editor.event_generate("<<Paste>>"))
editmenu.add_separator()
editmenu.add_command(label="Undo", command=lambda: editor.event_generate("<<Undo>>"))
editmenu.add_command(label="Redo", command=lambda: editor.event_generate("<<Redo>>"))
editmenu.add_command(label="Find and Replace", command=find_replace)
menubar.add_cascade(label="Edit", menu=editmenu)

# Run Menu
runmenu = tk.Menu(menubar, tearoff=0)
runmenu.add_command(label="Run Script", command=run_script)
menubar.add_cascade(label="Run", menu=runmenu)

# View Menu (for themes)
viewmenu = tk.Menu(menubar, tearoff=0)
current_theme = "light"
def switch_theme():
    global current_theme
    if current_theme == "light":
        current_theme = "dark"
        editor.config(bg="black", fg="white")
        console.config(bg="#333", fg="white")
        status_bar.config(bg="#333", fg="white")
        line_numbers.config(bg="#333")
    else:
        current_theme = "light"
        editor.config(bg="white", fg="black")
        console.config(bg="lightgray", fg="black")
        status_bar.config(bg="lightgray", fg="black")
        line_numbers.config(bg="lightgray")
viewmenu.add_command(label="Switch Theme", command=switch_theme)
menubar.add_cascade(label="View", menu=viewmenu)

root.config(menu=menubar)

#Notebook Implementation
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Text Editor
editor = scrolledtext.ScrolledText(notebook, font=("Courier", 14), wrap=tk.WORD)

# Output Console
console = scrolledtext.ScrolledText(notebook, state="disabled", font=("Courier", 12), bg="lightgray", wrap=tk.WORD)

#Placement of editor and console inside notebook
editor_frame = tk.Frame(notebook)
editor.window_create("end", window=editor)
console_frame = tk.Frame(notebook)
console.window_create("end", window=console)

# Status Bar
status_bar = tk.Label(root, text="Line: 1, Column: 1", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Syntax Highlighting (basic example)
editor.tag_configure("keyword", foreground="blue", font=("Courier", 14, "bold"))
editor.tag_configure("comment", foreground="green", font=("Courier", 14, "italic"))
editor.tag_configure("string", foreground="red")
editor.tag_configure("number", foreground="purple")

keywords = {r"\\\\\\\\b(if|elif|else|for|while|def|class|return|import|print|True|False|None)\\\\\\\\b": "keyword",
            r"#.*": "comment",
            r'[\\"\\\\\\\\'](.*?)[\\"\\\\\\\\']': "string",
            r"\\\\\\\\b\\\\\\\\d+\\\\\\\\b": "number"}  # Add more keywords and patterns as needed


def highlight_syntax(event=None):
    editor.tag_remove("keyword", "1.0", tk.END)
    editor.tag_remove("comment", "1.0", tk.END)
    editor.tag_remove("string", "1.0", tk.END)
    editor.tag_remove("number", "1.0", tk.END)
    code = editor.get("1.0", "end-1c")
    for pattern, tag in keywords.items():
        for match in re.finditer(pattern, code):
            start = "1.0 +" + str(match.start()) + "c"
            end = "1.0 +" + str(match.end()) + "c"
            editor.tag_add(tag, start, end)

editor.bind("<KeyRelease>", highlight_syntax)

# Line Numbers
line_numbers = tk.Canvas(root, width=30, bg="lightgray")

def update_line_numbers(event=None):
    line_numbers.delete("all")
    line_count = editor.index(tk.END).split(".")[0]
    for i in range(1, int(line_count) + 1):
        line_numbers.create_text(15, i * 15, text=str(i), anchor=tk.W)
    line_numbers.config(scrollregion=line_numbers.bbox("all"))
editor.bind("<KeyRelease>", update_line_numbers)

# Scrollbar synchronization
def sync_scroll(event):
    editor.yview_moveto(line_numbers.canvasy(event.y)/line_numbers.winfo_height())
def sync_scroll_line(event):
    line_numbers.yview_moveto(editor.yview()[0])
editor.bind("<MouseWheel>", sync_scroll)
line_numbers.bind("<MouseWheel>", sync_scroll_line)

#Theme Implementation
def change_theme(theme):
    if theme == "dark":
        editor.config(bg="black", fg="white")
        console.config(bg="#333", fg="white")
        status_bar.config(bg="#333", fg="white")
        line_numbers.config(bg="#333")
    else:
        editor.config(bg="white", fg="black")
        console.config(bg="lightgray", fg="black")
        status_bar.config(bg="lightgray", fg="black")
        line_numbers.config(bg="lightgray")

# Add theme menu items
viewmenu.add_command(label="Light Theme", command=lambda: change_theme("light"))
viewmenu.add_command(label="Dark Theme", command=lambda: change_theme("dark"))

#Status Bar Implementation
def update_status_bar(event):
    line, col = editor.index(tk.INSERT).split(".")
    status_bar.config(text=f"Line: {line}, Column: {col}")
editor.bind("<KeyRelease>", update_status_bar)

#Tab Implementation
def add_tab():
    new_editor = scrolledtext.ScrolledText(notebook, font=("Courier", 14), wrap=tk.WORD)
    notebook.add(new_editor, text=f"Untitled")
    new_editor.bind("<KeyRelease>", highlight_syntax)
    new_editor.bind("<KeyRelease>", update_line_numbers)
    new_editor.bind("<MouseWheel>", sync_scroll) #sync new editor scroll
    new_editor.bind("<Button-1>", sync_scroll_line) #sync new editor scroll
    new_editor.bind("<KeyRelease>", update_status_bar) #update status bar for new editor

filemenu.add_command(label="New Tab", command=add_tab)

#Placement of editor and console inside notebook
notebook.add(editor_frame, text="Code Editor")
notebook.add(console_frame, text="Console")
line_numbers.pack(in_=editor_frame, side=tk.LEFT, fill=tk.Y) #pack line number
editor.pack(in_=editor_frame, expand=True, fill="both") #pack editor
console.pack(in_=console_frame, expand=True, fill="both") #pack console

root.mainloop()