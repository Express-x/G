import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import font
import os
from tkinter import ttk  # Import ttk for themed widgets
import builtins
import keyword

class Autocomplete:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.bind("<KeyRelease>", self.show_autocomplete)
        self.autocomplete_window = None
        self.suggestions = []

    def show_autocomplete(self, event):
        if event.keysym in ["space", "period", "("]:
            word = self.get_current_word()
            if word:
                self.suggestions = self.get_suggestions(word)
                if self.suggestions:
                    self.show_suggestions()
        elif self.autocomplete_window:
            self.hide_suggestions()

    def get_current_word(self):
        cursor_position = self.text_widget.index(tk.INSERT)
        line, column = cursor_position.split('.')
        line_text = self.text_widget.get(f"{line}.0", f"{line}.end")
        word = ""
        for i in range(column - 1, -1, -1):
            char = line_text[i]
            if char.isalnum() or char == '_':
                word = char + word
            else:
                break
        return word

    def get_suggestions(self, word):
        suggestions = []
        for module in list(builtins.__dict__.keys()) + list(keyword.kwlist):
            if module.startswith(word) and module != word:
                suggestions.append(module)
        return suggestions

    def show_suggestions(self):
        if self.autocomplete_window:
            self.autocomplete_window.destroy()
        x, y = self.text_widget.winfo_rootx(), self.text_widget.winfo_rooty() + self.text_widget.winfo_height()
        self.autocomplete_window = tk.Toplevel(self.text_widget)
        self.autocomplete_window.wm_overrideredirect(True)
        self.autocomplete_window.geometry(f"+{x}+{y}")

        listbox = tk.Listbox(self.autocomplete_window)
        listbox.pack()
        for suggestion in self.suggestions:
            listbox.insert(tk.END, suggestion)

        listbox.bind("<Double-Button-1>", self.insert_suggestion)

    def insert_suggestion(self, event):
        selected_suggestion = event.widget.get(tk.ACTIVE)
        word = self.get_current_word()
        cursor_position = self.text_widget.index(tk.INSERT)
        line, column = cursor_position.split('.')
        self.text_widget.delete(f"{line}.{column - len(word)}", tk.INSERT)
        self.text_widget.insert(tk.INSERT, selected_suggestion)
        self.hide_suggestions()

    def hide_suggestions(self):
        if self.autocomplete_window:
            self.autocomplete_window.destroy()
            self.autocomplete_window = None


class SmartphoneIDE:
    def __init__(self, master):
        self.master = master
        master.title("Smartphone IDE")

        # Use ttk.Notebook for tabbed interface
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create a frame for the line numbers and text area
        editor_frame = tk.Frame(self.notebook)
        editor_frame.pack(fill=tk.BOTH, expand=True)

        # Add line numbers
        self.line_numbers = tk.Text(editor_frame, width=4, bg="lightgrey", state="disabled")
        self.line_numbers.pack(side="left", fill="y")

        # Add text area for code editing
        self.text_area = tk.Text(editor_frame, wrap=tk.WORD)
        self.text_area.pack(side="right", fill="both", expand=True)

        self.notebook.add(editor_frame, text="Code")

        # Add a console for output
        self.console = tk.Text(self.notebook, wrap=tk.WORD, bg="black", fg="white")
        self.notebook.add(self.console, text="Console")

        self.font = font.Font(family="Courier New", size=12)
        self.text_area.configure(font=self.font)
        self.console.configure(font=self.font)

        self.autocomplete = Autocomplete(self.text_area)

        self.create_menu()

        # Bind events for smart indentation
        self.text_area.bind("<Return>", self.smart_indent)
        self.text_area.bind("<KeyPress-{>", self.handle_open_bracket)  # Bind to '<KeyPress-{>'
        self.text_area.bind("<KeyPress-[>", self.handle_open_bracket)  # Bind to '<KeyPress-[>'
        self.text_area.bind("<KeyPress-(>", self.handle_open_bracket)  # Bind to '<KeyPress-(>'
        self.text_area.bind("<BackSpace>", self.handle_backspace)

        self.update_line_numbers()
        self.text_area.bind("<KeyRelease>", self.update_line_numbers)

    def create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.new_file)
        filemenu.add_command(label="Open", command=self.open_file)
        filemenu.add_command(label="Save", command=self.save_file)
        filemenu.add_command(label="Save As", command=self.save_as_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.master.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Cut", command=self.cut)
        editmenu.add_command(label="Copy", command=self.copy)
        editmenu.add_command(label="Paste", command=self.paste)
        editmenu.add_command(label="Find", command=self.find)
        editmenu.add_command(label="Replace", command=self.replace)
        editmenu.add_command(label="Indent", command=self.indent_selection)
        editmenu.add_command(label="Unindent", command=self.unindent_selection)
        editmenu.add_command(label="Comment/Uncomment", command=self.comment_uncomment)
        menubar.add_cascade(label="Edit", menu=editmenu)

        runmenu = tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="Run", command=self.run_code)
        menubar.add_cascade(label="Run", menu=runmenu)

        # Add view menu for font size
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Increase Font Size", command=self.increase_font_size)
        viewmenu.add_command(label="Decrease Font Size", command=self.decrease_font_size)
        menubar.add_cascade(label="View", menu=viewmenu)

        # Add theme menu
        thememenu = tk.Menu(menubar, tearoff=0)
        thememenu.add_command(label="Light", command=lambda: self.change_theme("light"))
        thememenu.add_command(label="Dark", command=lambda: self.change_theme("dark"))
        menubar.add_cascade(label="Theme", menu=thememenu)

    def new_file(self):
        self.text_area.delete("1.0", tk.END)
        self.update_line_numbers()

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                with open(file_path, "r") as f:
                    self.text_area.delete("1.0", tk.END)
                    self.text_area.insert(tk.END, f.read())
                self.update_line_numbers()
            except:
                messagebox.showerror("Error", "Could not open file.")

    def save_file(self):
        if hasattr(self, "file_path") and self.file_path:
            try:
                with open(self.file_path, "w") as f:
                    f.write(self.text_area.get("1.0", tk.END))
            except:
                messagebox.showerror("Error", "Could not save file.")
        else:
            self.save_as_file()

    def save_as_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".py")
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.text_area.get("1.0", tk.END))
                self.file_path = file_path
            except:
                messagebox.showerror("Error", "Could not save file.")

    def cut(self):
        self.text_area.event_generate("<<Cut>>")
        self.update_line_numbers()

    def copy(self):
        self.text_area.event_generate("<<Copy>>")

    def paste(self):
        self.text_area.event_generate("<<Paste>>")
        self.update_line_numbers()

    def find(self):
        self.find_window = tk.Toplevel(self.master)
        self.find_window.title("Find")

        tk.Label(self.find_window, text="Find:").grid(row=0, column=0, sticky=tk.E)
        self.find_entry = tk.Entry(self.find_window)
        self.find_entry.grid(row=0, column=1)

        tk.Button(self.find_window, text="Find", command=self.find_text).grid(row=1, column=1)

    def replace(self):
        self.replace_window = tk.Toplevel(self.master)
        self.replace_window.title("Replace")

        tk.Label(self.replace_window, text="Find:").grid(row=0, column=0, sticky=tk.E)
        self.replace_find_entry = tk.Entry(self.replace_window)
        self.replace_find_entry.grid(row=0, column=1)

        tk.Label(self.replace_window, text="Replace:").grid(row=1, column=0, sticky=tk.E)
        self.replace_entry = tk.Entry(self.replace_window)
        self.replace_entry.grid(row=1, column=1)

        tk.Button(self.replace_window, text="Replace", command=self.replace_text).grid(row=2, column=1)

    def find_text(self):
        text_to_find = self.find_entry.get()
        if text_to_find:
            start = self.text_area.search(text_to_find, "1.0", tk.END)
            if start:
                end = f"{start}+{len(text_to_find)}c"
                self.text_area.tag_add("highlight", start, end)
                self.text_area.tag_config("highlight", background="yellow")
                self.text_area.see(start)

    def replace_text(self):
        text_to_find = self.replace_find_entry.get()
        text_to_replace = self.replace_entry.get()
        if text_to_find:
            start = self.text_area.search(text_to_find, "1.0", tk.END)
            if start:
                end = f"{start}+{len(text_to_find)}c"
                self.text_area.delete(start, end)
                self.text_area.insert(start, text_to_replace)

    def indent_selection(self):
        try:
            selected_text = self.text_area.get(tk.SEL_FIRST, tk.SEL_LAST)
            lines = selected_text.splitlines()
            indented_lines = ["    " + line for line in lines]
            self.text_area.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.text_area.insert(tk.SEL_FIRST, "\\\\
".join(indented_lines))
            self.update_line_numbers()
        except tk.TclError:
            pass  # No selection

    def unindent_selection(self):
        try:
            selected_text = self.text_area.get(tk.SEL_FIRST, tk.SEL_LAST)
            lines = selected_text.splitlines()
            unindented_lines = [line[4:] if line.startswith("    ") else line for line in lines]
            self.text_area.delete(tk.SEL_FIRST, tk.SEL_LAST)
            self.text_area.insert(tk.SEL_FIRST, "\\\\
".join(unindented_lines))
            self.update_line_numbers()
        except tk.TclError:
            pass  # No selection

    def comment_uncomment(self):
        try:
            selected_text = self.text_area.get(tk.SEL_FIRST, tk.SEL_LAST)
            lines = selected_text.splitlines()

            if all(line.startswith("# ") for line in lines):  # Uncomment
                uncommented_lines = [line[2:] for line in lines]
                self.text_area.delete(tk.SEL_FIRST, tk.SEL_LAST)
                self.text_area.insert(tk.SEL_FIRST, "\\\\
".join(uncommented_lines))
            else:  # Comment
                commented_lines = ["# " + line for line in lines]
                self.text_area.delete(tk.SEL_FIRST, tk.SEL_LAST)
                self.text_area.insert(tk.SEL_FIRST, "\\\\
".join(commented_lines))
            self.update_line_numbers()
        except tk.TclError:
            pass  # No selection

    def run_code(self):
        self.console.delete("1.0", tk.END)  # Clear console before running
        code = self.text_area.get("1.0", tk.END)
        try:
            # Redirect stdout to console
            import sys
            old_stdout = sys.stdout
            sys.stdout = self  # Set the IDE as the stdout
            exec(code)
            sys.stdout = old_stdout  # Restore stdout
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def write(self, text):
        # Override write method for redirecting stdout
        self.console.insert(tk.END, text)
        self.console.see(tk.END)  # Scroll to the end

    def increase_font_size(self):
        current_size = self.font.cget("size")
        self.font.configure(size=current_size + 2)
        self.update_line_numbers()

    def decrease_font_size(self):
        current_size = self.font.cget("size")
        if current_size > 2:
            self.font.configure(size=current_size - 2)
        self.update_line_numbers()

    def change_theme(self, theme):
        if theme == "light":
            self.text_area.config(bg="white", fg="black", insertbackground="black")
            self.console.config(bg="white", fg="black", insertbackground="black")
            self.line_numbers.config(bg="lightgrey", fg="black")
        elif theme == "dark":
            self.text_area.config(bg="#272822", fg="#f8f8f2", insertbackground="white")
            self.console.config(bg="black", fg="white", insertbackground="white")
            self.line_numbers.config(bg="#272822", fg="#f8f8f2")

    def smart_indent(self, event):
        current_line = self.text_area.get("insert linestart", "insert lineend")
        current_indent = len(current_line) - len(current_line.lstrip())

        if current_line.rstrip().endswith(":"):
            new_indent = current_indent + 4
        elif current_line.lstrip().startswith(("return", "pass", "break", "continue")):
            new_indent = current_indent - 4
        else:
            # Check previous line for indentation context
            previous_line = self.text_area.get("insert -1l linestart", "insert -1l lineend")
            if previous_line.rstrip().endswith(":"):
                new_indent = current_indent + 4
            else:
                new_indent = current_indent

        self.text_area.insert("insert", "\\
" + " " * new_indent)
        self.update_line_numbers()
        return "break"  # Prevent default newline behavior

    def handle_open_bracket(self, event):
        if event.char in ['{', '[', '(']:
            self.text_area.insert("insert", event.char)
            self.text_area.insert("insert", self.get_closing_bracket(event.char))
            self.text_area.mark_set("insert", "insert-1c")

    def get_closing_bracket(self, open_bracket):
        if open_bracket == '{':
            return '}'
        elif open_bracket == '[':
            return ']'
        elif open_bracket == '(':
            return ')'
        else:
            return ''

    def handle_backspace(self, event):
        if self.text_area.get("insert-1c") == self.text_area.get("insert") and self.text_area.get("insert") in ["}", "]", ")"]:
            self.text_area.delete("insert-1c")

    def update_line_numbers(self, event=None):
        self.line_numbers.config(state="normal")
        self.line_numbers.delete("1.0", tk.END)
        line_count = int(self.text_area.index("end-1c").split('.')[0])
        for i in range(1, line_count + 1):
            self.line_numbers.insert(tk.END, f"{i}
")
        self.line_numbers.config(state="disabled")


root = tk.Tk()
ide = SmartphoneIDE(root)
root.mainloop()
