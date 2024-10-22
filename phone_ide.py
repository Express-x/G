import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QMessageBox,
                             QTabWidget, QFileDialog, QCompleter, QLineEdit,
                             QDialog, QLabel, QCheckBox)
from PySide6.QtGui import (QFontDatabase, QFont, QSyntaxHighlighter, 
                          QTextCharFormat, QTextCursor, QKeySequence)
from PySide6.QtCore import Qt, QRegularExpression, Signal
from PySide6.QtCore import QCoreApplication
import keyword
import io
import contextlib
import jedi


class FindReplaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find & Replace")
        layout = QVBoxLayout(self)
        
        # Find section
        self.find_input = QLineEdit()
        self.find_input.setPlaceholderText("Find text...")
        layout.addWidget(self.find_input)
        
        # Replace section
        self.replace_input = QLineEdit()
        self.replace_input.setPlaceholderText("Replace with...")
        layout.addWidget(self.replace_input)
        
        # Case sensitive option
        self.case_sensitive = QCheckBox("Case sensitive")
        layout.addWidget(self.case_sensitive)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.find_button = QPushButton("Find Next")
        self.replace_button = QPushButton("Replace")
        self.replace_all_button = QPushButton("Replace All")
        
        button_layout.addWidget(self.find_button)
        button_layout.addWidget(self.replace_button)
        button_layout.addWidget(self.replace_all_button)
        layout.addLayout(button_layout)

class CodeEditor(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.completer = None
        self.setup_completer()
        
    def setup_completer(self):
        # Create completer with Python keywords and builtins
        words = keyword.kwlist + dir(__builtins__)
        self.completer = QCompleter(words)
        self.completer.setWidget(self)
        self.completer.activated.connect(self.insert_completion)
        
    def insert_completion(self, completion):
        tc = self.textCursor()
        tc.movePosition(QTextCursor.Left)
        tc.movePosition(QTextCursor.EndOfWord)
        tc.insertText(completion[len(self.word_prefix):])
        self.setTextCursor(tc)
        
    def keyPressEvent(self, event):
        if self.completer and self.completer.popup().isVisible():
            if event.key() in (Qt.Key_Enter, Qt.Key_Return, Qt.Key_Escape, Qt.Key_Tab):
                event.ignore()
                return
                
        super().keyPressEvent(event)
        
        if event.key() == Qt.Key_Space and event.modifiers() == Qt.ControlModifier:
            self.show_completions()
            
    def show_completions(self):
        tc = self.textCursor()
        tc.select(QTextCursor.WordUnderCursor)
        self.word_prefix = tc.selectedText()
        
        if self.word_prefix:
            script = jedi.Script(code=self.toPlainText(), path='')
            completions = script.complete(self.textCursor().blockNumber() + 1,
                                       self.textCursor().columnNumber())
            suggestions = [c.name for c in completions]
            
            if suggestions:
                model = QStringListModel(suggestions)
                self.completer.setModel(model)
                rect = self.cursorRect()
                rect.setWidth(self.completer.popup().sizeHintForColumn(0)
                            + self.completer.popup().verticalScrollBar().sizeHint().width())
                self.completer.complete(rect)

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None, dark_mode=False):
        super().__init__(parent)
        self.dark_mode = dark_mode
        self.setup_highlighting_rules()
        
    def setup_highlighting_rules(self):
        self.highlighting_rules = []
        
        # Colors based on theme
        if self.dark_mode:
            keyword_color = Qt.cyan
            string_color = Qt.green
            function_color = Qt.yellow
            comment_color = Qt.gray
        else:
            keyword_color = Qt.blue
            string_color = Qt.darkGreen
            function_color = Qt.darkCyan
            comment_color = Qt.darkGray
        
        # Keyword format
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(keyword_color)
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [f'\\b{w}\\b' for w in keyword.kwlist]
        for pattern in keywords:
            self.highlighting_rules.append((
                QRegularExpression(pattern),
                keyword_format
            ))
        
        # String format
        string_format = QTextCharFormat()
        string_format.setForeground(string_color)
        self.highlighting_rules.extend([
            (QRegularExpression("'.*'"), string_format),
            (QRegularExpression('".*"'), string_format),
        ])
        
        # Function format
        function_format = QTextCharFormat()
        function_format.setForeground(function_color)
        self.highlighting_rules.append((
            QRegularExpression("\\b[A-Za-z0-9_]+(?=\\()"),
            function_format
        ))
        
        # Comment format
        comment_format = QTextCharFormat()
        comment_format.setForeground(comment_color)
        self.highlighting_rules.append((
            QRegularExpression("#.*"),
            comment_format
        ))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = pattern
            it = expression.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)

class CodeTab(QWidget):
    def __init__(self, parent=None, dark_mode=False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        self.code_editor = CodeEditor()
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setMaximumHeight(200)
        
        font = QFont("Courier New", 14)
        self.code_editor.setFont(font)
        self.output_display.setFont(font)
        
        self.highlighter = PythonHighlighter(self.code_editor.document(), dark_mode)
        
        layout.addWidget(self.code_editor)
        layout.addWidget(self.output_display)
        
        self.filepath = None

class MobilePythonIDE(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mobile Python IDE")
        self.dark_mode = False
        self.menu_panel_visible = False

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Menu button
        self.menu_button = QPushButton("☰")  # Three parallel lines
        self.menu_button.setFixedSize(40, 40)
        self.menu_button.clicked.connect(self.toggle_menu_panel)
        layout.addWidget(self.menu_button)

        # Create menu panel
        self.menu_panel = QWidget()
        self.menu_panel.setLayout(QVBoxLayout())
        self.menu_panel.setVisible(False)  # Initially hidden
        layout.addWidget(self.menu_panel)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        layout.addWidget(self.tab_widget)


        # Create initial tab
        self.new_file()

        self.apply_theme()

    def toggle_menu_panel(self):
        self.menu_panel_visible = not self.menu_panel_visible
        self.menu_panel.setVisible(self.menu_panel_visible)

    def get_button_style(self):
        if self.dark_mode:
            return """
                QPushButton {
                    background-color: #2c2c2c;
                    color: white;
                    border: none;
                    padding: 10px;
                    font-size: 14px;
                    margin: 2px;
                    border-radius: 5px;
                }
                QPushButton:pressed {
                    background-color: #1f1f1f;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 10px;
                    font-size: 14px;
                    margin: 2px;
                    border-radius: 5px;
                }
                QPushButton:pressed {
                    background-color: #45a049;
                }
            """

    def new_file(self):
        tab = CodeTab(dark_mode=self.dark_mode)
        self.tab_widget.addTab(tab, "Untitled")
        self.tab_widget.setCurrentWidget(tab)

    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Python Files (*.py)")
        if filepath:
            with open(filepath, 'r') as file:
                content = file.read()
                tab = CodeTab(dark_mode=self.dark_mode)
                tab.code_editor.setText(content)
                tab.filepath = filepath
                self.tab_widget.addTab(tab, os.path.basename(filepath))
                self.tab_widget.setCurrentWidget(tab)

    def save_file(self):
        current_tab = self.tab_widget.currentWidget()
        if not current_tab:
            return

        if not current_tab.filepath:
            filepath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Python Files (*.py)")
            if not filepath:
                return
            current_tab.filepath = filepath

        with open(current_tab.filepath, 'w') as file:
            file.write(current_tab.code_editor.toPlainText())
            self.tab_widget.setTabText(self.tab_widget.currentIndex(),
                                     os.path.basename(current_tab.filepath))

    def close_tab(self, index):
        if self.tab_widget.count() > 1:
            self.tab_widget.removeTab(index)

    def run_code(self):
        current_tab = self.tab_widget.currentWidget()
        if not current_tab:
            return

        code = current_tab.code_editor.toPlainText()
        output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output):
                exec(code)
            current_tab.output_display.setText(output.getvalue())
        except Exception as e:
            current_tab.output_display.setText(f"Error:\n{str(e)}")

    def show_find_replace(self):
        dialog = FindReplaceDialog(self)
        current_tab = self.tab_widget.currentWidget()

        def find_next():
            text = dialog.find_input.text()
            if not text:
                return

            flags = QTextDocument.FindFlags()
            if dialog.case_sensitive.isChecked():
                flags |= QTextDocument.FindCaseSensitively

            if not current_tab.code_editor.find(text, flags):
                cursor = current_tab.code_editor.textCursor()
                cursor.movePosition(QTextCursor.Start)
                current_tab.code_editor.setTextCursor(cursor)
                current_tab.code_editor.find(text, flags)

        def replace():
            if current_tab.code_editor.textCursor().hasSelection():
                current_tab.code_editor.insertPlainText(dialog.replace_input.text())
                find_next()

        def replace_all():
            text = dialog.find_input.text()
            replace_text = dialog.replace_input.text()
            if not text:
                return

            cursor = current_tab.code_editor.textCursor()
            cursor.beginEditBlock()

            cursor.movePosition(QTextCursor.Start)
            current_tab.code_editor.setTextCursor(cursor)

            flags = QTextDocument.FindFlags()
            if dialog.case_sensitive.isChecked():
                flags |= QTextDocument.FindCaseSensitively

            while current_tab.code_editor.find(text, flags):
                current_tab.code_editor.insertPlainText(replace_text)

            cursor.endEditBlock()

        dialog.find_button.clicked.connect(find_next)
        dialog.replace_button.clicked.connect(replace)
        dialog.replace_all_button.clicked.connect(replace_all)

        dialog.exec_()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def apply_theme(self):
        if self.dark_mode:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #1e1e1e;
                    color: #ffffff;
                }
                QTextEdit {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    border: 1px solid #3d3d3d;
                    border-radius: 5px;
                    padding: 8px;
                }
                QTabWidget::pane {
                    border: 1px solid #3d3d3d;
                }
                QTabBar::tab {
                    background-color: #2d2d2d;
                    color: #ffffff;
                    padding: 8px;
                    margin: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #3d3d3d;
                }
            """)
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #f0f0f0;
                    color: #000000;
                }
                QTextEdit {
                    background-color: white;
                    color: black;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 8px;
                }
                QTabWidget::pane {
                    border: 1px solid #ccc;
                }
                QTabBar::tab {
                    background-color: #e1e1e1;
                    color: #000000;
                    padding: 8px;
                    margin: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #ffffff;
                }
            """)
        
        # Recreate all tabs with new theme
        current_tabs = []
        for i in range(self.tab_widget.count()):
            tab = self.tab_widget.widget(i)
            current_tabs.append({
                'content': tab.code_editor.toPlainText(),
                'filepath': tab.filepath,
                'name': self.tab_widget.tabText(i),
                'output': tab.output_display.toPlainText()
            })
        
        # Clear and recreate tabs
        while self.tab_widget.count() > 0:
            self.tab_widget.removeTab(0)
            
        for tab_data in current_tabs:
            tab = CodeTab(dark_mode=self.dark_mode)
            tab.code_editor.setText(tab_data['content'])
            tab.filepath = tab_data['filepath']
            tab.output_display.setText(tab_data['output'])
            self.tab_widget.addTab(tab, tab_data['name'])
        
        # Update all buttons
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(self.get_button_style())

def main():
    app = QApplication(sys.argv)
    
    # Modern approach to handle high DPI screens
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
    # Set default font
    default_font = QFont("Courier New", 14)
    app.setFont(default_font)
    
    # Create and show IDE
    ide = MobilePythonIDE()
    
    # Add keyboard shortcuts
    
    ide.showMaximized()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
