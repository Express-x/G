follow the documentation snd write the code fir the features 
### NewFeatures for the Python IDE Using Tkinter

In Version 2 of the Python IDE, additional functionality will be introduced to enhance usability, improve the user interface, and provide more advanced features that are common in modern code editors. The following document outlines these new features, their locations, styling, behaviors, and their implementation details.

### 1. **Syntax Highlighting**
- **Description**: Python keywords, strings, comments, and other language-specific constructs will be color-coded in the editor to enhance readability and help users easily identify different code components.
  
  ##### Location:
  - This feature will apply directly within the **Text Editor Area**.
    
    ##### Styling:
    - **Keywords**: Blue, bold.
    - **Comments**: Green, italic.
    - **Strings**: Red.
    - **Numbers**: Purple.
      
      ##### Behavior:
      - As the user types, the editor will automatically highlight Python syntax. This should be dynamic, updating the colors of keywords and symbols in real-time.

      ##### Implementation Details:
      - Use the `tag_configure` method of the `Text()` widget to define styles for keywords, comments, strings, etc.
      - Use regular expressions to identify specific patterns (like keywords, comments, and strings) and apply the appropriate tag to color-code them.
      - Implement a function to run whenever text is inserted or modified in the editor to update the syntax highlighting.

      ---

      ### 2. **Auto-Completion**
      - **Description**: Provide suggestions to the user as they type, offering possible completions for Python keywords, variables, and functions.
        
        ##### Location:
        - This feature will be integrated into the **Text Editor Area**.
          
          ##### Behavior:
          - When the user starts typing a keyword or function, a dropdown with possible completions should appear. The user can select from the list using the arrow keys and press `Enter` to autocomplete.
            
            ##### Implementation Details:
            - Create a list of Python keywords and built-in functions.
            - Track user input and dynamically display suggestions using a small popup or dropdown widget near the cursor position.
            - On selecting a suggestion and pressing `Enter`, automatically insert the completed word or function.
              
              ---

              ### 3. **Error Highlighting**
              - **Description**: When the user runs the script and encounters syntax or runtime errors, highlight the line(s) in the editor where the error occurred.
                
                ##### Location:
                - The **Text Editor Area** will be used for error highlighting.
                  
                  ##### Styling:
                  - **Error Line Background Color**: Light red.
                    
                    ##### Behavior:
                    - After executing the script, if an error is found, the IDE should jump to the line where the error occurred and highlight it with a light red background.
                      
                      ##### Implementation Details:
                      - Use `subprocess` to capture the error message along with the line number where the error occurred.
                      - Scroll the editor to the specified line and highlight it using `tag_configure`.

                      ---

                      ### 4. **Code Folding**
                      - **Description**: Allow users to collapse and expand sections of code, such as functions or classes, to improve code navigation and focus on specific areas.
                        
                        ##### Location:
                        - Code folding icons (small triangles) should appear next to the **Text Editor Area**, aligned with lines that begin a function, class, or loop block.
                          
                          ##### Behavior:
                          - When the user clicks the triangle next to a block of code (e.g., a function or class), that section of code will collapse and only the first line will remain visible. Clicking the triangle again will expand the code.
                            
                            ##### Implementation Details:
                            - Use regular expressions to detect functions, classes, and block-level code.
                            - Attach clickable buttons (like triangles) next to these blocks that trigger the hiding or showing of certain lines within the editor.
                              
                              ---

                              ### 5. **Find and Replace**
                              - **Description**: Provide a dialog box where the user can search for specific text in the code and optionally replace it with another string.
                                
                                ##### Location:
                                - The **Find and Replace** option will be placed under the `Edit` menu in the **Menu Bar**.
                                  
                                  ##### Behavior:
                                  - When selected, a pop-up dialog will appear with two input fields (`Find` and `Replace`). The user can enter the text they want to find and replace it with another string.
                                  - The IDE will highlight all occurrences of the search term in the editor, and the user can replace one or all occurrences.
                                    
                                    ##### Implementation Details:
                                    - Create a new `Toplevel()` window for the dialog box.
                                    - Use the `find` method of the `Text()` widget to locate the search term and highlight all matches.
                                    - Provide buttons for `Find`, `Replace`, and `Replace All` functionalities.

                                    ---

                                    ### 6. **Line Numbers**
                                    - **Description**: Display line numbers beside the text editor to help users keep track of their code’s structure and location.
                                      
                                      ##### Location:
                                      - Line numbers should appear on the **left side of the Text Editor Area**, in a fixed-width region.
                                        
                                        ##### Styling:
                                        - **Font**: Courier, size 12 (smaller than the code font for distinction).
                                        - **Background Color**: Light grey.
                                          
                                          ##### Behavior:
                                          - Line numbers should update dynamically as the user scrolls or adds/deletes lines in the editor.
                                            
                                            ##### Implementation Details:
                                            - Create a separate widget or a `Canvas` to the left of the text editor to display the line numbers.
                                            - Use the `yview` method of the `Text()` widget to synchronize the scrolling of the line numbers with the editor.
                                            - Update the line numbers whenever the content of the editor changes.

                                            ---

                                            ### 7. **Themes (Dark/Light Mode)**
                                            - **Description**: Provide users the ability to switch between a light and dark theme for the IDE.
                                              
                                              ##### Location:
                                              - The `View` menu in the **Menu Bar** will contain options for switching between themes (e.g., `Dark Mode`, `Light Mode`).
                                                
                                                ##### Styling:
                                                - **Light Mode**: 
                                                  - Background: White.
                                                    - Text: Black.
                                                    - **Dark Mode**:
                                                      - Background: Black.
                                                        - Text: White.
                                                          
                                                          ##### Behavior:
                                                          - When the user selects a theme, the colors of the text editor, console, and menu bar will change accordingly.
                                                            
                                                            ##### Implementation Details:
                                                            - Use a variable to track the current theme.
                                                            - When the user switches the theme, update the `bg` and `fg` properties of the `Text()` widget, console, and menu.
                                                            - Save the user's theme preference to a config file and apply it on startup.

                                                            ---

                                                            ### 8. **Advanced File Management (Tabs)**
                                                            - **Description**: Allow users to open multiple files at once, each in its own tab, so they can work on multiple scripts simultaneously.
                                                              
                                                              ##### Location:
                                                              - Tabs should appear at the **top of the Text Editor Area**, just below the Menu Bar.
                                                                
                                                                ##### Behavior:
                                                                - Each tab should represent an open file. Clicking a tab should switch the editor view to display the content of the corresponding file.
                                                                - The user should be able to open a new tab (via `File -> New Tab`) or close an existing tab.
                                                                  
                                                                  ##### Implementation Details:
                                                                  - Use `ttk.Notebook` to implement tabs.
                                                                  - Each tab will contain a separate instance of the `Text()` widget (text editor) so that each file can be edited independently.
                                                                  - Handle tab management such as opening, closing, and switching between tabs.

                                                                  ---

                                                                  ### 9. **Status Bar**
                                                                  - **Description**: A small bar at the bottom of the window to display useful information such as the current line and column of the cursor, the active Python environment, and file status (saved/unsaved).
                                                                    
                                                                    ##### Location:
                                                                    - The **Status Bar** should appear at the bottom of the window, below the output console.
                                                                      
                                                                      ##### Styling:
                                                                      - **Font**: Arial, size 10.
                                                                      - **Background Color**: Light grey.
                                                                        
                                                                        ##### Behavior:
                                                                        - The status bar should update in real-time as the user types, moves the cursor, or runs the script.
                                                                          
                                                                          ##### Implementation Details:
                                                                          - Create a `Label` widget and pack it at the bottom of the window.
                                                                          - Update the line, column, and file status dynamically using the `insert` event from the text editor.

                                                                          ---

                                                                          ### 10. **Virtual Environment Management**
                                                                          - **Description**: Allow users to select and manage Python virtual environments directly from the IDE.
                                                                            
                                                                            ##### Location:
                                                                            - A new menu `Environment` in the **Menu Bar** will contain options for selecting, creating, and managing virtual environments.
                                                                              
                                                                              ##### Behavior:
                                                                              - The user can select an active Python environment from a list of available virtual environments.
                                                                              - The output console will reflect the selected environment when running Python scripts.
                                                                                
                                                                                ##### Implementation Details:
                                                                                - Use the `venv` or `virtualenv` Python package to list and manage environments.
                                                                                - Store the path to the selected environment and use it when running the code via `subprocess`.

                                                                                ---

                                                                                These new features for Version 2 significantly improve the Python IDE's functionality, making it more powerful and user-friendly for code development. Each feature is designed to improve code readability, navigation, error handling, and overall productivity in the IDE.