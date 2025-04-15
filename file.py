import streamlit as st
import difflib
import os
import re
from io import StringIO
import base64

st.set_page_config(page_title="YML File Comparator", layout="wide")

# Initialize session state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "files" not in st.session_state:
    st.session_state.files = {}
if "comparison" not in st.session_state:
    st.session_state.comparison = [None, None, None]  # Support for three files
if "exclusions" not in st.session_state:
    st.session_state.exclusions = []
if "merge_history" not in st.session_state:
    st.session_state.merge_history = []

# CSS for improved styling including dark mode
def get_css():
    return """
    <style>
        /* General app styling */
        .main {
            background-color: var(--background-color);
            color: var(--text-color);
            transition: all 0.3s ease;
        }

        /* Code editor styling */
        .code-editor {
            font-family: 'Courier New', monospace;
            line-height: 1.4;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background-color: var(--editor-bg);
            color: var(--editor-text);
            width: 100%;
            height: 300px;
            overflow: auto;
            white-space: pre;
            tab-size: 4;
        }

        /* Line highlighting */
        .line {
            display: flex;
            padding: 0 4px;
            margin: 0;
            width: 100%;
            position: relative;
        }
        
        .line:hover {
            background-color: var(--hover-bg);
        }
        
        .line-number {
            user-select: none;
            text-align: right;
            padding-right: 8px;
            margin-right: 8px;
            min-width: 40px;
            color: var(--line-number-color);
            border-right: 1px solid var(--border-color);
        }
        
        .line-content {
            flex-grow: 1;
            white-space: pre;
            display: flex;
            align-items: center;
        }
        
        /* Diff styling */
        .normal {
            background-color: var(--normal-bg);
        }
        
        .added {
            background-color: var(--added-bg);
            color: var(--added-text);
        }
        
        .removed {
            background-color: var(--removed-bg);
            color: var(--removed-text);
        }
        
        .changed {
            background-color: var(--changed-bg);
            color: var(--changed-text);
        }
        
        /* Merge buttons */
        .merge-btn {
            visibility: hidden;
            cursor: pointer;
            background: none;
            border: none;
            color: var(--btn-color);
            font-size: 18px;
            padding: 0 4px;
            margin-left: 8px;
        }
        
        .line:hover .merge-btn {
            visibility: visible;
        }
        
        .merge-btn:hover {
            color: var(--btn-hover);
        }

        /* Theme variables */
        :root {
            /* Light theme */
            --background-color: #ffffff;
            --text-color: #333333;
            --border-color: #dddddd;
            --editor-bg: #f9f9f9;
            --editor-text: #333333;
            --hover-bg: #f0f0f0;
            --line-number-color: #888888;
            --normal-bg: transparent;
            --added-bg: #e6ffed;
            --added-text: #22863a;
            --removed-bg: #ffeef0;
            --removed-text: #cb2431;
            --changed-bg: #fff5b1;
            --changed-text: #735c0f;
            --btn-color: #6c757d;
            --btn-hover: #007bff;
        }

        /* Dark theme */
        .dark-mode {
            --background-color: #1a1a1a;
            --text-color: #f0f0f0;
            --border-color: #444444;
            --editor-bg: #252526;
            --editor-text: #e0e0e0;
            --hover-bg: #333333;
            --line-number-color: #888888;
            --normal-bg: transparent;
            --added-bg: #173b21;
            --added-text: #4dc67d;
            --removed-bg: #3b1c26;
            --removed-text: #f27983;
            --changed-bg: #3a3000;
            --changed-text: #c5b44b;
            --btn-color: #aaaaaa;
            --btn-hover: #3ea6ff;
        }

        /* Additional styles for container */
        .comparison-container {
            display: flex;
            flex-direction: row;
            gap: 10px;
            width: 100%;
            overflow-x: auto;
        }
        
        .file-container {
            flex: 1;
            min-width: 300px;
            display: flex;
            flex-direction: column;
        }
        
        .file-header {
            padding: 8px;
            background-color: var(--editor-bg);
            border: 1px solid var(--border-color);
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            font-weight: bold;
        }
        
        /* Buttons and controls */
        .action-btn {
            margin: 5px;
            padding: 5px 10px;
            border-radius: 4px;
            background-color: var(--btn-color);
            color: var(--text-color);
            border: none;
            cursor: pointer;
        }
        
        .action-btn:hover {
            background-color: var(--btn-hover);
        }
    </style>
    """

# Apply dark mode class based on state
def inject_custom_css():
    dark_class = "dark-mode" if st.session_state.dark_mode else ""
    st.markdown(f"""
    <style>
    .main {{
        {dark_class}
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(get_css(), unsafe_allow_html=True)

# Function to apply exclusions to a line
def apply_exclusions(line, exclusions):
    modified_line = line
    for exclusion in exclusions:
        if exclusion.strip():
            # For complex patterns like URLs or multiple-word exclusions
            modified_line = re.sub(re.escape(exclusion), "[EXCLUDED]", modified_line, flags=re.IGNORECASE)
    return modified_line

# Function to compare multiple files with exclusions
def compare_files(file_contents, exclusions):
    num_files = len(file_contents)
    
    # Process each file's content
    all_lines = [content.split('\n') for content in file_contents]
    
    # Apply exclusions for comparison
    comp_lines = []
    for lines in all_lines:
        comp_lines.append([apply_exclusions(line, exclusions) for line in lines])
    
    results = []
    
    # For two files comparison
    if num_files == 2:
        differ = difflib.Differ()
        diff = list(differ.compare(comp_lines[0], comp_lines[1]))
        
        result1 = []
        result2 = []
        
        for line in diff:
            if line.startswith('  '):  # Common line
                line_text = line[2:]
                try:
                    idx1 = next((i for i, x in enumerate(comp_lines[0]) if x == line_text), None)
                    idx2 = next((i for i, x in enumerate(comp_lines[1]) if x == line_text), None)
                    
                    if idx1 is not None and idx2 is not None:
                        result1.append(('normal', all_lines[0][idx1], idx1))
                        result2.append(('normal', all_lines[1][idx2], idx2))
                except Exception:
                    # Fallback for any issues
                    result1.append(('normal', line_text, -1))
                    result2.append(('normal', line_text, -1))
                    
            elif line.startswith('- '):  # Only in first file
                line_text = line[2:]
                try:
                    idx = next((i for i, x in enumerate(comp_lines[0]) if x == line_text), None)
                    if idx is not None:
                        result1.append(('removed', all_lines[0][idx], idx))
                except Exception:
                    result1.append(('removed', line_text, -1))
                    
            elif line.startswith('+ '):  # Only in second file
                line_text = line[2:]
                try:
                    idx = next((i for i, x in enumerate(comp_lines[1]) if x == line_text), None)
                    if idx is not None:
                        result2.append(('added', all_lines[1][idx], idx))
                except Exception:
                    result2.append(('added', line_text, -1))
        
        results = [result1, result2]
        
    # For three files comparison
    elif num_files == 3:
        # Using SequenceMatcher for more precise analysis
        result1 = []
        result2 = []
        result3 = []
        
        # Compare file 1 with file 2
        sm12 = difflib.SequenceMatcher(None, comp_lines[0], comp_lines[1])
        # Compare file 1 with file 3
        sm13 = difflib.SequenceMatcher(None, comp_lines[0], comp_lines[2])
        # Compare file 2 with file 3
        sm23 = difflib.SequenceMatcher(None, comp_lines[1], comp_lines[2])
        
        # Process all lines from file 1
        for i, line in enumerate(all_lines[0]):
            matching2 = False
            matching3 = False
            
            # Check if line exists in file 2
            for op, i1, i2, j1, j2 in sm12.get_opcodes():
                if op == 'equal' and i1 <= i < i2:
                    matching2 = True
                    break
            
            # Check if line exists in file 3
            for op, i1, i2, j1, j2 in sm13.get_opcodes():
                if op == 'equal' and i1 <= i < i2:
                    matching3 = True
                    break
            
            if matching2 and matching3:
                result1.append(('normal', line, i))
            elif matching2:
                result1.append(('changed', line, i))  # In 1 and 2 but not 3
            elif matching3:
                result1.append(('changed', line, i))  # In 1 and 3 but not 2
            else:
                result1.append(('removed', line, i))  # Only in 1
        
        # Process all lines from file 2
        for i, line in enumerate(all_lines[1]):
            matching1 = False
            matching3 = False
            
            # Check if line exists in file 1
            for op, i1, i2, j1, j2 in sm12.get_opcodes():
                if op == 'equal' and j1 <= i < j2:
                    matching1 = True
                    break
            
            # Check if line exists in file 3
            for op, i1, i2, j1, j2 in sm23.get_opcodes():
                if op == 'equal' and i1 <= i < i2:
                    matching3 = True
                    break
            
            if matching1 and matching3:
                result2.append(('normal', line, i))
            elif matching1:
                result2.append(('changed', line, i))  # In 1 and 2 but not 3
            elif matching3:
                result2.append(('changed', line, i))  # In 2 and 3 but not 1
            else:
                result2.append(('added', line, i))  # Only in 2
        
        # Process all lines from file 3
        for i, line in enumerate(all_lines[2]):
            matching1 = False
            matching2 = False
            
            # Check if line exists in file 1
            for op, i1, i2, j1, j2 in sm13.get_opcodes():
                if op == 'equal' and j1 <= i < j2:
                    matching1 = True
                    break
            
            # Check if line exists in file 2
            for op, i1, i2, j1, j2 in sm23.get_opcodes():
                if op == 'equal' and j1 <= i < j2:
                    matching2 = True
                    break
            
            if matching1 and matching2:
                result3.append(('normal', line, i))
            elif matching1:
                result3.append(('changed', line, i))  # In 1 and 3 but not 2
            elif matching2:
                result3.append(('changed', line, i))  # In 2 and 3 but not 1
            else:
                result3.append(('added', line, i))  # Only in 3
        
        results = [result1, result2, result3]
    
    return results

# Render code editor with line highlighting and merge buttons
def render_code_editor(file_idx, diff_result, file_name, files_content, comparison_files):
    st.markdown(f"""
    <div class="file-container">
        <div class="file-header">{file_name}</div>
        <div class="code-editor" id="editor-{file_idx}">
    """, unsafe_allow_html=True)
    
    for i, (line_type, line_content, line_idx) in enumerate(diff_result):
        # Create merge buttons
        merge_buttons = ""
        for target_idx, target_file in enumerate(comparison_files):
            if target_idx != file_idx and target_file is not None:
                direction = "<<" if target_idx < file_idx else ">>"
                merge_buttons += f"""
                <button class="merge-btn" 
                        onclick="mergeLine('{file_idx}', '{target_idx}', '{i}', '{line_idx}')"
                        title="Merge to {comparison_files[target_idx]}">
                    {direction}
                </button>
                """
        
        # Render line with syntax highlighting
        st.markdown(f"""
        <div class="line {line_type}" id="line-{file_idx}-{i}">
            <div class="line-number">{line_idx + 1 if line_idx >= 0 else ''}</div>
            <div class="line-content">{line_content.replace(' ', '&nbsp;')}{merge_buttons}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add JavaScript for the merge functionality
    st.markdown(f"""
    <script>
    function mergeLine(sourceFileIdx, targetFileIdx, lineNumber, originalLineIdx) {{
        // Create a message to send back to Streamlit
        const data = {{
            source: parseInt(sourceFileIdx),
            target: parseInt(targetFileIdx),
            line: parseInt(lineNumber),
            original_idx: parseInt(originalLineIdx)
        }};
        
        // Use Streamlit's setComponentValue mechanism
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            value: JSON.stringify(data)
        }}, '*');
    }}
    </script>
    """, unsafe_allow_html=True)

# Function to handle merge operations
def process_merge_operations():
    # This will get populated by JavaScript
    merge_data = st.query_params.get('merge_data')
    if merge_data:
        try:
            import json
            data = json.loads(merge_data)
            
            source_idx = data['source']
            target_idx = data['target']
            line_idx = data['line']
            original_idx = data['original_idx']
            
            source_file = st.session_state.comparison[source_idx]
            target_file = st.session_state.comparison[target_idx]
            
            if source_file and target_file:
                # Get the line to merge
                source_content = st.session_state.files[source_file].split('\n')
                if 0 <= original_idx < len(source_content):
                    line_to_merge = source_content[original_idx]
                    
                    # Update the target file
                    target_content = st.session_state.files[target_file].split('\n')
                    # For simplicity, we append to the end
                    target_content.append(line_to_merge)
                    st.session_state.files[target_file] = '\n'.join(target_content)
                    
                    # Record the merge operation
                    st.session_state.merge_history.append({
                        'source': source_file,
                        'target': target_file,
                        'line': line_to_merge
                    })
                    
                    st.rerun()
        except Exception as e:
            st.error(f"Error processing merge: {str(e)}")

# Define the extract section functionality
def extract_section(source_file, new_file, start_line, end_line):
    try:
        if source_file not in st.session_state.files:
            st.error(f"Source file {source_file} not found")
            return False
            
        lines = st.session_state.files[source_file].split('\n')
        if start_line <= 0 or end_line > len(lines) or start_line > end_line:
            st.error("Invalid line range")
            return False
            
        extracted_content = '\n'.join(lines[start_line-1:end_line])
        
        if new_file not in st.session_state.files:
            st.session_state.files[new_file] = extracted_content
        else:
            st.session_state.files[new_file] += '\n' + extracted_content
            
        return True
    except Exception as e:
        st.error(f"Error during extraction: {str(e)}")
        return False

# Main app UI
def main():
    st.title("YML File Comparator")
    
    # Apply custom CSS based on dark mode state
    inject_custom_css()
    
    # Process any merge operations from JavaScript
    process_merge_operations()
    
    # Sidebar with dark mode toggle
    with st.sidebar:
        st.subheader("Settings")
        if st.toggle("Dark Mode", value=st.session_state.dark_mode):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        
        st.divider()
        st.subheader("Recent Merges")
        if st.session_state.merge_history:
            for i, merge in enumerate(st.session_state.merge_history[-5:]):  # Show last 5 merges
                st.write(f"Merged from {merge['source']} to {merge['target']}")
        else:
            st.write("No recent merges")
    
    # File management section
    with st.expander("File Management", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Files")
            uploaded_files = st.file_uploader("Upload YML files", type=["yml"], accept_multiple_files=True)
            
            if uploaded_files:
                for file in uploaded_files:
                    file_content = file.read().decode('utf-8')
                    st.session_state.files[file.name] = file_content
                    st.success(f"Uploaded: {file.name}")
        
        with col2:
            st.subheader("Create New File")
            new_file_name = st.text_input("New file name (with .yml extension)")
            if st.button("Create Empty File") and new_file_name:
                if not new_file_name.endswith('.yml'):
                    new_file_name += '.yml'
                st.session_state.files[new_file_name] = ""
                st.success(f"Created: {new_file_name}")
    
    # File selection for comparison
    st.subheader("Select Files to Compare")
    cols = st.columns(3)
    
    for i in range(3):
        with cols[i]:
            file_options = ["None"] + list(st.session_state.files.keys())
            index = 0
            if st.session_state.comparison[i] is not None:
                try:
                    index = file_options.index(st.session_state.comparison[i])
                except ValueError:
                    index = 0
            
            selected_file = st.selectbox(f"Select file #{i+1}", 
                                        file_options,
                                        index=index,
                                        key=f"file_select_{i}")
            st.session_state.comparison[i] = selected_file if selected_file != "None" else None
    
    # Exclusion patterns
    with st.expander("Exclusion Settings"):
        st.write("Add terms to exclude during comparison (e.g., environment-specific terms, URLs)")
        
        # Dynamic exclusion list
        for i in range(len(st.session_state.exclusions) + 1):
            if i == len(st.session_state.exclusions):
                # Add new exclusion
                new_exclusion = st.text_input("Add new exclusion term or pattern", key=f"new_excl_{i}")
                if new_exclusion:
                    st.session_state.exclusions.append(new_exclusion)
                    st.rerun()
            else:
                # Edit existing exclusion
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.session_state.exclusions[i] = st.text_input("Exclusion term", 
                                                                 value=st.session_state.exclusions[i], 
                                                                 key=f"excl_{i}")
                with col2:
                    if st.button("Remove", key=f"rm_excl_{i}"):
                        st.session_state.exclusions.pop(i)
                        st.rerun()
        
        st.info("Exclusions are applied as case-insensitive patterns. You can exclude URLs, specific terms, or any text pattern.")
    
    # Determine how many files are selected for comparison
    selected_files = [f for f in st.session_state.comparison if f is not None]
    
    # File editing section
    if len(selected_files) > 0:
        st.subheader("File Editing")
        
        for i, file_name in enumerate(selected_files):
            if file_name:
                content = st.session_state.files[file_name]
                st.text_area(f"Edit {file_name}", 
                            value=content,
                            height=200,
                            key=f"edit_{i}")
                
                # Update file content if changed
                edited_content = st.session_state[f"edit_{i}"]
                if edited_content != content:
                    st.session_state.files[file_name] = edited_content
    
    # File comparison
    if len(selected_files) >= 2:
        st.subheader("File Comparison")
        
        if st.button("Compare Files"):
            try:
                # Get content for all selected files
                file_contents = [st.session_state.files[f] for f in selected_files]
                
                # Compare files
                comparison_results = compare_files(file_contents, st.session_state.exclusions)
                
                st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
                
                # Render each file's comparison results
                for i, (file_name, diff_result) in enumerate(zip(selected_files, comparison_results)):
                    render_code_editor(i, diff_result, file_name, file_contents, selected_files)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Extract section to new file
    with st.expander("Extract Section to New File"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            source_file = st.selectbox("Source file", 
                                     ["None"] + list(st.session_state.files.keys()),
                                     key="extract_source")
        
        with col2:
            new_file = st.text_input("New file name", value="extracted.yml", key="extract_dest")
            if not new_file.endswith('.yml'):
                new_file += '.yml'
        
        with col3:
            start_line = st.number_input("Start line", min_value=1, value=1, key="extract_start")
            end_line = st.number_input("End line", min_value=1, value=5, key="extract_end")
        
        if st.button("Extract Section") and source_file != "None":
            if extract_section(source_file, new_file, start_line, end_line):
                st.success(f"Extracted lines {start_line}-{end_line} to {new_file}")

if __name__ == "__main__":
    main()