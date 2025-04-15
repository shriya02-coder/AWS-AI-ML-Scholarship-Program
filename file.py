import streamlit as st
import yaml
import difflib
import os
import re
from io import StringIO

st.set_page_config(page_title="YML File Comparator", layout="wide")

# Dark mode toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Files in session state
if "files" not in st.session_state:
    st.session_state.files = {}

# Current comparison files
if "comparison" not in st.session_state:
    st.session_state.comparison = [None, None]

# Exclusion patterns
if "exclusions" not in st.session_state:
    st.session_state.exclusions = []

# Apply custom CSS for dark mode
def apply_custom_css():
    dark_mode_css = """
    <style>
        .dark-mode {
            background-color: #121212;
            color: #e0e0e0;
        }
        .dark-mode textarea {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border-color: #333333;
        }
        .dark-mode .diff-added {
            background-color: #103410;
            color: #a0ffa0;
        }
        .dark-mode .diff-removed {
            background-color: #341010;
            color: #ffa0a0;
        }
        .dark-mode .diff-normal {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        .light-mode .diff-added {
            background-color: #e6ffe6;
            color: #006600;
        }
        .light-mode .diff-removed {
            background-color: #ffe6e6;
            color: #660000;
        }
        .light-mode .diff-normal {
            background-color: white;
            color: black;
        }
    </style>
    """
    st.markdown(dark_mode_css, unsafe_allow_html=True)

# Function to parse YML file content
def parse_yml(content):
    try:
        return yaml.safe_load(content)
    except Exception as e:
        st.error(f"Error parsing YML: {str(e)}")
        return None

# Function to apply exclusions to a line
def apply_exclusions(line, exclusions):
    modified_line = line
    for exclusion in exclusions:
        if exclusion.strip():
            modified_line = re.sub(re.escape(exclusion), "[EXCLUDED]", modified_line)
    return modified_line

# Function to compare two files line by line with exclusions
def compare_files(content1, content2, exclusions):
    lines1 = content1.split('\n')
    lines2 = content2.split('\n')
    
    # Apply exclusions for comparison
    comp_lines1 = [apply_exclusions(line, exclusions) for line in lines1]
    comp_lines2 = [apply_exclusions(line, exclusions) for line in lines2]
    
    differ = difflib.Differ()
    diff = list(differ.compare(comp_lines1, comp_lines2))
    
    result1 = []
    result2 = []
    for line in diff:
        if line.startswith('  '):  # Common line
            result1.append(('normal', lines1[comp_lines1.index(line[2:])]))
            result2.append(('normal', lines2[comp_lines2.index(line[2:])]))
        elif line.startswith('- '):  # Only in first file
            try:
                result1.append(('removed', lines1[comp_lines1.index(line[2:])]))
            except ValueError:
                result1.append(('removed', line[2:]))  # Fallback if not found
        elif line.startswith('+ '):  # Only in second file
            try:
                result2.append(('added', lines2[comp_lines2.index(line[2:])]))
            except ValueError:
                result2.append(('added', line[2:]))  # Fallback if not found
    
    return result1, result2

def render_diff_line(line_type, line, index, side):
    if st.session_state.dark_mode:
        mode_class = "dark-mode"
    else:
        mode_class = "light-mode"
    
    if line_type == 'normal':
        css_class = "diff-normal"
    elif line_type == 'added':
        css_class = "diff-added"
    else:
        css_class = "diff-removed"
    
    html = f'<div class="{mode_class} {css_class}" style="padding: 2px; margin: 1px 0; font-family: monospace; white-space: pre;">'
    html += f'{line}'
    html += '</div>'
    
    # Add merge button if applicable
    if (side == 'left' and line_type == 'removed') or (side == 'right' and line_type == 'added'):
        other_side = 'right' if side == 'left' else 'left'
        if st.button(f"Merge to {other_side}", key=f"merge_{side}_{index}"):
            # Implement merge logic here
            st.session_state[f"merge_{side}_to_{other_side}_{index}"] = True
    
    return html

# Main app UI
def main():
    apply_custom_css()
    
    st.title("YML File Comparator")
    
    # Dark mode toggle in sidebar
    with st.sidebar:
        if st.toggle("Dark Mode", value=st.session_state.dark_mode):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # File management section
    with st.expander("File Management", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Files")
            uploaded_files = st.file_uploader("Upload YML files", type=["yml", "yaml"], accept_multiple_files=True)
            
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
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.selectbox("Select first file", 
                            ["None"] + list(st.session_state.files.keys()),
                            index=0 if st.session_state.comparison[0] is None else 
                                  list(st.session_state.files.keys()).index(st.session_state.comparison[0]) + 1)
        if file1 != "None":
            st.session_state.comparison[0] = file1
        else:
            st.session_state.comparison[0] = None
    
    with col2:
        file2 = st.selectbox("Select second file", 
                            ["None"] + list(st.session_state.files.keys()),
                            index=0 if st.session_state.comparison[1] is None else 
                                  list(st.session_state.files.keys()).index(st.session_state.comparison[1]) + 1)
        if file2 != "None":
            st.session_state.comparison[1] = file2
        else:
            st.session_state.comparison[1] = None

    # Exclusion patterns
    with st.expander("Exclusion Settings"):
        st.write("Add terms to exclude during comparison (e.g., environment-specific terms)")
        
        # Dynamic exclusion list
        for i in range(len(st.session_state.exclusions) + 1):
            if i == len(st.session_state.exclusions):
                # Add new exclusion
                new_exclusion = st.text_input("Add new exclusion term", key=f"new_excl_{i}")
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

    # File editing and comparison
    if st.session_state.comparison[0] and st.session_state.comparison[1]:
        st.subheader("File Comparison and Editing")
        
        file1, file2 = st.session_state.comparison
        content1 = st.session_state.files[file1]
        content2 = st.session_state.files[file2]
        
        # File editors
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(file1)
            edited_content1 = st.text_area("Edit content", value=content1, height=200, key="edit1")
            if edited_content1 != content1:
                st.session_state.files[file1] = edited_content1
                content1 = edited_content1
        
        with col2:
            st.subheader(file2)
            edited_content2 = st.text_area("Edit content", value=content2, height=200, key="edit2")
            if edited_content2 != content2:
                st.session_state.files[file2] = edited_content2
                content2 = edited_content2
        
        # Compare files
        if st.button("Compare Files"):
            try:
                diff1, diff2 = compare_files(content1, content2, st.session_state.exclusions)
                
                st.subheader("Comparison Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {file1}")
                    for i, (line_type, line) in enumerate(diff1):
                        st.markdown(render_diff_line(line_type, line, i, 'left'), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"### {file2}")
                    for i, (line_type, line) in enumerate(diff2):
                        st.markdown(render_diff_line(line_type, line, i, 'right'), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during comparison: {str(e)}")
    
    # Extract section to new file
    with st.expander("Extract Section to New File"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            source_file = st.selectbox("Source file", 
                                      ["None"] + list(st.session_state.files.keys()),
                                      key="extract_source")
        
        with col2:
            new_file = st.text_input("New file name", value="extracted.yml", key="extract_dest")
        
        with col3:
            start_line = st.number_input("Start line", min_value=1, value=1, key="extract_start")
            end_line = st.number_input("End line", min_value=1, value=5, key="extract_end")
        
        if st.button("Extract Section") and source_file != "None":
            try:
                lines = st.session_state.files[source_file].split('\n')
                if start_line <= len(lines) and end_line <= len(lines) and start_line <= end_line:
                    extracted_content = '\n'.join(lines[start_line-1:end_line])
                    
                    if new_file not in st.session_state.files:
                        st.session_state.files[new_file] = extracted_content
                    else:
                        st.session_state.files[new_file] += '\n' + extracted_content
                    
                    st.success(f"Extracted lines {start_line}-{end_line} to {new_file}")
                else:
                    st.error("Invalid line numbers for extraction")
            except Exception as e:
                st.error(f"Error during extraction: {str(e)}")

if __name__ == "__main__":
    main()