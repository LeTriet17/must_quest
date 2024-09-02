import streamlit as st
import base64
import os
from io import BytesIO
import re

def remove_dashes_not_in_table(markdown_text):
    # This regex finds tables by looking for lines that are part of a markdown table structure
    table_regex = re.compile(r'(\|.*\|(\r?\n|\r))')
    
    def table_preserve(match):
        # We need to preserve these lines as they are part of a table
        return match.group(0)
    
    # Split the markdown text into lines
    lines = markdown_text.splitlines(keepends=True)
    
    # Flag to check if we are inside a table
    inside_table = False
    result_lines = []
    
    for line in lines:
        if '|' in line:
            # Check if it's a table line
            if re.match(table_regex, line):
                inside_table = True
                result_lines.append(line)
            else:
                inside_table = False
                # Remove --- if it's not part of a table
                cleaned_line = re.sub(r'---', '', line)
                result_lines.append(cleaned_line)
        else:
            # If we are not in a table, remove ---
            if not inside_table:
                cleaned_line = re.sub(r'---', '', line)
                result_lines.append(cleaned_line)
            else:
                result_lines.append(line)
                inside_table = False

    return ''.join(result_lines)

# Function to process and escape content
def process_content(content):
    # if content:
    #     return content.replace("$", "").replace("\\", "")
    # return ""
    return content

def format_markdown(md_content):
    # Convert headers to plain text or bold text for easier reading
    def replace_headers(match):
        header_level = len(match.group(1))
        header_text = match.group(2).strip()
        if header_level == 1:
            # Optionally, use bold for H1 headers
            return f'**{header_text}**\n'
        else:
            return f'{header_text}\n'

    # Replace headers with the custom formatting
    formatted_md = re.sub(r'^(#{1,6})\s*(.*)', replace_headers, md_content, flags=re.MULTILINE)
    
    formatted_md = remove_dashes_not_in_table(formatted_md)
    # Replace code blocks with a more readable format
    formatted_md = re.sub(r'```', '', formatted_md)
    return formatted_md

def encode_file_to_base64(file):
    # Convert the file contents to Base64 encoding
    buffer = BytesIO()
    buffer.write(file.read())
    return base64.b64encode(buffer.getvalue()).decode()


def process_files(files):
    result = {"videos": [], "images": [], "audios": []}
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()

        # Process according to file extension
        if file_extension in ['.mp4', '.avi']:
            # Video file processing
            video_base64 = encode_file_to_base64(file)
            result["videos"].append(video_base64)
        elif file_extension in ['.jpg', '.png', '.jpeg']:
            # Image file processing
            image_base64 = encode_file_to_base64(file)
            result["images"].append(image_base64)
        elif file_extension in ['.mp3', '.wav', '.ogg', '.flac']:
            # Audio file processing
            audio_base64 = encode_file_to_base64(file)
            result["audios"].append(audio_base64)

    return result
