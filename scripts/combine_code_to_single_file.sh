#!/bin/bash

# Define the output file
output_file="merged_script.py"

# Check if the output file already exists and remove it
if [ -f "$output_file" ]; then
    echo "Removing existing $output_file..."
    rm "$output_file"
fi

# Loop through all .py files in the root folder
for py_file in *.py; do
    # Skip the output file if it exists in the same directory
    if [ "$py_file" == "$output_file" ]; then
        continue
    fi

    # Add a comment to indicate the source file
    echo "Appending $py_file to $output_file..."
    echo -e "\n# --- START OF $py_file ---\n" >> "$output_file"
    cat "$py_file" >> "$output_file"
    echo -e "\n# --- END OF $py_file ---\n" >> "$output_file"
done

echo "All .py files have been merged into $output_file"
