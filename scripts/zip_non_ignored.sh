#!/bin/bash
# Define the output zip file name
OUTPUT_ZIP="mmn11_by_guy_vitelson.zip"

# Check if the repository is a Git repository
if [ ! -d ".git" ]; then
  echo "Error: This is not a Git repository."
  exit 1
fi

# Create a zip file of all files that are tracked by Git (not ignored)
echo "Creating a zip file excluding files from .gitignore..."
git archive --format=zip --output="$OUTPUT_ZIP" HEAD

# Check if the zip file was created successfully
if [ $? -eq 0 ]; then
  echo "Zip file created successfully: $OUTPUT_ZIP"
else
  echo "Error: Failed to create zip file."
  exit 1
fi
