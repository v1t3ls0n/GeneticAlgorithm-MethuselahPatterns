#!/bin/bash
# Confirm the action with the user
echo "This script will delete all files and directories ignored by Git (as per .gitignore)."
read -p "Are you sure you want to proceed? This action is irreversible! (y/N): " confirm

if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Operation canceled."
  exit 0
fi

# Check if the repository is a Git repository
if [ ! -d ".git" ]; then
  echo "Error: This is not a Git repository."
  exit 1
fi

# Perform a dry run first to show what will be deleted
echo "Performing a dry run to show files that will be deleted..."
git clean -ndX

# Ask for confirmation again after dry run
read -p "Do you want to delete the above files and directories? (y/N): " confirm_final

if [[ "$confirm_final" != "y" && "$confirm_final" != "Y" ]]; then
  echo "Operation canceled."
  exit 0
fi

# Delete all files and directories ignored by Git
echo "Deleting ignored files and directories..."
git clean -fdX

# Check if the operation was successful
if [ $? -eq 0 ]; then
  echo "Successfully deleted all ignored files and directories."
else
  echo "Error: Failed to delete some files or directories."
  exit 1
fi
