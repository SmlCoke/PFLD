import os
import shutil
import argparse

def search_and_copy_files(folder_path, search_term, destination_folder):
    """
    Recursively searches for files containing a search term in a folder and copies them to a destination folder.

    Args:
        folder_path (str): The path to the folder to search.
        search_term (str): The string to search for in file names.
        destination_folder (str): The path to the folder where the files will be copied.
    """

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if search_term in file:
                source_path = os.path.join(root, file)
                # Create the destination subfolders if they don't exist
                relative_path = os.path.relpath(root, folder_path)
                destination_subfolder = os.path.join(destination_folder, relative_path)
                if not os.path.exists(destination_subfolder):
                    os.makedirs(destination_subfolder)
                destination_path = os.path.join(destination_subfolder, file)
                try:
                    shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
                    print(f"Copied: {source_path} -> {destination_path}")
                except OSError as e:
                    print(f"Error copying {source_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively search and copy files.")
    parser.add_argument("a", help="Path to the folder to search")
    parser.add_argument("b", help="Search term (part of the file name)")
    parser.add_argument("c", help="Path to the destination folder")

    args = parser.parse_args()

    search_and_copy_files(args.a, args.b, args.c)