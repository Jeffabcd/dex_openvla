import os
import shutil

def delete_unwanted_directories(base_path):
    # Iterate over each subject directory (s01, s02, ..., s10)
    for subject_dir in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject_dir)
        
        # Ensure it's a directory
        if os.path.isdir(subject_path):
            # Iterate over each object directory within the subject directory
            for object_dir in os.listdir(subject_path):
                object_path = os.path.join(subject_path, object_dir)
                
                # Ensure it's a directory
                if os.path.isdir(object_path):
                    # Iterate over each subdirectory (0, 1, ..., 8)
                    for sub_dir in os.listdir(object_path):
                        sub_dir_path = os.path.join(object_path, sub_dir)
                        
                        # Delete the directory if it's not '0'
                        if os.path.isdir(sub_dir_path) and sub_dir != '0':
                            print(f"Deleting: {sub_dir_path}")
                            shutil.rmtree(sub_dir_path)

# Define the base path to the data directory
base_path = 'arctic/downloads/data/cropped_images/'

# Call the function to delete unwanted directories
delete_unwanted_directories(base_path)
