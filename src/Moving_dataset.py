import os
import shutil
import random

def pad_dataset_for_batch_size(directory, batch_size):
    for class_folder in ['positive', 'negative']:  # Adjust these names based on your class folders
        class_dir = os.path.join(directory, class_folder)
        files = os.listdir(class_dir)
        num_files = len(files)
        remainder = num_files % batch_size
        
        if remainder != 0:
            files_to_add = batch_size - remainder  # Calculate how many files to add
            sample_files = random.sample(files, files_to_add)  # Randomly select 'files_to_add' number of files as samples

            for i, f in enumerate(sample_files):
                new_file_name = f"copy_{i}_{f}"
                shutil.copy2(os.path.join(class_dir, f), os.path.join(class_dir, new_file_name))
            print(f"Added {files_to_add} copies in {class_dir} to complete the last batch.")

# Configuration: Specify the paths to your dataset directories
train_dir =  '/Users/nishanttiwari/Desktop/farmland_classifier/dataset/train' # Replace with your actual train directory path
validation_dir =  '/Users/nishanttiwari/Desktop/farmland_classifier/dataset/validation'  # Replace with your actual validation directory path
batch_size = 5  # Specify your batch size

# Apply the padding function to both training and validation datasets
pad_dataset_for_batch_size(train_dir, batch_size)
pad_dataset_for_batch_size(validation_dir, batch_size)
