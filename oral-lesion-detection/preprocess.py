import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image by resizing and normalizing.

    Parameters:
    - image_path (str): Path to the image file.
    - target_size (tuple): Desired image size (width, height).

    Returns:
    - Preprocessed image as a numpy array.
    """
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Unable to load image {image_path}")
        return None

    # Resize image
    image = cv2.resize(image, target_size)

    # Convert image to float32 and Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image

def save_preprocessed_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Preprocesses all images in a directory and saves them to an output directory.

    Parameters:
    - input_dir (str): Directory containing input images (organized by disease).
    - output_dir (str): Directory to save preprocessed images.
    - target_size (tuple): Desired image size (width, height).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all disease folders in the input directory
    for disease_folder in os.listdir(input_dir):
        input_subdir = os.path.join(input_dir, disease_folder)
        output_subdir = os.path.join(output_dir, disease_folder)
        
        if os.path.isdir(input_subdir):
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            for filename in os.listdir(input_subdir):
                image_path = os.path.join(input_subdir, filename)
                if os.path.isfile(image_path):
                    preprocessed_image = preprocess_image(image_path, target_size)
                    if preprocessed_image is not None:
                        output_path = os.path.join(output_subdir, filename)
                        # Save preprocessed image
                        cv2.imwrite(output_path, (preprocessed_image * 255).astype(np.uint8))

def preprocess_all_images(base_input_dir, base_output_dir, target_size=(224, 224)):
    """
    Preprocesses images from training and validation directories.

    Parameters:
    - base_input_dir (str): Base directory containing 'train' and 'validation' subdirectories.
    - base_output_dir (str): Base directory to save preprocessed images.
    - target_size (tuple): Desired image size (width, height).
    """
    for subset in ['train', 'validation']:
        input_dir = os.path.join(base_input_dir, subset)
        output_dir = os.path.join(base_output_dir, subset)
        if os.path.exists(input_dir):
            save_preprocessed_images(input_dir, output_dir, target_size)
        else:
            print(f"Warning: {input_dir} does not exist.")

# Example usage
if __name__ == "__main__":
    base_input_directory = 'data'
    base_output_directory = 'data/preprocessed'
    preprocess_all_images(base_input_directory, base_output_directory)
