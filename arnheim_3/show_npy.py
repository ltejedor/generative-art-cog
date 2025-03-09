import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # For saving images

# Load the .npy file
file_path = "animals.npy"  # Update the path if necessary
images = np.load(file_path, allow_pickle=True)

# Check the shape and type of the loaded data
print(f"Loaded data type: {type(images)}")
print(f"Number of images: {len(images)}")

# Create a directory to save images
save_dir = "saved_images"
os.makedirs(save_dir, exist_ok=True)

# Process and save images
for i, img in enumerate(images):
    
    # Remove the alpha channel if it exists
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    
    # Define the file path for saving
    save_path = os.path.join(save_dir, f"image_{i}.png")
    
    # Save the image
    imageio.imwrite(save_path, img)
    print(f"Saved: {save_path}")

print("All images have been saved successfully.")