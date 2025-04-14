import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import sys
from tqdm import tqdm

def generate_random_mask(image_size, min_blobs=5, max_blobs=10, max_blob_size=50):
    """
    Generate a random binary mask with random ellipses.

    Parameters:
    - image_size (tuple): Size of the image (height, width)
    - min_blobs (int): Minimum number of blobs to generate
    - max_blobs (int): Maximum number of blobs to generate
    - max_blob_size (int): Maximum size of each blob

    Returns:
    - mask (numpy array): Binary mask with random ellipses
    """
    # Start with a mask of ones (background)
    mask = np.ones(image_size, dtype=np.uint8)
    
    # Determine a random number of blobs within the specified range
    num_blobs = random.randint(min_blobs, max_blobs)
    
    for _ in range(num_blobs):
        center = (random.randint(0, image_size[1]), random.randint(0, image_size[0]))
        axes = (random.randint(10, max_blob_size), random.randint(10, max_blob_size))
        angle = random.randint(0, 180)
        # Draw filled ellipse with value 0 (foreground)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 0, -1)

    # print(mask.mean())
    return mask

def process_images_in_directory(parent_dir):
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith("_cropped.png") and not file.endswith("_seg_cropped.png"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Image not found or unable to read: {image_path}")
                    continue
                
                image_size = image.shape
                larger = max(image_size)
                mask = generate_random_mask(image_size, min_blobs=0, max_blobs=3, max_blob_size=larger//5)
                
                # Ensure the image is the same size as the mask
                image = cv2.resize(image, (image_size[1], image_size[0]))
                
                # Multiply the image with the mask
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                
                # Save the generated mask and masked image
                output_mask_path = os.path.join(root, f"{os.path.splitext(file)[0]}_mask.png")
                output_masked_image_path = os.path.join(root, f"{os.path.splitext(file)[0]}_masked.png")
                plt.imsave(output_mask_path, mask, cmap='gray')
                plt.imsave(output_masked_image_path, masked_image, cmap='gray')
                # print(f"Processed and saved mask and masked image for {image_path}")

if __name__ == "__main__":
    # Example usage
    # image_size = (128, 128)  # Set the desired image size
    # mask = generate_random_mask(image_size, min_blobs=0, max_blobs=5, max_blob_size=20)
    
    # # Load a PNG image
    # image_path = "/home/sr2369/cspca/max_area_3d_voxel/cspca/0_11_picai/slice_9_cropped.png"  # Replace with your image path
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # if image is None:
    #     raise FileNotFoundError(f"Image not found at {image_path}")
    
    # # Ensure the image is the same size as the mask
    # image = cv2.resize(image, (image_size[1], image_size[0]))
    
    # # Multiply the image with the mask
    # masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # # Save the generated mask and masked image
    # output_mask_path = os.path.join(".", "mask.png")
    # output_masked_image_path = os.path.join(".", "masked_image.png")
    # plt.imsave(output_mask_path, mask, cmap='gray')
    # plt.imsave(output_masked_image_path, masked_image, cmap='gray')
    
    # open each folder in the directory and process the images
    cspca_path = "/home/sr2369/inpainting/max_area_3d_voxel/cspca"
    without_cspca_path = "/home/sr2369/inpainting/max_area_3d_voxel/without_cspca"
    dirs = [os.path.join(cspca_path, d) for d in os.listdir(cspca_path) if os.path.isdir(os.path.join(cspca_path, d))] + [os.path.join(without_cspca_path, d) for d in os.listdir(without_cspca_path) if os.path.isdir(os.path.join(without_cspca_path, d))]
    print(len(dirs))
    for d in tqdm(dirs):
        # print("Processing", d)
        process_images_in_directory(d)
    # process_images_in_directory("/home/sr2369/inpainting/max_area_3d_voxel/cspca/0_11_picai")