import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------------------------------------------------
# SUBJECT: Neural Networks and Deep Learning (NNDL)
# TOPIC  : Implement Average Pooling on a Sample Image
# -------------------------------------------------------------------------
# HOW TO EXPLAIN THIS TO YOUR TEACHER (SIMPLE & EASY):
# "Average Pooling reduces the size of an image.
#  It slides a small 2x2 box over the pixels.
#  It calculates the average of those 4 pixels and stores that average
#  in the new image.
#  This cuts the height and width in half and smooths out the image."
# -------------------------------------------------------------------------

def average_pooling(image, pool_size=2):
    """
    Applies 2D Average Pooling to an RGB image.
    
    Parameters:
        image     : 3D numpy array of shape (Height, Width, Channels)
        pool_size : size of the pooling box (default 2 means 2x2 box)
        
    Returns:
        pooled    : smaller image of shape (Height/2, Width/2, Channels)
    """
    height, width, channels = image.shape
    
    # Calculate the new smaller dimensions
    new_height = height // pool_size
    new_width = width // pool_size
    
    # Create an empty blank array for the smaller output image
    pooled = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    
    # Slide the 2x2 box across rows and columns
    for i in range(new_height):
        for j in range(new_width):
            # Find the boundaries of the 2x2 box
            r_start = i * pool_size
            r_end = r_start + pool_size
            c_start = j * pool_size
            c_end = c_start + pool_size
            
            # Extract the 2x2 box of pixels
            box = image[r_start:r_end, c_start:c_end]
            
            # Calculate the average value of the pixels in this box
            pooled[i, j] = np.mean(box, axis=(0, 1))
            
    return pooled


def main():
    print("=" * 65)
    print("   NNDL PBL: IMPLEMENT AVERAGE POOLING ON A SAMPLE IMAGE")
    print("=" * 65)
    
    # 1. Show a tiny numerical example (Easy for Viva!)
    print("\n--- STEP 1: HOW THE MATH WORKS (VIVA EXAMPLE) ---")
    tiny_box = np.array([
        [10, 20],
        [30, 40]
    ])
    tiny_avg = np.mean(tiny_box)
    print("Suppose we have a 2x2 box of pixels:")
    print("   [10, 20]")
    print("   [30, 40]")
    print(f"Formula: (10 + 20 + 30 + 40) / 4 = 100 / 4 = {tiny_avg:.1f}")
    print(f"Resulting Pooled Pixel Value: {tiny_avg:.1f}")

    # 2. Load the sample image
    image_file = "sample.jpg"
    if not os.path.exists(image_file):
        print(f"\n[Error] '{image_file}' not found in current folder.")
        return
        
    print("\n--- STEP 2: LOADING SAMPLE IMAGE ---")
    img = Image.open(image_file).convert("RGB")
    image_array = np.array(img)
    orig_h, orig_w, _ = image_array.shape
    print(f"Loaded: {image_file}")
    print(f"Original Dimensions: {orig_w} x {orig_h} pixels")

    # 3. Apply Average Pooling
    print("\n--- STEP 3: APPLYING AVERAGE POOLING (2x2) ---")
    pooled_array = average_pooling(image_array, pool_size=2)
    new_h, new_w, _ = pooled_array.shape
    print(f"New Dimensions After Pooling: {new_w} x {new_h} pixels")
    print(f"Reduction: Cut height and width in half (75% smaller data size)!")

    # 4. Save results
    output_image_file = "pooled_sample.jpg"
    comparison_file = "comparison_result.png"
    
    # Save the downsampled image
    Image.fromarray(pooled_array).save(output_image_file)
    print(f"\n[OK] Pooled image saved as: {output_image_file}")

    # 5. Create side-by-side comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(image_array)
    axes[0].set_title(f"Original Image\nSize: {orig_w}x{orig_h}", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(pooled_array)
    axes[1].set_title(f"After Average Pooling (2x2)\nSize: {new_w}x{new_h}", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(comparison_file, dpi=150)
    print(f"[OK] Side-by-side comparison saved as: {comparison_file}")
    print("\n" + "=" * 65)
    print("SUCCESS: Model ran completely!")
    print(f"You can show '{comparison_file}' to your professor.")
    print("=" * 65)

if __name__ == "__main__":
    main()
