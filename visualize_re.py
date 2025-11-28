import os
import random
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

def visualize_random_erasing(image_dir: str, num_images: int = 4):
    """
    Randomly selects images from a directory, applies Random Erasing,
    and saves a 2x2 grid visualization of the results.

    Args:
        image_dir (str): The directory containing images to process.
        num_images (int): The number of images to select, should be 4 for a 2x2 grid.
    """
    # --- 1. Validate directory and find image files ---
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found at '{image_dir}'")
        return

    try:
        all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if len(all_images) < num_images:
            print(f"Error: Need at least {num_images} images in '{image_dir}', but found only {len(all_images)}.")
            return
        selected_image_names = random.sample(all_images, num_images)
    except Exception as e:
        print(f"Error while accessing directory or selecting images: {e}")
        return

    # --- 2. Define the transformation pipeline ---
    # RandomErasing operates on tensors, so we need ToTensor first.
    # We use a high probability (p=1.0) to ensure the effect is visible.
    transform = T.Compose([
        T.Resize((256, 256)),  # Resize for consistency
        T.ToTensor(),         # Convert image to a PyTorch tensor (values in [0, 1])
        T.RandomErasing(p=1.0, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=0), # Always apply erasing
    ])

    # --- 3. Process images and create plot ---
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

    
    # Flatten the 2x2 axes array for easy iteration
    axes = axes.flatten()

    for i, (ax, img_name) in enumerate(zip(axes, selected_image_names)):
        try:
            img_path = os.path.join(image_dir, img_name)
            # Open image in RGB mode
            with Image.open(img_path).convert('RGB') as img:
                # Apply the transform
                transformed_tensor = transform(img)
                
                # Convert tensor back to a format suitable for plotting
                # (H, W, C) from (C, H, W)
                img_to_show = transformed_tensor.permute(1, 2, 0)
                
                ax.imshow(img_to_show)

                ax.axis('off') # Hide axes ticks

        except Exception as e:
            print(f"Could not process or plot image '{img_name}': {e}")
            ax.text(0.5, 0.5, 'Error', horizontalalignment='center', verticalalignment='center')
            ax.axis('off')

    # --- 4. Save the final visualization ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    save_path = os.path.join(image_dir, "random_erasing_visualization.png")
    try:
        plt.savefig(save_path)
        print(f"Successfully saved visualization to '{save_path}'")
    except Exception as e:
        print(f"Error saving the figure: {e}")

    plt.close(fig) # Close the figure to free up memory

if __name__ == '__main__':
    # The directory where your dataset is located.
    # IMPORTANT: Make sure to use the correct path for your system.
    TARGET_DIR = r"C:\Users\mino\Desktop\dataset"
    visualize_random_erasing(TARGET_DIR)
