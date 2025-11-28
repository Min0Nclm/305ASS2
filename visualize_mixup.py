import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def visualize_mixup(image_dir: str, alpha: float = 1.0):
    """
    Finds two images in a directory, applies mixup augmentation, and
    saves a 1x3 grid visualization of the original and mixed images.

    Args:
        image_dir (str): The directory containing the two images.
        alpha (float): The alpha parameter for the Beta distribution used in mixup.
    """
    # --- 1. Validate directory and find the two image files ---
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found at '{image_dir}'")
        return

    try:
        image_names = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if len(image_names) != 2:
            print(f"Error: Expected exactly 2 images in '{image_dir}', but found {len(image_names)}.")
            return
        
        img_path1 = os.path.join(image_dir, image_names[0])
        img_path2 = os.path.join(image_dir, image_names[1])

    except Exception as e:
        print(f"Error while accessing directory or finding images: {e}")
        return

    # --- 2. Define transformation and load images ---
    transform = T.Compose([
        T.Resize((256, 256)), # Resize for consistency
        T.ToTensor(),         # Convert image to a PyTorch tensor
    ])

    try:
        with Image.open(img_path1).convert('RGB') as img1, Image.open(img_path2).convert('RGB') as img2:
            tensor1 = transform(img1)
            tensor2 = transform(img2)
    except Exception as e:
        print(f"Error opening or transforming images: {e}")
        return

    # --- 3. Apply Mixup ---
    lam = np.random.beta(alpha, alpha)
    mixed_tensor = lam * tensor1 + (1 - lam) * tensor2

    # --- 4. Create and save the plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare images for plotting (permute from C,H,W to H,W,C)
    img1_plot = tensor1.permute(1, 2, 0)
    img2_plot = tensor2.permute(1, 2, 0)
    mixed_img_plot = mixed_tensor.permute(1, 2, 0)

    # Plot original image 1
    axes[0].imshow(img1_plot)
    axes[0].set_title("Cat")
    axes[0].axis('off')

    # Plot original image 2
    axes[1].imshow(img2_plot)
    axes[1].set_title("Rabbit")
    axes[1].axis('off')

    # Plot mixed image
    axes[2].imshow(mixed_img_plot)
    axes[2].set_title(f"Mixup")
    axes[2].axis('off')

    plt.tight_layout()

    save_path = os.path.join(image_dir, "mixup_visualization.png")
    try:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Successfully saved mixup visualization to '{save_path}'")
    except Exception as e:
        print(f"Error saving the figure: {e}")

    plt.close(fig)

if __name__ == '__main__':
    # The directory where your two images are located.
    TARGET_DIR = r"C:\Users\mino\Desktop\dataset\mixup"
    # The alpha parameter controls the strength of the mixup.
    # alpha=1.0 means the mixing ratio is sampled from a uniform distribution.
    MIXUP_ALPHA = 1.0
    visualize_mixup(TARGET_DIR, alpha=MIXUP_ALPHA)
