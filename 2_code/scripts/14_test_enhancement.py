import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Setup Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_DIR = os.path.join(ROOT_DIR, '1_data', 'raw') # Assuming raw images are here

def apply_clahe(image_path):
    # 1. Read Image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img = cv2.resize(img, (224, 224))
    
    # 2. Convert to LAB Color Space (L = Lightness, A/B = Colors)
    # We only want to enhance Lightness, not mess up the colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 3. Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # 4. Merge and Convert back to BGR
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return img, final

def visualize():
    # Find a random image in the raw folder
    found = False
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                print(f"Testing on: {file}")
                
                res = apply_clahe(full_path)
                if res:
                    original, enhanced = res
                    
                    # Plot
                    plt.figure(figsize=(10, 5))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                    plt.title("Original (Raw)")
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                    plt.title("Enhanced (CLAHE)")
                    plt.axis('off')
                    
                    plt.show()
                    found = True
                    break
        if found: break

    if not found:
        print("No raw images found to test! Do you have images in 1_data/raw?")

if __name__ == "__main__":
    visualize()