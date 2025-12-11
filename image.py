import albumentations as A
import cv2
import numpy as np
import os

def apply_filters(image):

    transform = A.Compose([
    A.HorizontalFlip(p=0.5),              # Random horizontal flip
    A.VerticalFlip(p=0.5),                # Random vertical flip
    A.RandomRotate90(p=0.5),              # Random 90 degree rotation
    A.RandomBrightnessContrast(p=0.2),    # Randomly adjust brightness and contrast
    A.HueSaturationValue(p=0.3),          # Adjust hue, saturation, and value
    A.RandomGamma(p=0.5),                 # Random gamma correction
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),  # Apply CLAHE (adaptive histogram equalization)
    A.GaussianBlur(blur_limit=3, p=0.5),  # Apply Gaussian Blur
    A.MedianBlur(blur_limit=3, p=0.3),    # Apply Median Blur
    A.ElasticTransform(alpha=1.0, sigma=50, p=0.4),  # Elastic transformation
    A.Perspective(p=0.5),  # Perspective transformation
    A.Affine(rotate=[1.0,1000.0], p=0.5),  # Random shift, scale, and rotation
    A.RandomFog(p=0.2),  # Add fog effect
    A.RandomShadow(p=0.2),  # Add random shadow effect
    A.ImageCompression(quality_range=[50,100], compression_type='jpeg', p=0.5),  # Simulate JPEG compression artifacts
    ])

    augmented_image = transform(image=image)['image']

    return augmented_image

i = []

def augment_save(data_dir,i):
    folds = os.listdir(data_dir)
    for fold in folds:
        files = f"{data_dir}/{fold}"
        fold_path = os.path.join(data_dir, fold)
        # files = os.path.join(data_dir,fold)
        imgs = os.listdir(files)
        for img in imgs:
            img_path = os.path.join(fold_path, img)
            name = os.path.join(fold_path, f"augmented_{img}")
            image = cv2.imread(img_path)
            if len(image.shape) == 2:
                i.append(image.shape)
            augment = apply_filters(image)
            cv2.imwrite(name, augment)
            print(f"Saved augmented image: {name}")

augment_save("./Training",i)
print(i)
