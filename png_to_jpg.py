from PIL import Image
import os

def convert_png_to_jpg(input_path, output_path, quality=95):
    with Image.open(input_path) as img:
        rgb_img = img.convert("RGB")  # Convert to RGB to remove transparency
        rgb_img.save(output_path, "JPEG", quality=quality)

# data-path with labels generating
def define_paths(data_dir):

    folds = os.listdir(data_dir)
    for fold in folds:
        files = f"{data_dir}/{fold}"
        # files = os.path.join(data_dir,fold)
        imgs = os.listdir(files)
        for img in imgs:
            path = f"{files}/{img}"
            img = img.split(".")
            if fold=='1':
                output_path = f"./brain-tumor/Training/meningioma/{img[0]}.jpg"
                convert_png_to_jpg(path,output_path)
            elif fold=='2':
                output_path = f"./brain-tumor/Training/glioma/{img[0]}.jpg"
                convert_png_to_jpg(path,output_path)
            elif fold=='3':
                output_path = f"./brain-tumor/Training/pituitary/{img[0]}.jpg"
                convert_png_to_jpg(path,output_path)
            elif fold=='4':
                output_path = f"./brain-tumor/Training/notumor/{img[0]}.jpg"
                convert_png_to_jpg(path,output_path)

# Example usage
define_paths("./archive")
