import os

# Set your paths here
image_folder = "/Users/jennadickman/Documents/fall23/bamlab/mm_gnc/dataset/train/images"
label_folder = "/Users/jennadickman/Documents/fall23/bamlab/mm_gnc/dataset/train/labels"

# Get list of all image and label files without extensions
image_files = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if not f.startswith('.')]
label_files = [os.path.splitext(f)[0] for f in os.listdir(label_folder) if not f.startswith('.')]

# Find discrepancies
images_without_labels = set(image_files) - set(label_files)
labels_without_images = set(label_files) - set(image_files)

print("Images without labels:", images_without_labels)
print("Labels without images:", labels_without_images)
