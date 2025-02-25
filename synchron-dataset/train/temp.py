import os

# Define the directories
images_dir = "images"
labels_dir = "labels"

# Create the labels directory if it doesn't exist
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# Iterate through all files in the images directory
for filename in os.listdir(images_dir):
    # Check if the file is an image (you can add more extensions if needed)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
        # Get the base name without the extension
        base_name = os.path.splitext(filename)[0]
        
        # Create the corresponding .txt file in the labels directory
        txt_file_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        # Write to the .txt file (you can customize the content as needed)
        with open(txt_file_path, "w") as txt_file:
            txt_file.write("")  # Write an empty file or add custom content here

        print(f"Created: {txt_file_path}")

print("All corresponding .txt files have been created in the labels/ directory.")
