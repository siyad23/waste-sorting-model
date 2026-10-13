import os
import numpy as np
import supervision as sv

def convert_xyxy_to_yolo(xyxy, image_width=1, image_height=1):
    """
    Converts bounding box coordinates from (x_min, y_min, x_max, y_max) format
    to YOLO's (center_x, center_y, width, height) normalized format.
    
    Since YOLO coordinates are already normalized, we assume image_width and image_height are 1.
    
    Args:
        xyxy (np.ndarray): An array containing [x_min, y_min, x_max, y_max].
        image_width (int): Width of the image (default is 1 for normalized coords).
        image_height (int): Height of the image (default is 1 for normalized coords).

    Returns:
        tuple: A tuple containing (center_x, center_y, width, height).
    """
    x_min, y_min, x_max, y_max = xyxy
    
    # Calculate width and height
    bbox_width = (x_max - x_min) / image_width
    bbox_height = (y_max - y_min) / image_height
    
    # Calculate center coordinates
    center_x = (x_min + x_max) / 2 / image_width
    center_y = (y_min + y_max) / 2 / image_height
    
    return center_x, center_y, bbox_width, bbox_height

def process_directory_with_supervision(input_dir):
    """
    Processes all .txt annotation files in a directory, converting polygons to
    bounding boxes using the supervision library. Modifies files in place.
    
    Args:
        input_dir (str): The path to the directory with the annotations to process.
        
    Returns:
        int: Number of files processed.
    """
    if not os.path.exists(input_dir):
        print(f"  ⚠️  Directory not found: '{input_dir}' - Skipping")
        return 0
    
    converted_files_count = 0
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_dir, filename)
            
            new_lines = []
            with open(input_file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                class_id = parts[0]
                coords = [float(p) for p in parts[1:]]

                # A YOLO bounding box has 4 coordinate values. A polygon will have more.
                if len(coords) > 4:
                    # This line is a polygon, so convert it
                    # Reshape flat list of coords into (N, 2) array of vertices
                    polygon_vertices = np.array(coords).reshape(-1, 2)
                    
                    # Use supervision to get the bounding box in (x_min, y_min, x_max, y_max) format
                    bounding_box_xyxy = sv.polygon_to_xyxy(polygon=polygon_vertices)
                    
                    # Convert the xyxy format back to YOLO's normalized format
                    center_x, center_y, width, height = convert_xyxy_to_yolo(bounding_box_xyxy)
                    
                    # Format the new line for the output file
                    new_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                    new_lines.append(new_line)
                else:
                    # This line is already a bounding box, so keep it as is
                    new_lines.append(line)
            
            # Write the processed lines back to the same annotation file
            with open(input_file_path, 'w') as f:
                f.writelines(new_lines)
            
            converted_files_count += 1
    
    return converted_files_count


def process_dataset(dataset_dir):
    """
    Processes all three subdirectories (train, valid, test) in a dataset directory.
    
    Args:
        dataset_dir (str): The path to the main dataset directory.
    """
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_dir}")
    print(f"{'='*60}\n")
    
    subdirs = ['train', 'valid', 'test']
    total_files = 0
    
    for subdir in subdirs:
        labels_path = os.path.join(dataset_dir, subdir, 'labels')
        print(f"📁 Processing {subdir} labels...")
        
        count = process_directory_with_supervision(labels_path)
        if count > 0:
            print(f"  ✅ Successfully processed {count} annotation files")
            total_files += count
        print()
            
    print(f"{'='*60}")
    print(f"🎉 Processing complete!")
    print(f"📊 Total files processed: {total_files}")
    print(f"{'='*60}\n")


def main():
    """Main function to interactively get dataset directory and process it."""
    print("\n" + "="*60)
    print("  YOLO Polygon to Bounding Box Converter")
    print("="*60)
    print("\nThis script will convert polygon annotations to bounding boxes")
    print("for train, valid, and test datasets.")
    print("\nNote: This will modify the original annotation files in place!")
    print("="*60 + "\n")
    
    # Get dataset directory from user
    dataset_dir = input("Enter the dataset directory path: ").strip()
    
    # Remove quotes if user pasted a path with quotes
    dataset_dir = dataset_dir.strip('"').strip("'")
    
    # Check if directory exists
    if not os.path.isdir(dataset_dir):
        print(f"\n❌ Error: Directory not found at '{dataset_dir}'")
        print("Please check the path and try again.\n")
        return
    
    # Confirm with user
    print(f"\n📂 Dataset directory: {dataset_dir}")
    confirm = input("\nProceed with conversion? (y/n): ").strip().lower()
    
    if confirm != 'y' and confirm != 'yes':
        print("\n⛔ Operation cancelled by user.\n")
        return
    
    # Process the dataset
    process_dataset(dataset_dir)

if __name__ == "__main__":
    main()
