import os
import cv2

print("=" * 60)
print("DATASET DIAGNOSTIC TOOL")
print("=" * 60)

# Check current directory
cwd = os.getcwd()
print(f"\nCurrent directory: {cwd}")

# List available datasets
datasets = ["Augmented_Dataset", "bone_tumor_dataset"]

for dataset in datasets:
    print(f"\n{'-' * 60}")
    print(f"Checking {dataset}...")
    print(f"{'-' * 60}")
    
    dataset_path = os.path.join(cwd, dataset)
    
    if not os.path.exists(dataset_path):
        print(f"❌ {dataset_path} NOT FOUND")
        continue
    
    no_path = os.path.join(dataset_path, "no")
    yes_path = os.path.join(dataset_path, "yes")
    
    print(f"Dataset path: {dataset_path}")
    print(f"  'no' folder exists: {os.path.exists(no_path)}")
    print(f"  'yes' folder exists: {os.path.exists(yes_path)}")
    
    if os.path.exists(no_path):
        no_files = [f for f in os.listdir(no_path) if os.path.isfile(os.path.join(no_path, f))]
        print(f"  Images in 'no': {len(no_files)}")
        if no_files:
            print(f"    Sample files: {no_files[:3]}")
            # Test reading first file
            test_file = os.path.join(no_path, no_files[0])
            img = cv2.imread(test_file)
            if img is not None:
                print(f"    ✓ Can read images from 'no' folder")
            else:
                print(f"    ❌ Cannot read images from 'no' folder")
    
    if os.path.exists(yes_path):
        yes_files = [f for f in os.listdir(yes_path) if os.path.isfile(os.path.join(yes_path, f))]
        print(f"  Images in 'yes': {len(yes_files)}")
        if yes_files:
            print(f"    Sample files: {yes_files[:3]}")
            # Test reading first file
            test_file = os.path.join(yes_path, yes_files[0])
            img = cv2.imread(test_file)
            if img is not None:
                print(f"    ✓ Can read images from 'yes' folder")
            else:
                print(f"    ❌ Cannot read images from 'yes' folder")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)
print("When loading a dataset, select one of the above folders.")
print("The app will automatically look for 'yes' and 'no' subfolders.")
print("=" * 60 + "\n")
