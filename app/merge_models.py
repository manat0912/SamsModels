import os
import glob

def merge_files(original_filename):
    # Find all parts
    parts = sorted(glob.glob(f"{original_filename}.part*"))
    if not parts:
        print(f"No parts found for {original_filename}")
        return

    print(f"Merging {len(parts)} parts into {original_filename}...")
    
    with open(original_filename, 'wb') as outfile:
        for part in parts:
            print(f"Reading {part}...")
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    
    print(f"Successfully merged {original_filename}")

    print(f"Removing part files for {original_filename}...")
    for part in parts:
        try:
            os.remove(part)
            print(f"Removed {part}")
        except OSError as e:
            print(f"Error removing file {part}: {e}")

if __name__ == "__main__":
    search_path = "app"
    print(f"Scanning {search_path} for split files to merge...")
    
    # Set of base filenames to merge (to avoid duplicate processing)
    files_to_merge = set()
    
    for root, dirs, files in os.walk(search_path):
        for name in files:
            if name.endswith(".part000"):
                # Reconstruct base filename
                base_name = name.replace(".part000", "")
                full_path = os.path.join(root, base_name)
                files_to_merge.add(full_path)
                
    print(f"Found {len(files_to_merge)} files to merge.")
    
    for full_path in files_to_merge:
        merge_files(full_path)
