import os

def split_file(file_path, chunk_size=95 * 1024 * 1024):  # 95 MB
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    file_size = os.path.getsize(file_path)
    if file_size <= chunk_size:
        print(f"File {file_path} is smaller than chunk size. Skipping.")
        return

    print(f"Splitting {file_path} ({file_size / (1024*1024):.2f} MB)...")
    
    part_num = 0
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            part_filename = f"{file_path}.part{part_num:03d}"
            with open(part_filename, 'wb') as part_file:
                part_file.write(chunk)
            
            print(f"Created {part_filename}")
            part_num += 1
    
    print(f"Done splitting {file_path} into {part_num} parts.")

if __name__ == "__main__":
    # Generic recursive scanner for large files
    search_path = "app"
    chunk_size_mb = 95
    limit_bytes = chunk_size_mb * 1024 * 1024
    
    print(f"Scanning {search_path} for files larger than {chunk_size_mb}MB...")
    
    files_to_split = []
    
    ignored_dirs = {".git", "env", "venv", "node_modules", "__pycache__"}
    
    for root, dirs, files in os.walk(search_path):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for name in files:
            filepath = os.path.join(root, name)
            # Skip existing part files
            if ".part" in name:
                continue
                
            try:
                size = os.path.getsize(filepath)
                if size > limit_bytes:
                    print(f"Found large file: {filepath} ({size/1024/1024:.2f} MB)")
                    files_to_split.append(filepath)
            except OSError:
                pass
                
    print(f"Found {len(files_to_split)} files to split.")
    
    for full_path in files_to_split:
        split_file(full_path)
