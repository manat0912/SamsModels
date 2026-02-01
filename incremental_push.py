import os
import subprocess
import glob
import time

def run_command(command):
    print(f"Running: {command}")
    try:
        if isinstance(command, list):
            subprocess.check_call(command, shell=True) 
        else:
            subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error command failed: {e}")
        return False
    return True

GIT_CMD = r'"C:\Program Files\Git\cmd\git.exe"'

def main():
    # 1. Soft reset to undo the massive commit but keep files
    print("Skipping reset to preserve code history...")
    # NOTE: Reset removed to append model commits instead
    # if not run_command(f"{GIT_CMD} reset --mixed HEAD~1"):
    #     print("Reset failed. Maybe no commit to reset? Proceeding anyway check status.")


    # 2. Get list of files
    part_files = []
    ignored_dirs = {".git", "env", "venv", "node_modules", "__pycache__"}
    
    for root, dirs, files in os.walk("app"):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            if ".part" in file:
                part_files.append(os.path.join(root, file))
    
    # Sort files naturally
    part_files.sort()
    
    print(f"Found {len(part_files)} part files.")

    # 3. Batch push part files
    BATCH_SIZE = 5 # 5 * 95MB = ~475MB per push
    
    total_batches = (len(part_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # for i in range(55, len(part_files), BATCH_SIZE):
    #     batch = part_files[i:i+BATCH_SIZE]
    #     batch_num = (i // BATCH_SIZE) + 1
    #     print(f"Processing Batch {batch_num}/{total_batches} ({len(batch)} files)...")
        
    #     # Git Add
    #     # Quote filenames to handle spaces/parentheses
    #     quoted_batch = [f'"{f}"' for f in batch]
    #     add_cmd = f"{GIT_CMD} add {' '.join(quoted_batch)}"
    #     if not run_command(add_cmd):
    #         print("Failed to add files")
    #         exit(1)
            
    #     # Git Commit
    #     commit_msg = f"Upload model parts batch {batch_num}"
    #     if not run_command(f'{GIT_CMD} commit -m "{commit_msg}"'):
    #         print("Failed to commit")
    #         exit(1)
            
    #     # Git Push
    #     # Retry logic for push
    #     max_retries = 3
    #     success = False
    #     for attempt in range(max_retries):
    #         print(f"Pushing (Attempt {attempt+1}/{max_retries})...")
    #         if run_command(f"{GIT_CMD} push origin main"):
    #             success = True
    #             break
    #         print("Push failed, retrying in 5 seconds...")
    #         time.sleep(5)
            
    #     if not success:
    #         print(f"Failed to push batch {batch_num} after retries. Manual intervention needed.")
    #         exit(1)
            
    # 4. Push remaining files
    print("Processing remaining files...")
    run_command(f"{GIT_CMD} add .") # Add everything else
    run_command(f'{GIT_CMD} commit -m "Update configuration and scripts"')
    run_command(f"{GIT_CMD} push origin main")
    
    print("Incremental push complete!")

if __name__ == "__main__":
    main()
