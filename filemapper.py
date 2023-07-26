import os

excluded_dirs = ['env', 'sandbox', 'sighthound_email_patch', 'yolov5', '.git', '.pytest_cache', '.vscode', '__pycache__']
file_map = ''

print('starting')
for dirpath, dirnames, filenames in os.walk('.'):
    # Check if the current directory should be excluded
    if any(ex_dir in dirpath for ex_dir in excluded_dirs):
        continue
    
    # Add the current directory to the file map
    file_map += f"{dirpath}\n"

    # Add the files in the current directory to the file map
    for filename in filenames:
        file_map += f"{' ' * 4}{filename}\n"

    # Print the directory name and number of files in the directory
    print(f"Directory: {dirpath} - Number of files: {len(filenames)}")

print('saving file')
print(len(file_map))
with open('filemap.txt', 'w') as f:
    f.write(file_map)
print('finished')
