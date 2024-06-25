import os


def find_zero_size_files(folder_path):
    zero_size_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) == 0:
                zero_size_files.append(file_path)
                print(root, dirs, file)
                os.remove((file_path))
    return zero_size_files


folder_path = 'output'
zero_size_files = find_zero_size_files(folder_path)
for file_path in zero_size_files:
    print(f'Zero size file: {file_path}')
