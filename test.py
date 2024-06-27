import os

# 目标文件夹路径
folder_path = r'C:\Users\86176\PycharmProjects\HumanSpeak'


def get_folder_size(folder_path):
    total_size = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            total_size += get_folder_size(item_path)
        elif os.path.isfile(item_path):
            total_size += os.path.getsize(item_path)
    return total_size


folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
folder_sizes = {folder: get_folder_size(os.path.join(folder_path, folder)) for folder in folders}

max_size_folder = max(folder_sizes, key=folder_sizes.get)
max_size = folder_sizes[max_size_folder]

print(f"最大的文件夹是：{max_size_folder}，其大小为：{max_size} 字节。")
