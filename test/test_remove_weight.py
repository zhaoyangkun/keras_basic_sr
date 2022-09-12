import os


file_list = os.listdir("./result/srgan/weights")
for file in file_list:
    file_path = os.path.join("./result/srgan/weights", file)
    if os.path.exists(file_path):
        os.remove(file_path)
