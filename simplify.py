import os

for root, dirs, files in os.walk(r"/Users/jeremy/Desktop/target clean-label attacks/results"):
    for file in files:
        pos = file.find('.')
        if file[pos:] == ".pth":
            os.remove(os.path.join(root, file))
            # print(os.path.join(root, file))