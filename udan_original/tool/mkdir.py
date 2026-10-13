import os


def mkdir(path):
    if os.path.exists(path):
        return
    else:
        os.mkdir(path)
