import os
import numpy as np
from matplotlib import pyplot as plt
import cv2
import shutil


def GetFileList(FindPath, FlagStr=[]):
    import os
    FileList = []
    FileNames = os.listdir(FindPath)
    if len(FileNames) > 0:
        for fn in FileNames:
            if len(FlagStr) > 0:
                if IsSubString(FlagStr, fn):
                    fullfilename = os.path.join(FindPath, fn)
                    FileList.append(fullfilename)
            else:
                fullfilename = os.path.join(FindPath, fn)
                FileList.append(fullfilename)

    if len(FileList) > 0:
        FileList.sort()

    return FileList


def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag

def get_txt(txt_name, root_dir):
    txt = open(txt_name, 'w')
    list = os.listdir(root_dir)
    for i, l in enumerate(list):
        file_path = os.path.join(root_dir, l)
        imgfile = GetFileList(file_path)
        for img in imgfile:
            strs = img + ' ' + str(i) + '\n'
            txt.writelines(strs)

    # imgfile = GetFileList('first_batch/train_male')
    # for img in imgfile:
    #     str = img + '\t' + '0' + '\n'
    #     txt.writelines(str)
    txt.close()



if __name__ == '__main__':
    get_txt('train.txt', 'train/')
    get_txt('val.txt', 'val/')

