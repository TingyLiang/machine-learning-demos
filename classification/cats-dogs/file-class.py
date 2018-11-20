import os
import shutil

'''将猫狗图片数据集分开'''
output_train_path = '../../output/catsVSdogs/cat'
output_test_path = '../../output/catsVSdogs/dog'

if not os.path.exists(output_train_path):
    os.makedirs(output_train_path)
if not os.path.exists(output_test_path):
    os.makedirs(output_test_path)


def scanDir_lable_File(dir, flag=True):
    if not os.path.exists(output_train_path):
        os.makedirs(output_train_path)
    if not os.path.exists(output_test_path):
        os.makedirs(output_test_path)
    for root, dirs, files in os.walk(dir, True, None, False):  # 遍列目录
        # 处理该文件夹下所有文件:
        for f in files:
            if os.path.isfile(os.path.join(root, f)):
                a = os.path.splitext(f)
                # print(a)
                # lable = a[0].split('.')[1]
                lable = a[0].split('.')[0]
                print(lable)
                if lable == 'cat':
                    img_path = os.path.join(root, f)
                    mycopyfile(img_path, os.path.join(output_train_path, f))
                else:
                    img_path = os.path.join(root, f)
                    mycopyfile(img_path, os.path.join(output_test_path, f))


def mycopyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


root_path = 'F:/data/cats vs dogs'
train_path = root_path + '/train/'
test_path = root_path + '/test1/'
# s0canDir_lable_File(train_path)
