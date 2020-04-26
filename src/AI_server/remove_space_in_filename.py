#coding=utf-8
import os

base_path = '/u02/dataset/car_brands/data/train/'

def remove_space(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.find(' ') > 0:
                src_file = os.path.join(root, file) 
                dst_file = os.path.join(root, file.replace(' ', ''))
                if os.path.exists(dst_file):
                    dst_file = os.path.join(root, file.replace(' ', '_'))
                os.rename(src_file, dst_file)
                count += 1
    print(count) 

if __name__ == '__main__':
    remove_space(base_path)