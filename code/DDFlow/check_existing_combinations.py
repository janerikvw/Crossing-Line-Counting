images_dir = 'smaller_untrained_images/'
new_images_dir = 'smaller_untrained_images/'
video_dir = 'train_videos/'
test_write_file = 'untrained_images/test_images.txt'
training_write_file = 'untrained_images/train_images.txt'
new_training_write_file = 'untrained_images/train_images3.txt'


from glob import glob
import cv2
import os

from PIL import Image


def check_img_size(file_name, directory):
    try:
        im = Image.open('{}{}'.format(directory, file_name))
    except IOError:
        return False
    return im.size[0] * im.size[1] > 100


a = open(new_training_write_file, 'w+')
f = open(training_write_file, 'r')
c = 0
for line in f.readlines():
    splitted = line.split(' ')
    if check_img_size(splitted[0], images_dir) and check_img_size(splitted[1], images_dir) and len(splitted) == 3:
        a.write(line)
        c += 1

    if c == 1500:
        break
print('Lines existing: {}'.format(c))
f.close()


exit()

f = open(new_training_write_file, 'r')
l = set()

lines = f.readlines()
for i, line in enumerate(lines):
    splitted = line.split(' ')

    if(splitted[0] == '2020' or splitted[1] == '2020'):
        print(i, line)
        print(lines[i+1])

    l.add(splitted[0])
    l.add(splitted[1])
f.close()

from pathlib import Path
#Path(new_images_dir).mkdir(parents=True)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os.path

for file_name in l:
    #print('{}{}'.format(new_images_dir, file_name))
    if file_name == '2020':
        print(file_name)
        continue

    if not os.path.isfile('{}{}'.format(new_images_dir, file_name)):
        im = Image.open('{}{}'.format(images_dir, file_name))
        im.thumbnail((1280, 720), Image.ANTIALIAS)
        im.save('{}{}'.format(new_images_dir, file_name), 'PNG')

