import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import os.path as osp
import pdb
import random
import re
import shutil
import subprocess as sp
import sys
import time
import cv2 as cv


LABEL_FOLDER = 'label_making'  # each folder is a clip for making labels
#######################################
# every time add more batches, update in
# 1. BATCH_STR
# 2. BATCH_NUM
# 3. find_prefix()
# 4. arg_parser()
# 5. generate_clip_name()
####################################### 
BATCH_STR = { # these are used for represent many videos at once: '-c mlb' equals '-c 60 70 80 90 100'
    'mlb': [60, 70, 80, 90, 100],
    'clip1': [1, 3, 4, 7, 8],
    'clip2': [10, 11, 12, 13, 14, 15, 16],
    'tcs1': ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8'],
    'tcs2': ['t9', 't10', 't11', 't12', 't13', 't14', 't15', 't16'],
    'indoor1': ['in1', 'in2', 'in3', 'in4', 'in5', 'in6', 'in7',
                'in8', 'in9', 'in10', 'in11', 'in12'],
    'indoor2': ['in25', 'in27', 'in29', 'in30', 'in31', 'in33', 'in34'],
    'youtube': ['ytb1', 'ytb2', 'ytb3', 'ytb4', 'ytb5', 'ytb6', 'ytb7', 'ytb8', 'ytb9', 'ytb10'],
    'positionA': [1, 8, 10, 14],
}
BATCH_NUM = {# clip1 have 15 data folder: clip_1_1, clip_1_2, ...
    1: 15, 3: 1, 4: 9, 7: 1, 8: 1, 
    10: 5, 11: 6, 12: 6, 13: 6, 14: 5, 15: 5, 16: 5,
    60: 10, 70: 10, 80: 10, 90: 10, 100: 10,
    't1': 11, 't2': 13, 't3': 10, 't4': 7, 't5': 8, 't6': 10, 't7': 9, 't8': 6,
    't9': 18, 't10': 15, 't11': 16, 't12': 15, 't13': 10, 't14': 10, 't15': 10, 't16': 10,
    'in1': 1, 'in2': 2, 'in3': 4, 'in4': 1, 'in5': 4, 'in6': 4, 'in7': 15, 'in8': 4, 'in9': 1, 'in10': 3, 'in11': 7, 'in12': 5,
    'in25': 5, 'in27': 7, 'in29': 7, 'in30': 7, 'in31': 7, 'in33': 7, 'in34': 7,
    'ytb1': 10, 'ytb2': 8, 'ytb3': 10, 'ytb4': 9, 'ytb5': 10,
    'ytb6': 9, 'ytb7': 11, 'ytb8': 10, 'ytb9': 7, 'ytb10': 10,
    'speed48': 5, 'speed49': 4, 'speed50': 10, 'speed51': 6, 'speed52': 7, 'speed53': 7,
    'speed54': 10, 'speed55': 10, 'speed56': 10, 'speed57': 10,
    'speed58': 10, 'speed59': 10, 'speed60': 10, 'speed61': 10,
    'speed73': 5, 'speed79': 5,
}
VALIDATE_PERCENT = 30  # put all data in training set first, then move 30% of the data into validation set


def timer(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        r = fn(*args, **kwargs)
        end = time.time()
        print(f"{fn.__name__} uses: {end - start:.4f}s")
        return r

    return wrapper


def arg_parser():
    parser = argparse.ArgumentParser(description='functions for building datasets')

    # clips for processing
    parser.add_argument('-c', '--clip', nargs='+', action='append', metavar='', help='clips for processing')

    # process function
    parser.add_argument('-p', '--parse', action='store_true', help='set this to parse clips, default is false')
    parser.add_argument('--no-parse', dest='parse', action='store_false')
    parser.set_defaults(parse=False)

    parser.add_argument('-b', '--build', action='store_true', help='set this to build datasets, default is false')
    parser.add_argument('--no-build', dest='build', action='store_false')
    parser.set_defaults(build=False)

    parser.add_argument('-m', '--merge', action='store_true', help='set this to merge datasets, default is false')
    parser.add_argument('--no-merge', dest='merge', action='store_false')
    parser.set_defaults(merge=False)

    parser.add_argument('-r', '--resize', action='store_true', help='set this to resize image data to 720p, default is false')
    parser.add_argument('--no-resize', dest='resize', action='store_false')
    parser.set_defaults(resize=False)


    args = parser.parse_args()
    print(args)
    all_batch = []
    for clips in args.clip:
        temp = []
        for clip in clips:
            if clip in BATCH_STR:  # indoor, tcs1, tc2, clip1, clip2, mlb
                temp.extend(BATCH_STR[clip])
            elif clip.startswith('t'):  # t1, t2, t3
                temp.append(clip)
            elif clip.startswith('in'):  # in1, in2, in3
                temp.append(clip)
            elif clip.startswith('ytb'):  # ytb1, ytb2, ytb3
                temp.append(clip)
            elif clip.startswith('speed'):
                temp.append(clip)
            else:  # number: 1, 2, 3
                temp.append(int(clip))
        all_batch.append(temp)
    args.clip = all_batch
    return args


def set_logger():
    """
    set logger to print info to sys.stdout
    """
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(message)s', '%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log


def find_prefix(file):
    '''
    every time add more batches, add prefix here
    used in other functions
    '''
    return (
    file.startswith('clip') or
    file.startswith('mlb') or 
    file.startswith('in') or 
    file.startswith('ytb') or 
    file.startswith('tcs') or 
    file.startswith('speed')
    )


def make_new_dir(dir, delete=False):
    """
    give a name, create a directory
    if the directory exists, delete before creating if delete arg is True
    return directory's name
    """
    if not osp.isdir(dir):
        os.mkdir(dir)
        log.info(f'create {dir}')
    elif delete:
        shutil.rmtree(dir)
        os.mkdir(dir)
        log.info(f'{dir} already existed, delete and create new one')
    else:
        log.info(f'{dir} already existed')
    return dir


def delete_batch(dir='', prefix=None, suffix=None):
    """
    give a directory, delete files according to prefix and suffix
    """
    if dir:
        before = os.getcwd()
        os.chdir(dir)
    for i in os.listdir():
        if prefix and suffix:
            if i.startswith(prefix) and i.endswith(suffix):
                os.remove(i)
        elif prefix:
            if i.startswith(prefix):
                os.remove(i)
        elif suffix:
            if i.endswith(suffix):
                os.remove(i)
    if dir:
        os.chdir(before)


def parse_clip(video_path):  # clip1.mov
    """
    give a clip, parse into frames
    create a directory and store photos into the directory
    return the directory

    mlb clips are given as mlb_p100_1
    local clips are give as clip1, clip_1_1, clip_3_1
    """
    if osp.isdir(LABEL_FOLDER):
        os.chdir(LABEL_FOLDER)
    before = os.getcwd()

    for v in generate_clip_name(video_path):
        video_name = osp.splitext(v)[0]
        flag = False
        for dirpath, _, filenames in os.walk('../raw_clips'):
            if flag:
                break
            for filename in filenames:
                if filename.find(video_name) != -1:
                    v = osp.join(dirpath, filename)
                    flag = True
                    break
        dir_name = make_new_dir(video_name)
        vidcap = cv.VideoCapture(v)
        success, image = vidcap.read()
        count = 0
        while success:
            cv.imwrite(osp.join(dir_name, f"{video_name}_{count}.jpg"), image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
            if not count % 100:
                log.info(f'read until frame {count}')
        log.info(f'read until frame {count}')

        os.chdir(before)


def delete_class_person(dataset):
    """
    issue: there are some class1(person) labels in the dataset
    some are labeled as 1, some are wrongly labled as 0
    delete these labels
    """
    before = os.getcwd()

    os.chdir(f'{dataset}/labels')
    count = 0
    for dir in ('train', 'validate'):
        for file_name in os.listdir(dir):
            with open(osp.join(dir, file_name), 'r') as f:
                old_label = f.readlines()
                # new_label = [i for i in old_label if i[0] == '0']  # class 0
                new_label = [old_label[0]]  # TODO: only select the first box, cause some class1 are wrongly labled as class0
            if len(new_label) != len(old_label):  # contain class1(person) in old label
                with open(osp.join(dir, file_name), 'w') as f:
                    f.writelines(new_label)
                    count += 1    
    if count == 0:
        log.info(f'try delete class 1(person) in all labels but nothing found')
    else:
        log.info(f'delete class 1(person) in {count} labels')
    
    os.chdir(before)


def delete_empty_frame(clip):
    """
    delete frames that don't have the box(don't have a txt)
    """
    if osp.isdir(LABEL_FOLDER):
        os.chdir(LABEL_FOLDER)
    before = os.getcwd()
    count = 0
    for c in generate_clip_name(clip):

        os.chdir(c)
        label_frame = set()
        for file in os.listdir():
            if find_prefix(file) and file.endswith('txt'):
                label_frame.add(file[:-4])
        empty = False
        for file in os.listdir():
            if find_prefix(file) and file[:-4] not in label_frame:
                os.remove(file)
                empty = True
        if empty:
            count += 1
            log.info(f'delete empty frame in {c}')
        os.chdir(before)
    if count == 0:
        log.info(f'try delete empty frame but no clips contain empty frame')
    else:
        log.info(f'delete empty frame in {count} clips')


def delete_empty_txt(clip):
    """
    delete empty file(mostly txt)
    """
    if osp.isdir(LABEL_FOLDER):
        os.chdir(LABEL_FOLDER)
    before = os.getcwd()
    count = 0
    for c in generate_clip_name(clip):
        os.chdir(c)
        empty = False
        for i in os.listdir():
            if osp.getsize(i) == 0:
                empty = True
                os.remove(i)
        if empty:
            count += 1
            log.info(f'delete empty txt in {c}')
        os.chdir(before)
    if count == 0:
        log.info(f'try delete empty txt but no clip contains empty txt')
    else:
        log.info(f'delete empty txt in {count} clips')


def get_dir_size(start_path='.'):
    """
    give a directory name, return the size of the directory
    osp.getsize() is useless, since we are having a directory not a file
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def generate_clip_name(batchs):
    """
    used in functions that accpet clip folder name as the input `clip_1_1, mlb_p70_4, tcs_1_1, 'in_1_1`
    give some description of clips, generate clip names 
    description can be string: 'mlb', 'clip', or number: 1, 3, 4, 60, 70
    all batch less than 50 are local clips, start with `clip`
    all batch greater than 50 are mlb clips, start with `mlb`

    all tcs batch start with `tcs`
    all indoor batch start with `in'
    all youtube batch start with 'ytb'
    """
    # flatten 2d list to a set(remove repeated items)
    # pdb.set_trace()
    process_batchs = []
    for batch in batchs:
        if batch in BATCH_STR:
            process_batchs.extend(BATCH_STR[batch])
        # elif isinstance(batch, int):
        #     process_batchs.append(batch)
        # elif batch.startswith('t'):
        #     process_batchs.append(batch)
        # elif batch.startswith('in'):
        #     process_batchs.append(batch)
        else:
            process_batchs.append(batch)
    process_batchs = sorted(list(set(process_batchs)))
    all_clip = []
    for batch in process_batchs:
        if isinstance(batch, int) and batch < 50:
            prefix = 'clip'
        elif isinstance(batch, int) and batch >= 60:
            prefix = 'mlb'
        elif batch.startswith('t'):
            prefix = 'tcs'
        elif batch.startswith('in'):
            prefix = 'in'
        elif batch.startswith('ytb'):
            prefix = 'ytb'
        elif batch.startswith('speed'):
            prefix = 'speed'

        if prefix == 'clip':
            all_clip.extend([prefix + f'_{batch}_{i}' for i in range(1, BATCH_NUM[batch] + 1)])
        elif prefix == 'mlb':
            all_clip.extend([prefix + '_p' + f'{batch}_{i}'for i in range(1, BATCH_NUM[batch] + 1)])
        elif prefix == 'tcs':
            all_clip.extend([prefix + f'_{batch[1:]}_{i}' for i in range(1, BATCH_NUM[batch] + 1)])  # batch='t1', batch[1:]='1'
        elif prefix == 'in':
            all_clip.extend([prefix + f'_{batch[2:]}_{i}' for i in range(1, BATCH_NUM[batch] + 1)])  # batch='in1', batch[2:]='1'
        elif prefix == 'ytb':
            all_clip.extend([prefix + f'_{batch[3:]}_{i}' for i in range(1, BATCH_NUM[batch] + 1)])  # batch='ytb1', batch[3:]='1'
        elif prefix == 'speed':
            all_clip.extend([prefix + f'_{batch[5:]}_{i}' for i in range(1, BATCH_NUM[batch] + 1)])  # batch='speed1', batch[5:]='1'

    return all_clip


def build_base_dataset(dataset_name='fast_baseball'):
    """
    build a empty dataset with the valid folder hierarchy
    """
    clip_dir = os.getcwd()
    make_new_dir(dataset_name, delete=True)
    os.chdir(dataset_name)
    image_label_dir = os.getcwd()
    os.mkdir('images')
    os.chdir('images')
    os.mkdir('train')
    os.mkdir('validate')
    os.chdir(image_label_dir)
    shutil.copytree('images', 'labels')
    os.chdir(clip_dir)
    log.info(f'create empty dataset: {dataset_name}')


def build_dataset(clip_number, dataset_name='fast_baseball'):
    """
    give clip_number, generate clip files using `generate_clip_name`
    build dataset, store in train and validate
    delete the old dataset every time
    dataset folder are in /label_making
    """
    # delete empty txt and frame before build the dataset
    delete_empty_txt(clip_number)
    delete_empty_frame(clip_number)

    # build a dataset file hierarchy
    build_base_dataset(dataset_name)
    train_image_path = f'{dataset_name}/images/train'
    train_label_path = f'{dataset_name}/labels/train'
    validate_image_path = f'{dataset_name}/images/validate'
    validate_label_path = f'{dataset_name}/labels/validate'


    # move images and labels into dataset files
    # first, move all data into train
    all_txt_data = []
    for clip in generate_clip_name(clip_number):
        count_jpg = count_txt = 0
        for file_name in os.listdir(clip):
            if file_name.endswith('jpg') and find_prefix(file_name):  # image
                shutil.copyfile(osp.join(clip, file_name), osp.join(train_image_path, file_name))
                count_jpg += 1
            elif file_name.endswith('txt') and find_prefix(file_name):  # label
                shutil.copyfile(osp.join(clip, file_name), osp.join(train_label_path, file_name))
                all_txt_data.append(file_name[:-4])
                count_txt += 1
        if count_jpg != count_txt:
            log.error(f"#jpg: {count_jpg} and #txt: {count_txt} don't match")
            return
        else:
            log.info(f'data from {clip}, {count_jpg} items')
    log.info(f'all data in {dataset_name}, {len(all_txt_data)} items')

    # second, randomly choose part of data and move into validate
    random.shuffle(all_txt_data)
    for i in range(len(all_txt_data) * VALIDATE_PERCENT // 100):
        shutil.move(osp.join(train_image_path, f'{all_txt_data[i]}.jpg'), osp.join(validate_image_path, f'{all_txt_data[i]}.jpg'))
        shutil.move(osp.join(train_label_path, f'{all_txt_data[i]}.txt'), osp.join(validate_label_path, f'{all_txt_data[i]}.txt'))
    log.info(f'shuffle and put {VALIDATE_PERCENT}% data in validate')

    # delete person label in clip_1, clip_3, clip_4
    delete_class_person(dataset_name)

    # print dataset's size
    log.info(f'dataset size: {round(get_dir_size(dataset_name) / 1e9, 2)}G')


def merge_helper(large, small):
    """
    helper function used in merge_dataset
    give a large dataset and a small dataset, copy the small one to the large one
    """
    for p1 in ('images', 'labels'):
        for p2 in ('train', 'validate'):
            large_path = osp.join(large, p1, p2)
            small_path = osp.join(small, p1, p2)
            for file in os.listdir(small_path):
                shutil.copyfile(osp.join(small_path, file), osp.join(large_path, file))
    log.info(f'{small} has been merged into {large}')


def merge_dataset(datasets):
    """
    as the number of labeled images increases, don't build the dataset with all images every time
    give many datasets, merge all to one new dataset
    copy data to the new dataset(not in-place), for further use
    """
    if osp.isdir(LABEL_FOLDER):
        os.chdir(LABEL_FOLDER)
    # if len(datasets) <= 1:
    #     log.info(f'only one dataset: {datasets[0]}')
    #     return
    
    # build an empty datset
    build_base_dataset('fast_baseball')

    # merge 
    max_dataset = 'fast_baseball'
    for d in datasets:
        merge_helper(max_dataset, d)


def resize_helper(command, dirpath, file):
    sp.run(command, stdout=sp.DEVNULL, stderr=sp.STDOUT)
    os.remove(osp.join(dirpath, file))


@timer
def resize_dataset(datasets):
    """
    resize local clips' image data in dataset from 1920*1080 to 1280*720
    use ffmpeg to resize
    try multithread to boost resizing process
    """
    if osp.isdir(LABEL_FOLDER):
        os.chdir(LABEL_FOLDER)

    for dataset in datasets:
        if not osp.isdir(dataset):
            log.info(f'{dataset} is not here')
            continue 

        for dirpath, _, filenames in os.walk(osp.join(dataset, 'images')):
            if dirpath.endswith('images'):  # path doesn't have any image
                continue
            log.info(f'start shrinking images with MULTITHREADING in {dirpath}')

            with ThreadPoolExecutor() as executor:
                for file in filenames:
                    ffmpeg_input = f'{osp.join(dirpath, file)}'
                    ffmpeg_output = f'{osp.join(dirpath, osp.splitext(file)[0])}' + '.png'
                    ffmpeg_command = ['ffmpeg', '-i', ffmpeg_input, '-vf', 'scale=1280:-1', ffmpeg_output]
                    executor.submit(resize_helper, ffmpeg_command, dirpath, file)
            


if __name__ == '__main__':    
    # parse_clip('speed_51_1')
    # sys.exit()
    log = set_logger()
    table = arg_parser()

    all_dataset = []
    log.info(table)

    for dataset in table.clip:
        name = f"build_{'_'.join(str(i) for i in dataset)}"
        all_dataset.append(name)
        # print(generate_clip_name(dataset))
        if table.parse:
            parse_clip(dataset)
        if table.build:
            build_dataset(dataset, dataset_name=name)
    if table.resize:
        resize_dataset(all_dataset)
    if table.merge:
        # TODO: how to only use merge(-m)??
        merge_dataset(all_dataset)