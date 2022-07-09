import collections
from concurrent.futures import ThreadPoolExecutor
import string
import subprocess as sp
import os
import os.path as osp
import pdb
import logging
import sys
import shutil
import tqdm

def set_logger():
    """
    set logger to print info to sys.stdout
    """
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log
log = set_logger()


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


def trim_helper(raw_clip_names, all_trim_time):
    before = os.getcwd()
    os.chdir('raw_clips')
    for raw_clip in raw_clip_names:
        make_new_dir(raw_clip, delete=True)  # new dir 'tcs_3', to store trimmed clips 'tcs_3_1', 'tcs_3_2'
        raw_clip_fullname = [i for i in sorted(os.listdir()) if i.startswith(raw_clip) and i.find('.') != -1][0]  # find xxx.mp4
        raw_clip_root, raw_clip_ext = osp.splitext(raw_clip_fullname)
        for i, string in enumerate(all_trim_time[raw_clip], 1):
            if not string.strip():
                i -= 1
                continue
            
            start, end = string.split(' ')
            if len(start) == 6:
                start = start[0: 2] + ':' + start[2: 4] + ':' + start[4: 6]
            if len(end) == 6:
                end = end[0: 2] + ':' + end[2: 4] + ':' + end[4: 6]
            
            command = ('ffmpeg -ss ' + start + ' -to ' + end + ' -i ' + raw_clip_fullname + ' -c copy ' + raw_clip_root + f'_{i}' + raw_clip_ext)
            sp.run(command.split(' '), stdout=sp.DEVNULL, stderr=sp.STDOUT)
            os.rename(raw_clip + f'_{i}' + raw_clip_ext, osp.join(raw_clip, raw_clip + f'_{i}' + raw_clip_ext))  # move tcs_3_1.mp4 into tcs_3
            # print(command)
        log.info(f'{raw_clip_fullname} is trimmed into {i} clips')
    os.chdir(before)


def trim_clip(raw_clip_names):
    all_trim_time = collections.defaultdict(list)
    one_trim_time = []
    one_trim_name = None
    with open('trim.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) < 12:  # name line, not time line
                if one_trim_name:
                    all_trim_time[one_trim_name] = one_trim_time
                    one_trim_time = []
                one_trim_name = line
            else:
                one_trim_time.append(line)
        all_trim_time[one_trim_name] = one_trim_time
    trim_helper(raw_clip_names, all_trim_time)


def dl_helper(command):
    sp.run(command)


def dl(urls):
    os.chdir(r'/Users/stewart/Downloads/Compressed/')
    base_command = "youtube-dl "
    commands = [(base_command + url).split(' ') for url in urls]
    with ThreadPoolExecutor() as e:
        e.map(dl_helper, commands)


def crop_helper(command):
    sp.run(command, stdout=sp.DEVNULL, stderr=sp.STDOUT)


def crop(resize_video):
    os.chdir('label_making')
    before = os.getcwd()
    for dirname in tqdm.tqdm(os.listdir()):
        if dirname[:5] in resize_video:
            video = dirname[:5]
            os.chdir(dirname)
            with ThreadPoolExecutor() as e:
                for file in os.listdir():
                    if file.endswith('jpg'):
                        command = f"ffmpeg -y -i {file} -vf crop={resize_video[video]}:1080:{1920 - resize_video[video]}:0 {file}"
                        e.submit(crop_helper, command.split(' '))

            os.chdir(before)


def build_background_image():
    make_new_dir('background_image')
    with open('background_image.txt', 'r') as f:
        lines = f.readlines()
    existed_background_image = {file for file in os.listdir('background_image')}  # existed background images
    all_background_image = collections.defaultdict(list)
    for line in lines:
        line = line.strip()
        if line[0] in string.ascii_lowercase:  # clip name, starts with a letter
            clip_name = line
        else:  # image name
            all_background_image[clip_name].append(line)

    for clip_name, filenames in all_background_image.items():
        for file in filenames:
            whole_name = clip_name + '_' + file + '.jpg'  # target background image name: ytb_8_1_0.jpg
            if whole_name in existed_background_image:  # old image, already in the folder
                log.info(f'{whole_name} is old image')
                continue
            else:  # new image, copy to the folder
                log.info(f'{whole_name} is new image, copy')
                src = osp.join('label_making', clip_name, whole_name)  # src image path, in raw_clips
                dst = osp.join('background_image', whole_name)  # dst image path, in background_image
                shutil.copyfile(src, dst)


def send_background_image():
    '''
    copy all background images to the dataset folder
    '''
    dst_prefix = 'fast_baseball/images/train'
    src_prefix = 'background_image'
    seen = set(os.listdir(dst_prefix))
    for file in os.listdir('background_image'):
        if file in seen:
            log.info(f'{file} already in dataset')
            continue
        else:
            src = osp.join(src_prefix, file)
            dst = osp.join(dst_prefix, file)
            shutil.copyfile(src, dst)


if __name__ == "__main__":
    # trim_clip([f'speed_{i}' for i in range(3, 9)])
    trim_clip([f'speed_{i}' for i in range(73, 85)])
    # trim_clip(['speed_80', 'speed_81'])
    # trim_clip([f'speed_{i}' for i in range(57, 62)])

    # urls = [
    #     'https://www.youtube.com/watch?v=FSR6WeqWrTM',
    #     'https://www.youtube.com/watch?v=9lFHfd10XAY',
    #     'https://www.youtube.com/watch?v=ArOzltiJtrs',
    #     'https://www.youtube.com/watch?v=ejcTXI3yZw4',
    #     'https://www.youtube.com/watch?v=pFm25_EiUgs',
    # ]
    # dl(urls)

    # crop({'in_27': 1400, 'in_29': 1540})

    # build_background_image()