"""
pitch: know start, known end, 1 pitch: input pitch
video: know all data, know x pitches, input video
real time: unknown pitches: input is a video stream, generate a result as long as its available

algorithm: 
    1. when there are A consecutive valid detection: start
    2. when there are B consecutive frames that has no detection: stop

multithreading(?):
    one thread to read the data: txt file on pc, video stream detection result on camera
    one thread to calculate: got data in the queue at one time, ......?????
"""
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import os
import os.path as osp
import time
import math
import collections
import pdb
from threading import Thread
from queue import Queue
import subprocess as sp

img_x, img_y = 1920, 1080

def set_logger():
    """
    set logger to log.info info to sys.stdout
    """
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(osp.join('RealtimeTrimResult', datetime.strftime(datetime.now(), r'%Y-%m-%d-%H-%M-%S') + '.txt'), mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')  # many result now, shorten the log string
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log

def timer(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        r = fn(*args, **kwargs)
        end = time.time()
        log.info(f"{fn.__name__} uses: {end - start:.4f}s")
        return r

    return wrapper

def read_txt_file(filename, t=0):
    time.sleep(t)
    with open(filename, 'r') as f:
        line = f.readline().strip().split(' ')[1:]
        line.append(filename[filename.rfind('_') + 1: -4])  # x, y, width, height, probability, frame
        return line

def dis(pa, pb):
    '''
    distance between two points
    '''
    return math.sqrt((pa[0] - pb[0]) * (pa[0] - pb[0]) + (pa[1] - pb[1]) * (pa[1] - pb[1]))

def d2v(d, rlpp, interval):
    '''
    give pixel distance, return velocity in real world(mph)
    d: pixel distance
    rlpp: real length per pixel: how long one pixel stands for in real word
    interval: time interval for dis distance, can be (frame2 - frame1) * 1 / fps
    '''
    return abs(d) * rlpp * 3600 / interval    

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

def draw_all_velocity():
    for i in range(85, 89):
        label_dir = f'labels_{i}'
        minframe = float('inf')
        maxframe = float('-inf')
        count = 0
        current_dir = osp.join(os.getcwd(), label_dir)
        plt.figure(figsize=(16, 9))
        for index, txtfile in enumerate(os.listdir(label_dir)):
            count += 1
            x, y, _, _, _, frame = read_txt_file(osp.join(current_dir, txtfile))
            x, y, frame = img_x * float(x), img_y * float(y), int(frame)
            minframe, maxframe = min(frame, minframe), max(frame, maxframe)
            if index > 0:
                velocity = d2v(dis((x, y), (last_x, last_y)), RLPP[i], (frame - last_frame) * 1 / 60)
                plt.scatter(frame, velocity, s=3)
            last_x, last_y, last_frame = x, y, frame
        
        timestamps = all_trim_time[f'speed_{i}']
        for string in timestamps:
            start, end = string.split(' ')
            start, end = int(start[2:4]) * 60 + int(start[4:6]), int(end[2:4]) * 60 + int(end[4:6])
            plt.axvline(start * 60)
            plt.axvline(end * 60)


        plt.xlim([minframe, maxframe])
        plt.title(f'video #{i} velocity')
        plt.savefig(f'velocity_data_video{i}')
        log.info(f'draw all velocity and timestamp: video {i} finished')

RLPP = {
    85: 6.509297647066367e-06,
    86: 6.4008008908785775e-06,
    87: 6.472725674726573e-06,
    88: 6.472725674726573e-06,
}

class RealTime:
    """
    how to do the trimming work from a video stream?
    1. 2 consecutive repeat position, delete both position, also write this position into set()
    2. 2 consecutive position, invalid velocity, delete TODO: delete which one?
    3. when to start: whenever a valid position happens(TODO: or how many (a) valid position happens?)
    4. when to end: too many in the window(TODO: #fps?), or new_frame is too larger than old_frame(TODO: how many?)
    5. check: is this window actually a pitch? (TODO: how to check: use a ratio = valid data / window length ?)
    """
    def __init__(self, video_number):
        self.database = 'VideoDetect'
        self.video_number = video_number
        self.digit = 2
        self.fps = 60

        self.q = Queue()  # queue stores position data
        self.finish_read = False  # is the read worker still reading?
        self.window = []  # 
        self.repeat_position = set()  # store positions that are repetitive

        self.v1, self.v2 = 20, 110

        self.old_x = self.old_y = self.old_frame = None

        self.window_length = 80  # condition to trim: max frame difference in a pitch is 60
        self.no_data_length = 10  # condition to trim: max no-data interval is 10
        self.window_contain = 30  # check the trimmed pitch: at least 30 data

        self.pitches = []  # pitch data

    def frame2timestamp(self, frame_start, frame_end):
        start_seconds, end_seconds = frame_start / self.fps, frame_end / self.fps
        (start_minutes, start_seconds), (end_minutes, end_seconds) = divmod(start_seconds, 60), divmod(end_seconds, 60)
        # start_seconds, end_seconds = math.floor(start_seconds), math.ceil(end_seconds)
        # start_seconds, end_seconds = round(start_seconds), math.ceil(end_seconds)
        start_minutes, end_minutes = int(start_minutes), int(end_minutes)

        return f'00:{start_minutes:0>2}:{start_seconds}', f'00:{end_minutes:0>2}:{end_seconds}'

    def read_all(self, *filelist):
        for file in filelist:
            x, y, _, _, _, frame = read_txt_file(file)
            x, y, frame = img_x * float(x), img_y * float(y), int(frame)
            self.q.put((x, y, frame))
        self.finish_read = True

    def check_repeat(self, new_x, new_y):
        new_pos = (round(new_x, self.digit), round(new_y, self.digit))
        old_pos = (round(self.old_x, self.digit), round(self.old_y, self.digit))
        if new_pos in self.repeat_position:  # set has this one
            return False
        if old_pos == new_pos:  # new and old are repeat data
            self.repeat_position.add(new_pos)
            return False
        return True
        
    def check_neighbour_velocity(self, new_x, new_y, new_frame):
        interval = (new_frame - self.old_frame) * 1 / self.fps
        d = dis((new_x, new_y), (self.old_x, self.old_y))
        v = d2v(d, RLPP[self.video_number], interval)
        if not self.v1 <= v <= self.v2:  # invalid velocity
            return False
        return True 

    def trim_pitch(self):
        self.pitches.append(self.window)
        log.info(f'pitch #{len(self.pitches)}: {len(self.window)} data, start {self.window[0][2]}, end {self.window[-1][2]}')
        start_frame, end_frame = self.window[0][2], self.window[-1][2]
        start_timestamp, end_timestamp = self.frame2timestamp(start_frame, end_frame)

        original_video_filename = osp.join(self.database, str(self.video_number), f'speed{self.video_number}.mp4')
        trim_pitch_filename = osp.join('RealtimeTrimResult', f'{self.video_number}_pitch_{len(self.pitches)}.mp4')
        if osp.isfile(trim_pitch_filename):
            os.remove(trim_pitch_filename)
        command = 'ffmpeg -ss ' + start_timestamp + ' -to ' + end_timestamp + ' -i ' + original_video_filename + ' -c copy ' + trim_pitch_filename
        # print(command)
        sp.run(command.split(' '), stdout=sp.DEVNULL, stderr=sp.STDOUT)

    def business_logic(self, t=0):
        while not self.finish_read or not self.q.empty():  # still reading, or still data in queue, make sure there will be data in queue
            time.sleep(t)
            new_x, new_y, new_frame = self.q.get(block=True)  # block because there must be some data in queue
            if not self.old_x:  # nothing to do with the first data
                self.old_x, self.old_y, self.old_frame = new_x, new_y, new_frame

            else:  # have old data, check repetition and invalid velocity
                if (
                    self.check_repeat(new_x, new_y) 
                    # and 
                    # self.check_neighbour_velocity(new_x, new_y, new_frame)
                ):  # valid position data
                    self.window.append((new_x, new_y, new_frame))
                    if new_frame - self.window[0][2] >= self.window_length:  # check whether to trim this window as a pitch
                        if len(self.window) >= self.window_contain:  # at least some data in this pitch
                            # log.info('pitch happens')
                            self.trim_pitch()
                        else:  # otherwise this is not a pitch
                            # log.info('not a real pitch')
                            ...
                        self.window.clear()
                self.old_x, self.old_y, self.old_frame = new_x, new_y, new_frame

    def try_threading(self):
        label_dir = osp.join(self.database, str(self.video_number), 'labels')
        current_dir = osp.join(os.getcwd(), label_dir)

        worker_read = Thread(target=self.read_all, args=[osp.join(current_dir, txt) for txt in sorted(os.listdir(label_dir), key=lambda x: int(x[x.rfind('_') +1: -4]))], name='worker_read')
        worker_logic = Thread(target=self.business_logic, args=[0.001], name='worker_logic')
        worker_read.start()
        worker_logic.start()
        worker_read.join()
        worker_logic.join()
    
    def main(self):
        log.info(f'video #{self.video_number}')
        self.try_threading()

        
if __name__ == '__main__':
    log = set_logger()
    for i in range(74, 90):
        RealTime(i).main()
        timestamps = all_trim_time[f'speed_{i}']
        for index, string in enumerate(timestamps, 1):
            start, end = string.split(' ')
            start, end = int(start[2:4]) * 60 * 60 + int(start[4:6]) * 60, int(end[2:4]) * 60 * 60 + int(end[4:6]) * 60
            log.info(f'real #{index}: start {start} end {end}')
        log.info('=================================')
        print(f'video #{i} finished')
    # draw_all_velocity()
