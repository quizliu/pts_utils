import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import pdb
import math
import os
import os.path as osp
import re
import shutil
import subprocess as sp
import numpy as np
from sklearn.linear_model import HuberRegressor
import logging
from datetime import datetime


# radar gun's velocity in each clips
SPEED = {
    'speed_1': [49, 52, 54, 57, 54, 57, 54, 56, 56, 56, 55, 56, 57, 57, 57],
    'speed_2': [56, 58, 57, 58, 57, 60, 57, 58, 60, 57, 59, 60, 60, 61, 60],
    'speed_5': [50, 51, 51],
    'speed_10': [58, 61, 60, 63, 59],
    'speed_14': [53, 54, 53, 50, 48],
    'speed_48': [63, 61, 61, 51, 59],
    'speed_49': [51, 52, 54, 53],
    'speed_50': [60 ,62, 60, 51, 60, 58, 55, 60, 69, 63],
    'speed_51': [48, 47, 47, 44, 46, 50],
    'speed_52': [61, 58, 60, 51, 54, 50, 53],

    'speed_54': [60, 62, 61, 65, 65, 67, 68, 70, 68, 70],
    'speed_55': [54, 55, 55, 57, 60, 58, 60, 61, 59, 61],
    'speed_56': [69, 69, 53, 70, 72, 69, 70, 70, 71, 72],
    'speed_57': [63, 61, 62, 63, 60, 63, 63, 63, 64, 65],
    'speed_58': [62, 65, 64, 67, 67, 68, 65, 69, 69, 70],
    'speed_59': [56, 57, 59, 61, 61, 62, 62, 64, 62, 63],
    'speed_60': [73, 74, 73, 73, 73, 72, 74, 74, 73, 74],
    'speed_61': [61, 62, 63, 65, 64, 61, 64, 63, 63, 64],

    'speed_64': [59, 63, 64, 69, 64, 66, 65, 65, 64, 67],
    'speed_65': [59, 63, 63, 62, 64, 63, 63, 64, 64, 64],
    'speed_66': [55, 54, 51, 54, 53, 51, 55, 52, 54, 52],
    'speed_67': [63 ,63, 61, 62, 65, 73, 75, 75, 75, 75],
    'speed_68': [62, 61, 61, 64, 65, 65, 64, 66, 68, 64],
    'speed_69': [64, 63, 65, 69, 68, 74, 71, 72, 76, 76],
    'speed_70': [61, 64, 64, 64, 64],
    'speed_71': [62, 64, 63, 64, 65, 63, 62, 53, 64, 60],
    'speed_72': [62, 64, 63, 65, 64, 65, 65, 68, 66, 66],

    'speed_73': [54, 55, 56, 61, 55],
    'speed_74': [56, 55, 58, 57],
    'speed_75': [45, 43, 43, 44, 40],
    'speed_76': [46, 43, 44, 46, 44],
    'speed_77': [55, 57, 55, 53, 52],
    'speed_78': [55, 56, 56, 58, 53],
    'speed_79': [55, 55, 57, 56, 56],
    'speed_80': [58, 57, 55, 57, 58],
    'speed_81': [58, 57, 55, 57, 56],
    'speed_82': [57, 55, 57, 57, 54],
    'speed_83': [55, 57, 49, 55, 51],
    'speed_84': [55, 54, 58, 54, 49],

    'speed_85': [55, 55, 55, 57, 57],
    'speed_86': [61, 58, 58, 59, 60],
    'speed_87': [58, 49, 0.1, 58, 60],  # 0.1 means doesn't have a velocity value, but indeed is a pitch in the video
    'speed_88': [53, 55, 51, 48, 53],
    'speed_89': [53, 53, 51, 0.1, 51],

}

# number of clips each video has
VIDEO = {k: len(v) for k, v in SPEED.items()}

H = [0, 10, 60, 64, 68, 73]  # different player's height, inches, make revision easier

HEIGHT = {  # all in inches, 68 = 5 ft 8 inches
    'speed_54': [68, 73],
    'speed_55': [68, 73],
    'speed_56': [68, 73],
    'speed_57': [68, 73],
    'speed_58': [68, 73],
    'speed_59': [68, 73],
    'speed_60': [68, 73],
    'speed_61': [68, 73],

    'speed_64': [68, 73],
    'speed_65': [68, 73],
    'speed_66': [64, 68],
    'speed_67': [68, 73],
    'speed_68': [68, 73],
    'speed_69': [68, 73],
    'speed_70': [68, 73],
    'speed_71': [68, 73],
    'speed_72': [68, 73],

    'speed_73': [H[2], H[-1]],
    'speed_74': [H[2], H[-1]],
    'speed_75': [H[2], H[-1]],
    'speed_76': [H[2], H[-1]],
    'speed_77': [H[2], H[-1]],
    'speed_78': [H[2], H[-1]],
    'speed_79': [H[2], H[-1]],
    'speed_80': [H[2], H[-1]],
    'speed_81': [H[2], H[-1]],
    'speed_82': [H[2], H[-1]],
    'speed_83': [H[2], H[-1]],
    'speed_84': [H[2], H[-1]],

    'speed_85': [H[3], H[-1]],
    'speed_86': [H[3], H[-1]],
    'speed_87': [H[3], H[-1]],
    'speed_88': [H[3], H[-1]],
    'speed_89': [H[3], H[-1]],
}

# old videos are filmed under 30 fps
FPS_30 = {'speed_1', 'speed_2', 'speed_5', 'speed_52'}

def dis(pa, pb):
    '''
    distance between two points
    '''
    return math.sqrt((pa[0] - pb[0]) * (pa[0] - pb[0]) + (pa[1] - pb[1]) * (pa[1] - pb[1]))

def dis_x(pa, pb):
    '''
    distance at x-axis between two points
    '''
    return abs(pa[0] - pb[0])

def dis_y(pa, pb):
    """
    distance at y-axis between twn points
    """
    return abs(pa[1] - pb[1])

def d2v(d, rlpp, interval):
    '''
    give pixel distance, return velocity in real world(mph)
    d: pixel distance
    rlpp: real length per pixel: how long one pixel stands for in real word
    interval: time interval for dis distance
    '''
    return abs(d) * rlpp * 3600 / interval

def v2d(v, rlpp, interval):
    '''
    give real velocity in real world(mph), return pixel distance on an image
    '''
    return v * interval / (rlpp * 3600)

def get_detect_data(clip_name):
    '''
    change directory to yolov5
    input the clip path
    make the detection and return the label data
    '''
    img = 1280  # gcp3.pt are trained under 1280*1280
    weights = 'gcp3.pt'
    data = 'data/fast_baseball.yaml'
    source = osp.join('calculate_velocity', clip_name + '.mov')
    max_det = 1

    label_dir = 'runs/detect/exp/labels'
    result_dir = 'runs/detect/exp'

    dir_solution = os.getcwd()
    os.chdir('../yolov5')  # enter yolov5 dir

    if osp.isdir(result_dir):
        shutil.rmtree(result_dir)

    if not osp.isfile(source):  # colab should not confuse .mov and .MOV ??
        source = source[:-3] + 'MOV'

    # detect: 1 detection every frame, write labels, write probabilities
    command = 'python detect.py --img ' + str(img) + ' --weights ' + weights + ' --data ' + data + ' --source ' + source + ' --max-det ' + str(max_det) + ' --save-txt --save-conf'
    sp.run(command.split(' '))
    dir_yolov5 = os.getcwd()

    os.chdir(label_dir)  # get label data
    label_data = []

    pattern = re.compile(r'(.*)_(.*)_(.*).txt')
    for file in os.listdir():
        number_of_frame = re.search(pattern, file).groups()[-1]
        with open (file, 'r') as f:
            line = f.readlines()[0].strip()
            line += ' ' + number_of_frame
            label_data.append(line)
    
    os.chdir(dir_yolov5)  # delete this run directory
    shutil.rmtree(result_dir)
    os.chdir(dir_solution)
    return label_data

def get_calibration_data(clip_name):
    """
    change directory to yolov5
    input the clip path
    copy the calibration frame beforehead
    make the detection on this frame and return calibration data
    """
    weights = 'yolov5s6.pt'
    source_img = 'calibration_' + clip_name[:clip_name.rfind('_')] + '.jpg'
    source = osp.join('calculate_velocity',  source_img)
    class_filter = 0  # only detect person
    max_det = 2  # detect pitcher and catcher

    label_dir = 'runs/detect/exp/labels'
    result_dir = 'runs/detect/exp'

    dir_solution = os.getcwd()
    os.chdir('../yolov5')  # enter yolov5 dir

    if osp.isdir(result_dir):
        shutil.rmtree(result_dir)
        
    command = 'python detect.py' + ' --weights ' + weights + ' --source ' + source + ' --max-det ' + str(max_det) + ' --classes ' + str(class_filter) + ' --line-thickness 1' + ' --save-txt --hide-labels'
    sp.run(command.split(' '))
    dir_yolov5 = os.getcwd()

    os.chdir(label_dir)  # get label data
    calibration_data = []

    for file in os.listdir():
        with open(file, 'r') as f:
            for line in f.readlines():
                calibration_data.append(line.strip().split(' ')) 

    os.chdir(dir_yolov5)  # copy calibration image result 
    os.chdir(result_dir)
    shutil.copyfile(source_img, '../' + source_img)

    os.chdir(dir_yolov5)  # delete this run directory
    shutil.rmtree(result_dir)
    os.chdir(dir_solution)
    return calibration_data

def outlier_boxplot_2d(x,y, whis=2):
        xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
        ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

        ##the box
        box = Rectangle(
            (xlimits[0],ylimits[0]),
            (xlimits[2]-xlimits[0]),
            (ylimits[2]-ylimits[0]),
            ec = 'k',
            zorder=0
        )
        # ax.add_patch(box)

        ##the x median
        vline = Line2D(
            [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
            color='k',
            zorder=1
        )
        # ax.add_line(vline)

        ##the y median
        hline = Line2D(
            [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
            color='k',
            zorder=1
        )
        # ax.add_line(hline)

        # the central point
        # ax.plot([xlimits[1]],[ylimits[1]], color='k', marker='o')

        ##the x-whisker
        ##defined as in matplotlib boxplot:
        ##As a float, determines the reach of the whiskers to the beyond the
        ##first and third quartiles. In other words, where IQR is the
        ##interquartile range (Q3-Q1), the upper whisker will extend to
        ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
        ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
        ##the whiskers, data are considered outliers and are plotted as
        ##individual points. Set this to an unreasonably high value to force
        ##the whiskers to show the min and max values. Alternatively, set this
        ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
        ##whiskers at specific percentiles of the data. Finally, whis can
        ##be the string 'range' to force the whiskers to the min and max of
        ##the data.
        iqr = xlimits[2]-xlimits[0]

        ##left
        left = np.min(x[x > xlimits[0]-whis*iqr])
        whisker_line = Line2D(
            [left, xlimits[0]], [ylimits[1],ylimits[1]],
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [left, left], [ylimits[0],ylimits[2]],
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_bar)

        ##right
        right = np.max(x[x < xlimits[2]+whis*iqr])
        whisker_line = Line2D(
            [right, xlimits[2]], [ylimits[1],ylimits[1]],
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [right, right], [ylimits[0],ylimits[2]],
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_bar)

        ##the y-whisker
        iqr = ylimits[2]-ylimits[0]

        ##bottom
        bottom = np.min(y[y > ylimits[0]-whis*iqr])
        whisker_line = Line2D(
            [xlimits[1],xlimits[1]], [bottom, ylimits[0]], 
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [xlimits[0],xlimits[2]], [bottom, bottom], 
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_bar)

        ##top
        top = np.max(y[y < ylimits[2]+whis*iqr])
        whisker_line = Line2D(
            [xlimits[1],xlimits[1]], [top, ylimits[2]], 
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_line)
        whisker_bar = Line2D(
            [xlimits[0],xlimits[2]], [top, top], 
            color = 'k',
            zorder = 1
        )
        # ax.add_line(whisker_bar)

        ##outliers
        mask = (x<left)|(x>right)|(y<bottom)|(y>top)
        # print(mask)
        # ax.scatter(
            # x[mask],y[mask],
            # facecolors='none', edgecolors='k'
        # )
        return mask

def outlier_huber_loss(x, y):
    empirical_rule = [0, 0.6827, 0.9545, 0.9973]  # try 3-sigma rule
    huber = HuberRegressor().fit(x.reshape(-1, 1), y.reshape(-1, 1).ravel())
    w, b = huber.coef_, huber.intercept_
    # outlier_mask = huber.outliers_  # this outlier mask is not suitable 

    predict_gap = abs(huber.predict(x.reshape(-1, 1)) - y)
    # with np.printoptions(precision=3, suppress=True):
    #     print(predict_gap)
    mean, std = np.mean(predict_gap), np.std(predict_gap)
    outlier_mask =( ((predict_gap - mean) / std) > empirical_rule[2]) & (predict_gap > 5)
    return w, b, outlier_mask

def plot_moment(plt, left, right, color, label):
    """
    plot two vertical lines to show where the players are, or when the velocity happens
    """
    plt.axvline(left, color=color, label=label)
    plt.axvline(right, color=color)
    return plt

def arg_parser():
    parser = argparse.ArgumentParser(description='solution A')

    # -v 56 57 58 59
    parser.add_argument('-v', '--video', nargs='+', action='append', metavar='', help='pitch videos')
    parser.add_argument('-r', '--reference', metavar='', default='higher', help='plot with which reference object, higher or shorter or distance')

    args = parser.parse_args()
    args.video = ['speed_' + i for i in args.video[0]]
    # args.height1, args.height2 = int(args.height1), int(args.height2)
    return args

def set_logger():
    """
    set logger to print info to sys.stdout
    """
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.FileHandler(datetime.strftime(datetime.now(), r'%Y-%m-%d-%H:%M:%S') + '.txt', mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(message)s', '%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log

VELOCITY_DATA = 'velocity_data'
CALIBRATION = 'calibration'

class Velocity:
    def __init__(self, clip_name, reference):
        self.clip_name = clip_name
        self.save_path = osp.join(VELOCITY_DATA, f'label_data_{clip_name}.txt')
        self.calibration_path = osp.join(VELOCITY_DATA, CALIBRATION + '_' + clip_name[:clip_name.rfind('_')] + '.txt')

        self.draw_subplot = 1
        self.draw_row = 3
        self.draw_col = 3
        self.plt = plt.figure(figsize=(16, 9))

        self.img_x, self.img_y = 1920, 1080  # resolution
        self.fps = 30 if clip_name[:clip_name.rfind('_')] in FPS_30 else 60  # fps
        self.round = 2  # round digit in functions

        self.pos = []  # original trajectory data, [[position x, position y, number of frame]]
        self.distance = []  # calculated data, [[d, dx, dy, number of intervals, start frame, v, vx, vy]]
        self.frame_pos = {}  # use to plot velocity back on trajectory figure, {frame: [pos x, pos y]}

        self.reference = reference  # use higher or shorter or distance to plot figures

    def save_and_load(self):
        """
        if the clip has no detection data, detect first
        save detection data into txt and load the data
        """
        if not osp.isfile(self.save_path):
            label_data = get_detect_data(self.clip_name)
            with open(self.save_path, 'w') as f:
                for line in label_data:
                    self.pos.append(line)
                    f.write(line + '\n')
        else:
            # print(f'{self.clip_name} is already detected')
            with open(self.save_path, 'r') as f:
                self.pos = f.readlines()
        
        # a list containing each ball's size
        self.ball_size = [(float(i.split(' ')[3]) * self.img_x + float(i.split(' ')[4]) * self.img_y) / 2 for i in self.pos]
        self.pos = [[i.split(' ')[1]] + [i.split(' ')[2]] + [i.split(' ')[-1]] for i in self.pos]  # write x, y, frame number
        self.pos = [[float(i[0]), float(i[1]), int(i[2])] for i in self.pos]  # TODO: only use detection position now, not using probability
        self.pos.sort(key=lambda x: x[2])  # sort by frame number

    def setup(self):
        """
        if the clip has no calibration data, detect that frame first\n
        save calibration data into txt and load data\n
        do the setup for further use
        """
        self.calibration = []  # calibration data
        if not osp.isfile(self.calibration_path):
            calibration_data = get_calibration_data(self.clip_name)
            with open(self.calibration_path, 'w') as f:
                for line in calibration_data:
                    self.calibration.append(line)
                    line = ' '.join(line) + '\n'
                    f.write(line)
        else:
            # print(f'{self.clip_name} has calibration data')
            with open(self.calibration_path, 'r') as f:
                for line in f.readlines():
                    self.calibration.append(line.strip().split(' '))
        
        self.calibration_x = [float(i[1]) for i in self.calibration]  # use distance
        self.calibration_height = [float(i[4]) for i in self.calibration]  # use height

        # setup part
        self.left_player = min(self.calibration_x[0], self.calibration_x[1]) * self.img_x  # use distance
        self.right_player = max(self.calibration_x[0], self.calibration_x[1]) * self.img_x

        self.short_player = min(self.calibration_height[0], self.calibration_height[1]) * self.img_y  # use height
        self.tall_player = max(self.calibration_height[0], self.calibration_height[1]) * self.img_y
 
        pixel_length = abs(self.calibration_x[0] - self.calibration_x[1]) * self.img_x
        real_length_distance = (60 * 12 + 6) / 63360  # miles, 1 mile = 1,760 yards = 5,280 feet = 63,360 inches
        real_length_ball = (3) / 63360  # baseball diameter: 3 inches

        self.rlpp_distance = real_length_distance / pixel_length  # real length per pixel, use distance
        self.rlpp_higher = (max(HEIGHT[self.clip_name[:self.clip_name.rfind('_')]]) / 63360) / (self.tall_player)  # real lenght per pixel, use higher player's height
        self.rlpp_shorter = (min(HEIGHT[self.clip_name[:self.clip_name.rfind('_')]]) / 63360) / (self.short_player)  # real lenght per pixel, use shorter player's height
        self.rlpp_ball = real_length_ball / (sum(self.ball_size) / len(self.ball_size))  # real length per pixel, use ball size
        self.rlpp_average = 1  # calculate average velocity, no need for rlpp here

        self.rlpp_backup = {
            'distance': self.rlpp_distance, 
            'shorter': self.rlpp_shorter, 
            'higher': self.rlpp_higher, 
            'ball': self.rlpp_ball,
            'average': self.rlpp_average,
        }

        self.rlpp = self.rlpp_backup.pop(self.reference)  # use backup to calculate too

        self.interval = 1 / self.fps
        self.limit_v1, self.limit_v2 = 30, 110  # max and min velocity to detect

    def plot_trajectory_normalized_pixel(self):
        """
        plot trajectory in normalized pixel
        """
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'trajectory in normalized pixel, {len(self.pos)} data now')
        plt.xlim([0, 1])
        plt.ylim([-1, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

    def plot_trajectory_real_pixel(self):
        """
        plot trajectory in real pixel
        """
        self.pos = [[i * self.img_x, j * self.img_y, k] for i, j, k in self.pos]  # normalized pixel to the real pixel
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'trajectory in real pixel, {len(self.pos)} data now')
        plt.xlim([0, self.img_x])
        plt.ylim([-self.img_y, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

        for x, y, frame in self.pos:
            self.frame_pos[frame] = [x, y]

    def delete_repeat_detection(self):
        """
        1. all point at the same/nearly same position, delete
        now position data are normalized pixel
        round them and note which ones are detected repeatly
        in position data, round them, delete repeated ones
        still keep the original data(not rounded for further use)
        """
        copy_pos = [[round(x, self.round), round(y, self.round)] for x, y, frame in self.pos]
        seen = defaultdict(int)
        for index, (x, y) in enumerate(copy_pos):
            seen[(x, y)] += 1
        self.pos = [(x, y, frame) for x, y, frame in self.pos if seen[(round(x, self.round), round(y, self.round))] == 1]  # change original data

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'trajectory after deleting repeated points, {len(self.pos)} data now')
        plt.xlim([0, self.img_x])
        plt.ylim([-self.img_y, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

    def plot_velocity_between_detection(self):
        """
        self.distance: [[d, dx, dy, number of interval, start frame, v, vx, vy]]
        d: distance
        dx: distance at x-axis
        dy: distance at y-axis
        number of interval: intervals between two consecutive detections: frame2 - frame1
        start frame: frame1
        v: velocity
        vx: velocity at x-axis
        vy: velocity ay y-axis
        """
        for index, points in enumerate(zip(self.pos, self.pos[1:])):
            (*p1, f1), (*p2, f2) = points
            d, dx, dy = dis(p1, p2), dis_x(p1, p2), dis_y(p1, p2)
            number_of_interval = f2 - f1
            start_frame = f1
            vx = d2v(dx, self.rlpp, number_of_interval * self.interval)
            vy = d2v(dy, self.rlpp, number_of_interval * self.interval)
            v = d2v(d, self.rlpp, number_of_interval * self.interval)
            self.distance.append([d, dx, dy, number_of_interval, start_frame, v, vx, vy])

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'velocity betweeen consecutive detections, {len(self.distance)} data now')
        plt.axis('on')
        # self.distance.sort(key=lambda x: x[3])  # sort by the start frame
        for _, _, _, _, start_frame, v, _, _ in self.distance:
            plt.scatter(start_frame, v)

    def delete_outranged_velocity(self):
        '''
        # 2. all distances that are out of range, delete
        # set a limit range of distance beforehead
        # sort data
        # loop the data, use distance not distance x, if the distance is our of range, delete
        # delete corresponding distance x too 
        '''
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.axis('on')
        new_d = []
        for index, (_, _, _, _, start_frame, v, _, _) in enumerate(self.distance):
            if self.limit_v1 <= v <= self.limit_v2:
                plt.scatter(start_frame, v)
                new_d.append(self.distance[index])
        self.distance = new_d
        plt.title(f'velocity, deleting unvalid ones, {len(self.distance)} data now')

    def delete_outliers_1dbox(self):
        """
        # 3. all distances that are outliers in boxplot, delete
        # use 1-d boxplot to find outliers in distance, not distance x
        # round all distances, round all outliers
        # delete all outliers
        """
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title('1-d boxplot of velocity')
        plt.axis('on')
        ret = plt.boxplot([v for _, _, _, _, _, v, _, _ in self.distance])
        outliers = {round(i, self.round) for i in ret['fliers'][0].get_ydata()}

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.axis('on')
        new_d = []
        for index, (_, _, _, _, start_frame, v, _, _) in enumerate(self.distance):
            if round(v, self.round) not in outliers:
                plt.scatter(start_frame, v)
                new_d.append(self.distance[index])
        self.distance = new_d
        plt.title(f'velocity, deleting outliers in 1d-boxplot, {len(self.distance)}, data now')

    def delete_outliers_2dbox(self):
        """
        4. all distances that are outliers in 2d-boxplot, delete
        similar to 1-d boxplot
        """
        data = [[start_frame, v] for _, _, _, _, start_frame, v, _, _ in self.distance]
        x_data, y_data = np.array([i[0] for i in data]), np.array([i[1] for i in data])
        outlier_mask = outlier_boxplot_2d(x_data, y_data)

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.axis('on')

        for outlier, x, y in zip(outlier_mask, x_data, y_data):
            if outlier:
                plt.scatter(x, y, marker='x', s=35, c='k')
            else:
                plt.scatter(x, y)
        self.distance = [d for d, m in zip(self.distance, outlier_mask) if not m]

        plt.title(f'velocity, deleting outliers in 2d-boxplot, {len(self.distance)} data now')

    def delete_outliers_huber(self):
        data = [[start_frame, v] for _, _, _, _, start_frame, v, _, _ in self.distance]
        x_data, y_data = np.array([i[0] for i in data]), np.array([i[1] for i in data])
        weight, bias, outlier_mask = outlier_huber_loss(x_data, y_data)

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.axis('on')
        plt.plot(x_data.reshape(-1, 1), x_data.reshape(-1, 1) * weight + bias)
        for outlier, x, y in zip(outlier_mask, x_data, y_data):
            if outlier:
                plt.scatter(x, y, marker='x', s=35, c='k')
            else:
                plt.scatter(x, y)

        self.distance = [d for (d, m) in zip(self.distance, outlier_mask) if not m]
        plt.title(f'velocity, deleting outliers using huber regression, {len(self.distance)} data now')

    def update_pos_after_distance(self):
        """
        used for average velocity: average velocity needs all valid position data
        use velocity data to rule out invalid position data

        self.distance: [[d, dx, dy, number of interval, start frame, v, vx, vy]]
        """

    def calculate_and_save(self):
        """
        
        """
        index = self.clip_name.rfind('_')
        video, clip = self.clip_name[:index], int(self.clip_name[index + 1:])
        self.real_v = real_v = SPEED[video][clip - 1]

        self.max_v = max_v = max(v for _, _, _, _, _, v, _, _ in self.distance)
        error_max = (max_v - real_v) / real_v * 100

        # calculate result and error with all reference objects
        ref, err = [], []
        for k, v in self.rlpp_backup.items():
            ref.append(k)
            if k == 'average':  # try use whole distance and whole time interval to get a average velocity
                # average_v = 3600 * ((60 * 12 + 6) / 63360) / ((self.pos[-1][-1] - self.pos[0][-1]) * self.interval)
                average_v = 3600 * ((60 * 12 + 6) / 63360) / ((self.distance[-1][4] - self.distance[0][4] + 1) * self.interval)
                err.append((average_v - real_v) / real_v * 100)
            else:
                err.append((max_v / self.rlpp * v - real_v) / real_v * 100)
        log_backup_string = ', '.join([f'{r}: {e:.1f}%' for r, e in zip(ref, err)])  # string that gives results in backup
        log_string = f'{self.clip_name} {self.reference}: {error_max:.1f}%, ' + log_backup_string  # all results
        log.info(log_string)

        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'final figure, with real velocity')

        for _, _, _, _,start_frame, v, _, _ in self.distance:
            plt.scatter(start_frame, v)
        plt.axhline(real_v, color='g', label=f'real velocity {real_v:.1f}')  # horizontal line, real velocity
        plt.axhline(max_v, color='r', label=f'max velocity {max_v:.1f}')  # horizontal line, max velocity
        plt.legend()

    def plot_velocity_on_trajectory(self):
        """
        visualize where the velocity happens on the original trajectory
        """
        error = (self.max_v - self.real_v) / self.real_v * 100
        plt.subplot(self.draw_row, self.draw_col, self.draw_subplot)
        self.draw_subplot += 1
        plt.title(f'trajectory with the max velocity, error={error:.1f}%')

        plt.xlim([0, self.img_x])
        plt.ylim([-self.img_y, 0])
        plt.axis('on')
        for x, y, frame in self.pos:
            plt.scatter(x, -y)

        # find where is the max velocity and real velocity and tricky velocity on trajectory
        max_frame = real_frame = 0
        max_intv = real_intv = 0
        
        min_real_error = float('inf')  # use error to find which velocity is the closest to the real velocity

        for _, _, _, number_of_interval, start_frame, v, _, _ in self.distance:
            if v == self.max_v:
                max_frame = start_frame
                max_intv = number_of_interval
            if abs(v - self.real_v) < min_real_error:
                min_real_error = abs(v - self.real_v)
                real_frame = start_frame
                real_intv = number_of_interval

        max_v_start, _ = self.frame_pos[max_frame]
        max_v_end, _ = self.frame_pos[max_frame + max_intv]
        real_v_start, _ = self.frame_pos[real_frame]
        real_v_end, _ = self.frame_pos[real_frame + real_intv]

        handle = plot_moment(plt, max_v_start, max_v_end, 'r', 'max velocity range')  # range of max velocity
        handle = plot_moment(handle, real_v_start, real_v_end, 'g', 'real velocity range(estimated)')  # range of real velocity

        # find where is the players
        handle = plot_moment(plt, self.left_player, self.right_player, 'b', label='players')

        handle.legend()
        handle.savefig(osp.join(VELOCITY_DATA, self.clip_name + '.png'))
        handle.close()
 
    def main(self):
        self.save_and_load()
        self.setup()

        # self.plot_trajectory_normalized_pixel()  # 9 figures for now so don't plot this one.
        # if so or adding more figures, remember to revise number of row and col in class Velocity.
        self.plot_trajectory_real_pixel()
        self.delete_repeat_detection()
        self.plot_velocity_between_detection()
        self.delete_outranged_velocity()

        # self.delete_outliers_1dbox()
        self.delete_outliers_huber()
        self.delete_outliers_2dbox()

        self.calculate_and_save()
        self.plot_velocity_on_trajectory()

if __name__ == '__main__':
    table = arg_parser()
    log = set_logger()
    log.info(table)
    for video in table.video:
        for pitch in range(1, VIDEO[video] + 1):
            v = f'{video}_{pitch}'
            V = Velocity(v, reference=table.reference)  # if not 1080p or want to use different reference, input the keyword arguments
            V.main()
        log.info('==============================')
        print(f'video {video} finished')