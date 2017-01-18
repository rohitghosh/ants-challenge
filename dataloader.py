import glob
import os
import scipy.ndimage
import numpy as np
import csv
import sys
from generate_barcode import get_barcode
import itertools


DATA_PATH = '/data4/ants-challenge'
train_valid_split = 0.8

ant_id_list = [101, 106, 109, 119, 128, 133, 135, 143, 15, 161, 166,
               174, 175, 18, 194, 195, 199, 1, 202, 262, 265, 267, 269, 291,
               295, 298, 324, 331, 334, 33, 353, 36, 397, 428, 429, 42, 43,
               448, 46, 494, 532, 533, 538, 539, 561, 570, 571, 594, 598,
               600, 630, 633, 637, 646, 657, 671, 67, 698, 699, 727, 72,
               756, 758, 763, 764, 76, 77, 790, 797, 818, 819]


def load_csv(file_name):
    """
    Reads CSV at file_name as list of lists for each line.
    """
    if sys.version_info[0] < 3:
        lines = []
        infile = open(file_name, 'rb')
    else:
        lines = []
        infile = open(file_name, 'r', newline='')

    ant_dict = {}
    with infile as f:
        csvreader = csv.reader(f)
        for c, lines in enumerate(csvreader):
            if c == 0:
                continue
            ant_id = lines[0]
            if ant_id not in ant_dict:
                ant_dict[ant_id] = {}
                ant_dict[ant_id]['frames'] = []
                ant_dict[ant_id]['coordiantes'] = []
            if lines[-1]:
                ant_dict[ant_id]['frames'].append(
                                                '{:05d}'.format(int(lines[1])))
                ant_dict[ant_id]['coordiantes'].append([lines[2], lines[3]])

        return ant_dict


def get_padded_barcode(ant_id, xs, ys):
    barcode = get_barcode(ant_id)
    img = np.zeros((xs, ys))
    s1, s2 = barcode.shape
    img[(xs-s1)/2:(xs+s1)/2, (ys-s2)/2:(ys+s2)/2] = barcode
    return img


def data_generator(mode, ant_id, batch_size=2, xs=288, ys=512, s1=270, s2=480):
    ant_dict = load_csv(os.path.join(DATA_PATH, 'training_dataset.csv'))
    spc_ant_dict = ant_dict[ant_id]

    frames = spc_ant_dict['frames']
    frames = [DATA_PATH+'/frames/'+x+'.jpeg' for x in frames]

    padded_barcode = get_padded_barcode(int(ant_id), xs, ys)
    padded_barcode = padded_barcode.reshape(1, 1, xs, ys)

    if mode == 'train':
        images = frames[:int(len(frames)*train_valid_split)]
    elif mode == 'valid':
        images = frames[int(len(frames)*train_valid_split):]

    for index in range(len(images)):
        X = None
        Y1 = None
        Y2 = None
        for file_index, files in enumerate(images[index:index+batch_size]):
            img = scipy.ndimage.imread(files)
            coord = (float(spc_ant_dict['coordiantes'][index + file_index][0]),
                     float(spc_ant_dict['coordiantes'][index + file_index][1]))
            for i in range(0, img.shape[0], s1):
                for j in range(0, img.shape[1], s2):
                    if i+xs <= img.shape[0] and j+ys <= img.shape[1]:
                        crop_img = img[i:i+xs, j:j+ys, :]
                    elif i+xs > img.shape[0] and j+ys <= img.shape[1]:
                        crop_img = img[-xs:, j:j+ys, :]
                    elif j+ys > img.shape[0]and i+xs <= img.shape[0]:
                        crop_img = img[i:i+xs:, -ys:, :]
                    else:
                        crop_img = img[-xs:, -ys:, :]

                    if i <= coord[1] < i+xs and j <= coord[0] < j+ys:
                        y1 = np.asarray([0, 1])
                        y2 = np.asarray([float((coord[1]-i)/xs),
                                        float((coord[0]-j)/ys)])
                        y1 = y1.reshape((1, 2))
                        y2 = y2.reshape((1, 2))
                    else:
                        y1 = np.asarray([1, 0])
                        y2 = np.asarray([0, 0])
                        y1 = y1.reshape((1, 2))
                        y2 = y2.reshape((1, 2))

                    crop_img = crop_img.reshape((1, 3, xs, ys))
                    crop_img = np.concatenate((crop_img, padded_barcode),
                                              axis=1)

                    if X is None:
                        X = crop_img
                        Y1 = y1
                        Y2 = y2
                    else:
                        X = np.concatenate((X, crop_img), axis=0)
                        Y1 = np.concatenate((Y1, y1), axis=0)
                        Y2 = np.concatenate((Y2, y2), axis=0)

        yield [X], [Y1, Y2]


def framewise_load_csv(file_name):
    """
    Reads CSV at file_name as list of lists for each line.
    """
    if sys.version_info[0] < 3:
        lines = []
        infile = open(file_name, 'rb')
    else:
        lines = []
        infile = open(file_name, 'r', newline='')

    frame_dict = {}
    with infile as f:
        csvreader = csv.reader(f)
        for c, lines in enumerate(csvreader):
            if c == 0:
                continue
            frame_id = '{:05d}'.format(int(lines[1]))
            if frame_id not in frame_dict:
                frame_dict[frame_id] = {}
                frame_dict[frame_id]['ant_id'] = []
                frame_dict[frame_id]['coordinates'] = []
            if lines[-1]:
                frame_dict[frame_id]['ant_id'].append(int(lines[0]))
                frame_dict[frame_id]['coordinates'].append([float(lines[2]),
                                                            float(lines[3])])

        return frame_dict


def framewise_data_generator(mode, batch_size=2, xs=288,
                             ys=512, s1=270, s2=480):
    frame_dict = framewise_load_csv(os.path.join(DATA_PATH,
                                    'training_dataset.csv'))
    frames_list = frame_dict.keys()

    if mode == 'train':
        images = frames_list[:int(len(frames_list)*train_valid_split)]
    elif mode == 'valid':
        images = frames_list[int(len(frames_list)*train_valid_split):]

    image_list = [DATA_PATH+'/frames/'+x+'.jpeg' for x in images]

    for index in itertools.cycle(range(len(image_list))):
        X = None
        Y1 = None
        Y2 = None
        for file_index, files in enumerate(image_list[index:index+batch_size]):
            img = scipy.ndimage.imread(files)
            frame_id = images[index + file_index]
            for i in range(0, img.shape[0], s1):
                for j in range(0, img.shape[1], s2):
                    if i+xs <= img.shape[0] and j+ys <= img.shape[1]:
                        crop_img = img[i:i+xs, j:j+ys, :]
                    elif i+xs > img.shape[0] and j+ys <= img.shape[1]:
                        crop_img = img[-xs:, j:j+ys, :]
                    elif j+ys > img.shape[0]and i+xs <= img.shape[0]:
                        crop_img = img[i:i+xs:, -ys:, :]
                    else:
                        crop_img = img[-xs:, -ys:, :]
                    y1 = np.zeros((1, 72))
                    y2 = np.zeros((1, 144))
                    for c, coord in enumerate(
                                    frame_dict[frame_id]['coordinates']):
                        if i <= float(coord[1]) < i+xs and j <= float(
                                                            coord[0]) < j+ys:
                            ant_id = frame_dict[frame_id]['ant_id'][c]
                            cls_no = ant_id_list.index(ant_id) + 1
                            y1[0][cls_no] = 1
                            y2[0][cls_no:cls_no+2] = np.asarray([
                                                float((coord[1]-i)/xs),
                                                float((coord[0]-j)/ys)])

                    crop_img = crop_img.reshape((1, 3, xs, ys,))

                    if X is None:
                        X = crop_img
                        Y1 = y1
                        Y2 = y2
                    else:
                        X = np.concatenate((X, crop_img), axis=0)
                        Y1 = np.concatenate((Y1, y1), axis=0)
                        Y2 = np.concatenate((Y2, y2), axis=0)

        yield [X], [Y1, Y2]


def framewise_valid_data(batch_size=2, nb_val_samples=320, xs=288,
                         ys=512, s1=270, s2=480):
    frame_dict = framewise_load_csv(os.path.join(DATA_PATH,
                                    'training_dataset.csv'))
    frames_list = frame_dict.keys()

    images = frames_list[int(len(frames_list)*train_valid_split):]

    image_list = [DATA_PATH+'/frames/'+x+'.jpeg' for x in images]
    X = None
    Y1 = None
    Y2 = None

    for index in range(len(image_list)):
        files = image_list[index]
        for file_index, files in enumerate(image_list[index:index+batch_size]):
            img = scipy.ndimage.imread(files)
            frame_id = images[index + file_index]
            for i in range(0, img.shape[0], s1):
                for j in range(0, img.shape[1], s2):
                    if i+xs <= img.shape[0] and j+ys <= img.shape[1]:
                        crop_img = img[i:i+xs, j:j+ys, :]
                    elif i+xs > img.shape[0] and j+ys <= img.shape[1]:
                        crop_img = img[-xs:, j:j+ys, :]
                    elif j+ys > img.shape[0]and i+xs <= img.shape[0]:
                        crop_img = img[i:i+xs:, -ys:, :]
                    else:
                        crop_img = img[-xs:, -ys:, :]
                    y1 = np.zeros((1, 72))
                    y2 = np.zeros((1, 144))
                    for c, coord in enumerate(
                                    frame_dict[frame_id]['coordinates']):
                        if i <= float(coord[1]) < i+xs and j <= float(
                                                            coord[0]) < j+ys:
                            ant_id = frame_dict[frame_id]['ant_id'][c]
                            cls_no = ant_id_list.index(ant_id) + 1
                            y1[0][cls_no] = 1
                            y2[0][2*(cls_no):2*(cls_no+1)] = np.asarray([
                                                float((coord[1]-i)/xs),
                                                float((coord[0]-j)/ys)])

                    crop_img = crop_img.reshape((1, 3, xs, ys,))

                    if X is None:
                        X = crop_img
                        Y1 = y1
                        Y2 = y2
                    else:
                        X = np.concatenate((X, crop_img), axis=0)
                        Y1 = np.concatenate((Y1, y1), axis=0)
                        Y2 = np.concatenate((Y2, y2), axis=0)
                        if X.shape[0] >= nb_val_samples:
                            return [X], [Y1, Y2]

if __name__ == '__main__':
    count = 0
    [X], [Y1, Y2] = framewise_valid_data()
    print(X.shape)
    print(Y1.shape)
    print(Y2.shape)
