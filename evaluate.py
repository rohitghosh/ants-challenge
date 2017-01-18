from keras.models import model_from_json
import glob
import os
import scipy.ndimage
import numpy as np
import csv
import sys
from collections import defaultdict

xs = 288
ys = 512
s1 = 270
s2 = 480

WEIGHTS_LOCATION = '/data_nas/ants/weights/'
ARCHITECTURE_LOCATION = '/data_nas/ants/architectures/'
DATA_PATH = '/data4/ants-challenge'

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

ant_dict = load_csv(os.path.join(DATA_PATH,
                                 'training_dataset.csv'))

best_max_x = 0
best_max_y = 0

for ant_id in ant_dict.keys():
    print (ant_id)
    ant_id = str(ant_id)
    coord_list = ant_dict[ant_id]['coordiantes']
    x = [int(v[0]) for v in coord_list]
    y = [int(v[1]) for v in coord_list]
    max_x = max(x)
    max_y = max(y)
    if max_x > best_max_x:
        best_max_x = max_x
    if max_y > best_max_y:
        best_max_y = max_y

print (best_max_x, best_max_y)

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

            frame_dict[frame_id]['ant_id'].append(int(lines[0]))

        return frame_dict

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)

filename_weights = 'Renewed_AlexNet_lr_0.00_RMSprop_weights.43-1.3787.hdf5'
filename_weights = os.path.join(WEIGHTS_LOCATION, filename_weights)
filename_architecture = 'Renewed_AlexNet_lr_0.00_RMSprop_model_architecture.json'
filename_architecture = os.path.join(ARCHITECTURE_LOCATION,
                                     filename_architecture)

model_json_string = open(filename_architecture, 'r').read()
nn = model_from_json(model_json_string)
nn.load_weights(filename_weights)
nn.compile(loss='mean_squared_error', optimizer='adamax')

frame_dict = framewise_load_csv(os.path.join(DATA_PATH,
                                'testing_dataset.csv'))
frames_list = frame_dict.keys()
image_list = [DATA_PATH+'/frames/'+x+'.jpeg' for x in frames_list]

predict_frame_dict = {}

for index, files in enumerate(image_list[:2]):
    frame_id = frames_list[index]
    img = scipy.ndimage.imread(files)
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
            crop_img = crop_img.reshape((1, 3, xs, ys,))
            probablity = nn.predict(crop_img, verbose=0)
            ant_index = np.where(probablity[0][0] > 0.5)
            for k in range(ant_index[0].shape[0]):
                p = probablity[0][0][ant_index[k]][0]
                ant_id = ant_id_list[ant_index[k]-1]
                x = (max(probablity[1][0][2*ant_index[k]][0], 0)*xs) + i
                y = (max(probablity[1][0][2*ant_index[k] + 1][0], 0)*ys) + j
                if frame_id not in predict_frame_dict:
                    predict_frame_dict[frame_id] = {}
                    predict_frame_dict[frame_id]['ant_id'] = []
                    predict_frame_dict[frame_id]['coordinates'] = []
                    predict_frame_dict[frame_id]['probability'] = []

                predict_frame_dict[frame_id]['ant_id'].append(ant_id)
                predict_frame_dict[frame_id]['coordinates'].append([x, y])
                predict_frame_dict[frame_id]['probability'].append(p)

    # print (ant_dict.keys())
    # for dup in (sorted(list_duplicates(predict_frame_dict[frame_id]['ant_id']))):
    #     print (dup[0])
    #     ant_id = str(dup[0])
    #     coord_list = ant_dict[ant_id]['coordiantes']
    #     x = [int(v[0]) for v in coord_list]
    #     y = [int(v[1]) for v in coord_list]
    #     print (sum(x)/len(x), sum(y)/len(y))
    #     for v in dup[1]:
    #         # ant_id = predict_frame_dict[frame_id]['ant_id'][v]
    #         print (predict_frame_dict[frame_id]['ant_id'][v])
    #         print (predict_frame_dict[frame_id]['probability'][v])
    #         print (predict_frame_dict[frame_id]['coordinates'][v])

# for j,ant_id in enumerate(predict_frame_dict[frame_id]['ant_id']):
#     print (ant_id)
#     ant_id = str(ant_id)
#     coord_list = ant_dict[ant_id]['coordiantes']
#     x = [int(v[0]) for v in coord_list]
#     y = [int(v[1]) for v in coord_list]
#     max_x = max(x)
#     max_y = max(y)
#     if max_x > best_max_x:
#         best_max_x = max_x
#     if max_y > best_max_y:
#         best_max_y = max_y
    # print (sum(x)/len(x), sum(y)/len(y))
    # print (predict_frame_dict[frame_id]['ant_id'][j])
    # print (predict_frame_dict[frame_id]['probability'][j])
    # print (predict_frame_dict[frame_id]['coordinates'][j])


# with open('test_prediction.csv', 'wb') as csvfile:
#     f = csv.writer(csvfile, delimiter=',',
#                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     for ant_id in ant_id_list:
#         for frame_id in predict_frame_dict.keys():
#             if ant_id in predict_frame_dict[frame_id]['ant_id']:
#                 index = predict_frame_dict[frame_id]['ant_id'].index(ant_id)
#                 coord = predict_frame_dict[frame_id]['coordinates'][index]
#                 x, y = coord[0], coord[1]
#                 f.writerow([ant_id, frame_id, x, y, 1])
#             else:
#                 f.writerow([ant_id, frame_id, 0, 0, 0])
