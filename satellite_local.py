#!/usr/bin/env python
import pickle
from sklearn import svm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import time
import csv

'''
SVM.fit requires a 2 dimensional matrix 
this means we need to 'flatten' each 28x28x3 image into a list of 2352 RGB values
'''
def make_2d_data(training_data):

    data_list = []
    for data in training_data:
        # print(data)
        # print(data.shape)
        list_rgbs = np.array([])
        for i in range(28):
            for j in range(28):
                list_rgbs = np.append(list_rgbs, data[i][j][0])
                list_rgbs = np.append(list_rgbs, data[i][j][1])
                list_rgbs = np.append(list_rgbs, data[i][j][2])

        # print(list_rgbs.shape)
        # two_d_data = np.append(two_d_data, list_rgbs)
        data_list.append(list_rgbs)

    return np.array(data_list)

'''
get the mean and std deviation of the HSV values found within the entire image


'''
def make_2d_dataHSV(training_data):

    data_list = []
    for index, data in enumerate(training_data):
        if index % 500 == 0:
            print('finished processing image ' + str(index))
        # print(data)
        # print(data.shape)
        h_components = np.array([])
        s_components = np.array([])
        v_components = np.array([])
        ir_components = np.array([])
        for i in range(28):
            for j in range(28):
                # list_rgbs = np.append(list_rgbs, data[i][j][0])
                # list_rgbs = np.append(list_rgbs, data[i][j][1])
                # list_rgbs = np.append(list_rgbs, data[i][j][2])
                # TODO: convert to hsv here and take average
                hsv_colors = colorsys.rgb_to_hsv(data[i][j][0], data[i][j][1], data[i][j][2])

                # hsv_mean = np.mean(hsv_colors)
                # hsv_std = np.std(hsv_colors)
                h_components = np.append(h_components, hsv_colors[0])
                s_components = np.append(s_components, hsv_colors[1])
                v_components = np.append(v_components, hsv_colors[2])
                ir_components = np.append(ir_components, data[i][j][3])
        # print(list_rgbs.shape)
        # two_d_data = np.append(two_d_data, list_rgbs)
        # print(list_hsv_mean_std.shape)
        data_list.append([
            np.mean(h_components), np.std(h_components),
            np.mean(s_components), np.std(s_components),
            np.mean(v_components), np.std(v_components),
            np.mean(ir_components), np.std(ir_components)
        ])

    np_data = np.array(data_list)
    print(np_data.shape)
    return np.array(data_list)


'''
Make predictions on the classifications of the 
test_date given the trained support vector machine

Map these results to ascii strings and save to file
'''
def make_predictions(svm, test_data):

    string_mapping = ['barren land', 'trees', 'grassland', 'none']
    predictions = svm.predict(test_data)
    print(predictions)

def score_svm_object(svm, test_data, test_labels):
    # print(svm_object.predict(test_data))
    print(svm.score(test_data, test_labels))
    write_to_csv(svm.predict(test_data))

def write_to_csv(array):


    num_to_string = {
        1: 'barren land',
        2: 'trees',
        3: 'grassland',
        4: 'none'
    }

    with open('landuse.csv', 'w') as csvfile:
        writers = csv.writer(csvfile, delimiter=',')
        string_array = []
        for value in array:
            string_array.append(num_to_string[value])
        print(string_array)


        print(string_array)
        writers.writerow(string_array)

if __name__ == '__main__':

    svm_object = pickle.load(open('model.dat', 'rb'))
    test_labels_FULL = pickle.load(open('test_labels.dat', 'rb'))

    file_list = [
        'test_x_only.mat'
    ]
    print('loading file')
    data_dict = loadmat(file_list[0])
    test_data = data_dict['test_x']



    current_time = time.time()


    num_images = test_data.shape[3]
    print(num_images)
    print(num_images)
    print(num_images)
    print(num_images)
    print(num_images)
    MASTER_LIST_LENGTH = 100
    testing_data = []

    print("shape of test data is: " + str(test_data.shape))
    for x in range(MASTER_LIST_LENGTH):
        testing_data.append(test_data[0:28, 0:28, 0:4, x])

    clean_test_data = make_2d_dataHSV(testing_data)

    test_labels = test_labels_FULL[0:MASTER_LIST_LENGTH]
    # make_predictions(svm_object, clean_test_data, test_labels)
    print('scoring the data')
    score_svm_object(svm_object, clean_test_data, test_labels)

    print('scoring time took: ' + str(time.time()-current_time) + ' seconds')
# for x in range(50):
#     plt.imshow(testing_data[x])
#     plt.show()
#     plt.clf()