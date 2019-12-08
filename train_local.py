#!/usr/bin/env python
from scipy import io
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import pickle
import colorsys
import time



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
                # TODO: convert to hsv here and take average

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

        '''
        extracting 8 features total 
        '''
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
SVM.fit requires the y data to be one dimensional of length n_samples
currently the result data is n-dimensional n samples where each sample is of the form [0, 0, 1, 0]
reduce [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]] to [1, 0, 3]
'''
def clean_result_data(messy_result):
    return np.array([(np.where(x == 1)[0][0] + 1) for x in messy_result])


'''
Trains a support vector machine by fitting on the train_data and train_results
returns the trained svm object and saves it to model.dat via pickling
'''
def make_SVM(train_data, train_results):
    print('training')
    svm_object = svm.SVC(max_iter=10000)
    svm_object.fit(train_data, train_results)

    print(svm_object.score(train_data, train_results))
    pickle.dump(svm_object, open('model33.dat', 'wb'))
    print('saved support vector machine to file: model.dat')


def show_images(img_master_list):

    for x in range(5):
        plt.imshow(img_master_list[x])
        plt.show()
        plt.clf()


def drive_training(amount_of_training_data):
    img_master_list = []
    img_master_y_list = []
    training_data = []


    for x in range(amount_of_training_data):
        img_master_y_list.append(train_data_y[:, x])
        training_data.append(train_data[0:28, 0:28, 0:4, x])


    print('collapsing data to 2d')
    two_d_data = make_2d_dataHSV(training_data)
    print('done collapsing data')
    test_results = clean_result_data(img_master_y_list)

    print('making svm')
    make_SVM(two_d_data, test_results)

if __name__ == '__main__':

    file_list = [
        'sat-4-full.mat',
        'test_x_only.mat'
    ]
    print('loading file')
    data_dict = io.loadmat(file_list[0])
    print(data_dict.keys())

    # print(data_dict['train_x'][1, 2, 3, 4])
    train_data = data_dict['train_x']
    train_data_y = data_dict['train_y']

    current_time = time.time()



    drive_training(1000)

    print('training time took: ' + str(time.time() - current_time) + ' seconds ')







