#!/usr/bin/env python
import pickle
from sklearn import svm
from scipy.io import loadmat
import matplotlib
# matplotlib.use('pdf')
from matplotlib import pyplot as plt
import numpy as np
import colorsys
import time
import csv
from mpl_toolkits.mplot3d import Axes3D

import random

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
make plot 

take 50 random samples out of test_x and map them 
make histogram with vertical bars that have height = # of items in that bin
repeatedly take 50 random samples and plot accuracy over time 

'''




# old code
# old code
# old code
# old code
# old code
# old code
# old code
# old code
# old code
# old code
# old code
# choose k indices to take from - valid indices range from 0-99999 inclusive
# random_image_indices = random.sample(population=[x for x in range(100000)], k=200)
#
# testing_data = []
# print("shape of test data is: " + str(test_data.shape))
# for x in random_image_indices:
#     print(x)
#     testing_data.append(test_data[0:28, 0:28, 0:4, x])
#
# clean_test_data = make_2d_dataHSV(testing_data)
#
# # test_labels = test_labels_FULL[0:MASTER_LIST_LENGTH]
#
#
# # get their corresponding labeled classes
# corresponding_test_indices = [test_labels_FULL[index] for index in random_image_indices]
# corresponding_test_labels = [test_labels_FULL[index] for index in corresponding_test_indices]
#
# # make_predictions(svm_object, clean_test_data, test_labels)
# print('scoring the data')
#
# print(svm_object.score(clean_test_data, corresponding_test_labels))

#end old code
#end old code
#end old code
#end old code
#end old code
#end old code
#end old code
#end old code
#end old code
#end old code
#end old code




'''
Randomly select a subset of size N images and score our svm model on these images 
'''
def score_n_images_randomly(num_images):
    random_image_indices = random.sample(population=[x for x in range(100000)], k=num_images)
    testing_data = []
    print("shape of test data is: " + str(test_data.shape))
    MASTER_LIST_LENGTH = num_images
    for x in random_image_indices:
        print(x)
        testing_data.append(test_data[0:28, 0:28, 0:4, x])

    clean_test_data = make_2d_dataHSV(testing_data)

    # test_labels = test_labels_FULL[0:MASTER_LIST_LENGTH]

    print(test_labels_FULL)

    # get their corresponding labeled classes
    corresponding_test_indices = [test_labels_FULL[index] for index in random_image_indices ]
    print(corresponding_test_indices)
    # make_predictions(svm_object, clean_test_data, test_labels)
    print('scoring the data')
    FINAL_SCORE = svm_object.score(clean_test_data, corresponding_test_indices)
    print(FINAL_SCORE)
    print(svm_object.get_params())
    return FINAL_SCORE



'''
Score the FIRST N images with our svm model 
'''
def score_n_images(num_images):
    # random_image_indices = random.sample(population=[x for x in range(100000)], k=200)
    testing_data = []
    print("shape of test data is: " + str(test_data.shape))
    MASTER_LIST_LENGTH = num_images
    for x in range(MASTER_LIST_LENGTH):
        print(x)
        testing_data.append(test_data[0:28, 0:28, 0:4, x])

    clean_test_data = make_2d_dataHSV(testing_data)

    # test_labels = test_labels_FULL[0:MASTER_LIST_LENGTH]

    print(test_labels_FULL)

    # get their corresponding labeled classes
    corresponding_test_indices = [test_labels_FULL[index] for index in range(MASTER_LIST_LENGTH) ]
    print(corresponding_test_indices)
    # corresponding_test_indices.sort()
    # corresponding_test_labels = [test_labels_FULL[index] for index in corresponding_test_indices]
    # print(corresponding_test_labels)


    # make_predictions(svm_object, clean_test_data, test_labels)
    print('scoring the data')

    print(svm_object.score(clean_test_data, corresponding_test_indices))
    print(svm_object.get_params())



'''
make a plot showing the svm score on a randomly selected set of test data of size n for differnet values of n 
'''
def make_plot1():
    # TODO: LABEL THIS SHIT
    # score_n_images(200)
    score_n_images_randomly(500)
    x_values = [10, 20, 30, 40, 50]
    y_values = [score_n_images_randomly(x) for x in x_values]
    print(x_values)
    print(y_values)
    plt.bar(x_values, y_values)
    plt.show()


'''
get 5 images that are labeled barren land
5 labeled trees 

etc
'''
def get_4_images_saved(n):
    num_to_string = {
        1: 'barren land',
        2: 'trees',
        3: 'grassland',
        4: 'none'
    }

    testing_data = []
    MASTER_LIST_LENGTH = 1000
    for x in range(MASTER_LIST_LENGTH):
        print(x)
        testing_data.append(test_data[0:28, 0:28, 0:3, x])

    barrenland_count = n
    trees_count = n
    grassland_count = n
    none_count = n

    done = barrenland_count == 0 and trees_count == 0 and grassland_count == 0 and none_count == 0


    index = 0
    while not done:
        print(barrenland_count, trees_count, grassland_count, none_count)
        print(done)

        img_label = num_to_string[test_labels_FULL[index]]
        if img_label == 'barren land' and barrenland_count!=0:
            barrenland_count -= 1

        elif img_label == 'trees' and trees_count != 0:
            trees_count -= 1

        elif img_label == 'grassland' and grassland_count != 0:
            grassland_count -= 1

        elif img_label == 'none' and none_count != 0:
            none_count -= 1
        else:
            index += 1
            continue


        # show the image

        plt.imshow(testing_data[index])
        plt.title('Image: ' + str(index) + ', Labeled as: ' + img_label)
        # plt.show()
        plt.savefig(img_label + str(index))
        plt.clf()

        index += 1
        done = barrenland_count == 0 and trees_count == 0 and grassland_count == 0 and none_count == 0



'''
make a plot for points of mean H components vs mean S components of each image 

'''
def compare_two_features(num_images):
    testing_data = []
    for x in range(num_images):
        print(x)
        testing_data.append(test_data[0:28, 0:28, 0:4, x])

    clean_test_data = make_2d_dataHSV(testing_data)

    print(clean_test_data)
    print(type(clean_test_data))
    print(clean_test_data.shape)

    # # h_means_on_x
    #
    # for data in clean_test_data:
    #     print(data)
    #     print(data.shape)
    #     print(data[0])

    h_means_on_x = [data[0] for index, data in enumerate(clean_test_data)]
    s_means_on_y = [data[2] for index, data in enumerate(clean_test_data)]
    v_means_on_z = [data[4] for index, data in enumerate(clean_test_data)]
    #
    print(len(h_means_on_x))
    print(len(s_means_on_y))

    print(h_means_on_x)
    print(s_means_on_y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    # num_to_string = {
    #     1: 'barren land',
    #     2: 'trees',
    #     3: 'grassland',
    #     4: 'none'
    # }
    for index, value in enumerate(h_means_on_x):
        if test_labels_FULL[index] == 1:
            ax.scatter(value, s_means_on_y[index], v_means_on_z[index], 'o', c='blue')
        elif test_labels_FULL[index] == 2:
            ax.scatter(value, s_means_on_y[index], v_means_on_z[index], 'o', c='green')
        elif test_labels_FULL[index] == 3:
            ax.scatter(value, s_means_on_y[index], v_means_on_z[index], 'o', c='red')
        else:
            ax.scatter(value, s_means_on_y[index], v_means_on_z[index], 'o', c='purple')


    ax.set_xlabel('Hue Mean')
    ax.set_ylabel('')


    plt.show()
    # plt.close(ax)
    plt.clf()






def plot_std_3d(num_images):
    testing_data = []
    for x in range(num_images):
        print(x)
        testing_data.append(test_data[0:28, 0:28, 0:4, x])

    clean_test_data = make_2d_dataHSV(testing_data)


    h_stds_on_x = [data[1] for index, data in enumerate(clean_test_data)]
    s_stds_on_y = [data[3] for index, data in enumerate(clean_test_data)]
    v_stds_on_z = [data[5] for index, data in enumerate(clean_test_data)]
    #

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    # num_to_string = {
    #     1: 'barren land',
    #     2: 'trees',
    #     3: 'grassland',
    #     4: 'none'
    # }
    for index, value in enumerate(h_stds_on_x):
        if test_labels_FULL[index] == 1:
            ax.scatter(value, s_stds_on_y[index], v_stds_on_z[index], 'o', c='blue')
        elif test_labels_FULL[index] == 2:
            ax.scatter(value, s_stds_on_y[index], v_stds_on_z[index], 'o', c='green')
        elif test_labels_FULL[index] == 3:
            ax.scatter(value, s_stds_on_y[index], v_stds_on_z[index], 'o', c='red')
        else:
            ax.scatter(value, s_stds_on_y[index], v_stds_on_z[index], 'o', c='purple')


    ax.set_xlabel('Hue Standard Deviation')
    ax.set_ylabel('Saturation Standard Deviation')
    ax.set_zlabel('Value Standard Deviation')


    plt.show()
    # plt.close(ax)
    plt.clf()




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
    # make_plot1()

    compare_two_features(2000)
    plot_std_3d(2000)














































