from scipy.io import loadmat
import pickle
import numpy as np

file_list = [
    'sat-4-full.mat',
    'test_x_only.mat'
]
print('loading file')
data_dict = loadmat(file_list[0])

# pickle.dumps(data_dict['test_y'], open('test_labels.dat', 'wb'))

test_labels = data_dict['test_y']

print(test_labels.shape[1])


img_master_y_list = []
for x in range(test_labels.shape[1]):
    img_master_y_list.append(test_labels[0:4, x])

def clean_messy_data(messy_data):
    return np.array([(np.where(x == 1)[0][0] + 1) for x in messy_data])


clean_data = clean_messy_data(img_master_y_list)

pickle.dump(clean_data, open('test_labels.dat', 'wb'))