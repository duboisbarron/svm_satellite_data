import csv

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
#
#
#
write_to_csv([3, 1, 3, 4, 2])