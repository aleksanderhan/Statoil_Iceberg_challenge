import csv


read_file = "0.91277258566978192.csv"
write_file = "0.91277258566978192-2.csv"

original = []
with open(read_file, 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        original.append(row[0].split(','))

threshold = 0.05
with open(write_file, 'wb') as csvfile:
    csvfile.write(",".join(original[0]) + "\n")

    for i in xrange(1, len(original)):
        number = float(original[i][1])
        if number > 1.0-threshold:
            rounded = 1.0
        elif number < threshold:
            rounded = 0.0
        else:
            rounded = number

        csvfile.write(",".join([original[i][0], str(rounded)]) + "\n")

