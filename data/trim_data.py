"""
this script is used to trim the original data to a specific size
"""


import random

mb = 1024 * 1024
# generate file size between 3Mb to 5Mb

true_file_size = random.randint(4 * mb, 6 * mb)
fake_file_size = random.randint(4 * mb, 6 * mb)

print("File size is: ", true_file_size)
print("File size is: ", fake_file_size)

train_file_size = true_file_size * 0.7
test_file_size = true_file_size * 0.15
val_file_size = true_file_size * 0.15


with open("./original_data/true.csv", "r") as of, open("./trim_data/true_train.csv", "w") as tf, open("./trim_data/true_test.csv", "w") as tef, open("./trim_data/true_val.csv", "w") as vf:
    readed = 0
    while readed < train_file_size:
        line = of.readline()
        tf.write(line)
        readed += len(line)
    readed = 0
    while readed < test_file_size:
        line = of.readline()
        tef.write(line)
        readed += len(line)
    readed = 0
    while readed < val_file_size:
        line = of.readline()
        vf.write(line)
        readed += len(line)

with open("./original_data/fake.csv", "r") as of, open("./trim_data/fake_train.csv", "w") as tf, open("./trim_data/fake_test.csv", "w") as tef, open("./trim_data/fake_val.csv", "w") as vf:
    readed = 0
    while readed < train_file_size:
        line = of.readline()
        tf.write(line)
        readed += len(line)
    readed = 0
    while readed < test_file_size:
        line = of.readline()
        tef.write(line)
        readed += len(line)
    readed = 0
    while readed < val_file_size:
        line = of.readline()
        vf.write(line)
        readed += len(line)
