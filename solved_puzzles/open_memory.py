import pickle
import sys, os
import numpy as np
import copy
import argparse


puzzle_size="1x2"
# we only need a single run, so tje default is run_2
filename = 'memory_' + str(puzzle_size) + ".pkl"
# filename = os.path.join(filename)
puzzle_prefixes = []
with open(filename, 'rb') as infile:
    data_dict = pickle.load(infile)
infile.close()
# print("data_dict", data_dict)

# make sure that the data is in the format that we expect it to be:
for puzzle, values in data_dict.items ():
    traject_data_list = list (values)[1:]  # the first entry is the position in which the puzzle was solved, so we ignore this
    assert len (traject_data_list) == 2  # one entry for the list of states, the second for the list of actions
    assert len (traject_data_list[0]) == len (traject_data_list[1])

print(len(data_dict))
print(data_dict.keys())