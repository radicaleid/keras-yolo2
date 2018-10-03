import os
import glob
import pickle
import numpy as np
from pyspark.context import SparkContext
from scipy import sparse

data_path = "/lus/scratch/kristyn/atlasml/data"
file_list = glob.glob(data_path + "/*jets/0000*/*.npz")
save_path = "/lus/scratch/aheye/data/atlas/"

sc = SparkContext.getOrCreate()


def fname2data(fname):
    ret_list = []
    comp = np.load(fname)
    raw   = comp["raw"]
    event = comp["event"]
    truth = comp["truth"]
    for i in range(10):
        stack_sparse = []
        for stk in raw[i]:
            stack_sparse.append(sparse.csr_matrix(stk))
        # Prepend index of image to the filename to differentiate and avoid overwriting
        ret_list.append( (fname + "_" + str(i), stack_sparse, event[i], truth[i]) )
    return ret_list 

def save_lists(tup):
    # grab filename, split by directory, take last 3, rejoin with /, remove extension
    path_tail = "/".join(tup[0].split("/")[-3:])
    file_path = save_path + path_tail + ".pickle"
    os.makedirs("/".join(file_path.split("/")[:-1]), exist_ok=True)
    with open(file_path, "wb") as fp:
        pickle.dump([tup[1], tup[2], tup[3]], fp)

# Convert filelist to spark rdd
fl_rdd = sc.parallelize(file_list)

# From filenames convert to sparse scipy arrays
sprs_data = fl_rdd.flatMap(fname2data)

# Save numpy arrays and all other relevant information via pickle
sprs_data.foreach(save_lists)

#print(com_np.take(1))
