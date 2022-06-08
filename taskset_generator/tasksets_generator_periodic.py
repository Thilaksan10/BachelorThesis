from __future__ import division
import numpy as np
import generator as gen
from io import StringIO
import os
import math
import sys
import getopt


def main(argv):
    ntasks = 5
    msets = 1
    processors = 1
    # num of resources
    res_num = 1
    c_min = 0.05
    c_max = 0.1
    subset=1
    try:
        opts, args = getopt.getopt(argv, "hn:m:p:r:s:l:t:",
                                   ["ntasks=", "msets=", "processors", "res_num=", "c_min=", "c_max=", "subset="])
    except getopt.GetoptError:
        print ('tasksets_generater.py -n <n tasks for each set> -m <m tasksets> -p <num of processors> -r <num of resources> -s <min length of the critical section> -l <max length of the critical section>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('tasksets_generater.py -n <n tasks for each set> -m <m tasksets> -p <num of processors> -r <num of resources> -s <min length of the critical section>-l <max length of the critical section>')
            sys.exit()
        elif opt in ("-n", "--ntasks"):
            ntasks = int(arg)
        elif opt in ("-m", "--msets"):
            msets = int(arg)
        elif opt in ("-p", "--processors"):
            processors = int(arg)
        elif opt in ("-r", "--res_num"):
            res_num = int(arg)
        elif opt in ("-s", "--c_min"):
            c_min = float(arg)
        elif opt in ("-l", "--c_max"):
            c_max = float(arg)
        elif opt in ("-t", "--subset"):
            subset = int(arg)

    for i in range(5, 30, 5):
        # print(i)
        utli = float(i / 100)
        tasksets_name = '../experiments/inputs/input_task_periodic/' + str(subset) + '/tasksets_n' + str(ntasks) + '_m' + str(msets) + '_p' + str(processors) + '_u' + str(
            utli) + '_r' + str(res_num) + '_s' + str(c_min) + '_l' + str(c_max)
        tasksets = gen.generate(ntasks, msets, processors * utli, res_num, 0.5, c_min, c_max, 1)
        np.save(tasksets_name, tasksets)


if __name__ == "__main__":
    main(sys.argv[1:])
