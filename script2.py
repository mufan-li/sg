'''
Script to test a subset of data
'''

import pandas as pd
import numpy as np
import numpy.linalg as la
import numpy.random as rd

# helper functions
from sg_functions import *

# load and filter the data file
execfile('preprocess.py')

# filter for repeated courses
# use group.last() assuming the last is most recent
sgGroup = sgdata[['ID','COURSE','GRADE']].groupby(['ID', 'COURSE'])
# sgAgg = sgGroup.count().reset_index()
# sgAgg[sgAgg['GRADE']>1]

sgdataFilter = sgGroup.last().reset_index()
sgdata_pivot = sgdataFilter.pivot(index='ID', \
				columns='COURSE', values='GRADE')

sgdata_matrix = np.asarray(sgdata_pivot)
# print sgdata_matrix[:5,:5]

execfile('mf.py')