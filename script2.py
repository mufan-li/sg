'''
Script to test a subset of data
'''

import pandas as pd
import numpy as np

execfile('preprocess.py')

# sgdata_test = sgdata[:50]
# sgdata_pivot = sgdata.pivot(index='ID', \
# 				columns='COURSE', values='GRADE')

# sgdata_matrix = np.asarray(sgdata_pivot)

sgGroup = sgdata[['ID','COURSE','GRADE']].groupby(['ID', 'COURSE'])
# sgAgg = sgGroup.count().reset_index()
# sgAgg[sgAgg['GRADE']>1]

sgdataFilter = sgGroup.last().reset_index()
sgdata_pivot = sgdataFilter.pivot(index='ID', \
				columns='COURSE', values='GRADE')

sgdata_matrix = np.asarray(sgdata_pivot)