'''
Script to load and filter out data
	- prints out filter results
	- loads all relevant data into variable sgdata
	- saves variables idFilter, courseFilter
'''

## required imports
import pandas as pd
import numpy as np
from sg_functions import *

print '... prepocessing data'

# import .csv file of data
sgdata_raw = pd.read_csv('allgradesanon3.csv')
sgdata_raw['CREDIT'] = 0.5
sgdata_raw.ix[sgdata_raw.WEIGHT == 'Y','CREDIT'] = 1
sgdata_raw.ix[sgdata_raw.GRADE == 0,'GRADE'] = 1
# sgdata_raw['UPPER_YEAR'] = sgdata_raw['COURSE'].str[3].astype(int) >= 3

# Year starts at 0
sgdata_raw['UPPER_YEAR'] = sgdata_raw['YEAR'] >= 2

# filter for only math courses
sgdata_raw = sgdata_raw.ix[sgdata_raw.DEPT.isin(
	['MAT','STAT','PHY','CSC','ECO','COMPG','ENG','HIS','POL']
	# ['MAT','PHY']
	# ['ECO']
	# ['ENG','HIS','POL']
	)]

# size of data
n_students = len(sgdata_raw['ID'].unique())
n_courses = len(sgdata_raw['COURSE'].unique())
n_grades = len(sgdata_raw)

print '\nraw data:'
print 'number of students: ' + str(n_students)
print 'number of courses: ' + str(n_courses)
print 'number of grades: ' + str(n_grades) + '\n'

#################################
# aggregated data by student
#  - count the number of courses taken per student

# Lower Years
groupById = sgdata_raw[sgdata_raw['UPPER_YEAR'] == False][[\
						'ID','GRADE']].groupby('ID')
aggById = groupById.count().add_suffix('_count').reset_index()

groupByIdCount = aggById.groupby('GRADE_count')
aggByIdCount = groupByIdCount.count().add_suffix('_count').reset_index()

aggByIdCount['c_pct'] = np.cumsum(aggByIdCount['ID_count']) / n_students

# Upper Years
groupById = sgdata_raw[sgdata_raw['UPPER_YEAR']][[\
						'ID','GRADE']].groupby('ID')
aggById2 = groupById.count().add_suffix('_count').reset_index()

# print aggByIdCount

#################################
# aggregate data by course
#  - count the number of students enrolled per course

groupByCourse = sgdata_raw[['ID','COURSE']].groupby('COURSE')
aggByCourse = groupByCourse.count().add_suffix('_count').reset_index()

groupByCourseCount = aggByCourse.groupby('ID_count')
aggByCourseCount = groupByCourseCount.count().add_suffix('_count').\
					reset_index()

aggByCourseCount['c_pct'] = np.cumsum(aggByCourseCount['COURSE_count'])\
								/ n_courses

# print aggByCourseCount

#################################
# take only the data interested #
#################################

idFilter = aggById[aggById['GRADE_count']>=5]['ID'].unique()
idFilter2 = aggById2[aggById2['GRADE_count']>=5]['ID'].unique()
courseFilter = aggByCourse[aggByCourse['ID_count']>=20]['COURSE'].\
				unique()

sgdata = sgdata_raw[ ( sgdata_raw['ID'].isin(idFilter) ) & \
		( sgdata_raw['ID'].isin(idFilter2) ) & \
		( sgdata_raw['COURSE'].isin(courseFilter) ) ]

# size of data
n_students2 = len(sgdata['ID'].unique())
n_courses2 = len(sgdata['COURSE'].unique())
n_grades2 = len(sgdata)

print 'after filter: '
print 'number of students: ' + str(n_students2) + ', ' + \
		str(round(n_students2*100.0 / n_students,2)) + '%'
print 'number of courses: ' + str(n_courses2) + ', ' + \
		str(round(n_courses2*100.0 / n_courses,2)) + '%'
print 'number of grades: ' + str(n_grades2) + ', ' + \
		str(round(n_grades2*100.0 / n_grades,2)) + '%' + '\n'

# del sgdata_raw, groupByCourse, groupById, groupByCourseCount
# del groupByIdCount, aggByCourse, aggById, aggByCourseCount
# del aggByIdCount
# del n_students, n_students2, n_courses, n_courses2, n_grades, n_grades2

#################################
# find department representation

print '... aggregating by department'
groupByIdDept = sgdata[sgdata['UPPER_YEAR']][[\
					'ID','DEPT','CREDIT']].groupby(['ID','DEPT'])
aggByIdDept = groupByIdDept.sum().add_suffix('_TOTAL').reset_index()
groupByIdDeptCred = aggByIdDept.groupby('ID')
aggByIdDeptCred = groupByIdDeptCred.apply(get_major_prop)

sgIdDept_pivot = aggByIdDeptCred.pivot(index = 'ID', \
					columns = 'DEPT', values = 'DEPT_PROP')
sgDept = sgIdDept_pivot.columns.values
sgDept_matrix = np.asarray(sgIdDept_pivot)
sgDept_nan = np.isnan(sgDept_matrix)
sgDept_matrix[sgDept_nan] = 0

sgMaj_argmax = np.argmax(sgDept_matrix, axis = 1)
sgMaj_matrix = np.zeros(np.shape(sgDept_matrix))
for i in range(np.shape(sgDept_matrix)[0]):
	sgMaj_matrix[i,sgMaj_argmax[i]] = 1

# Filter for more than 100 graduates
# total 47
sgDeptColFilter = np.sum(sgMaj_matrix,axis=0) > 100
sgDeptRowFilter = np.sum(sgMaj_matrix[:,sgDeptColFilter], \
							axis=1).astype(bool)

#################################
# filter for repeated courses
# use group.last() assuming the last is most recent

print '... format and output'
sgGroup = sgdata[['ID','COURSE','GRADE']].groupby(['ID', 'COURSE'])
sgdataFilter = sgGroup.last().reset_index()
sgdata_pivot = sgdataFilter.pivot(index='ID', \
				columns='COURSE', values='GRADE')
sgdata_matrix = np.asarray(sgdata_pivot)

# include lower year courses only
sgGroup_ly = sgdata[sgdata['UPPER_YEAR']==False][[\
			'ID','COURSE','GRADE']].groupby(['ID', 'COURSE'])
sgdataFilter_ly = sgGroup_ly.last().reset_index()
sgdata_pivot_ly = sgdataFilter_ly.pivot(index='ID', \
				columns='COURSE', values='GRADE')
sgdata_matrix_ly = np.asarray(sgdata_pivot_ly)

# upper years only
sgGroup_uy = sgdata[sgdata['UPPER_YEAR']][[\
			'ID','COURSE','GRADE']].groupby(['ID', 'COURSE'])
sgdataFilter_uy = sgGroup_uy.last().reset_index()
sgdata_pivot_uy = sgdataFilter_uy.pivot(index='ID', \
				columns='COURSE', values='GRADE')
sgdata_matrix_uy = np.asarray(sgdata_pivot_uy)

# set to zero
missing_entries = np.isnan(sgdata_matrix)
sgdata_matrix[missing_entries] = 0
missing_entries_ly = np.isnan(sgdata_matrix_ly)
sgdata_matrix_ly[missing_entries_ly] = 0
missing_entries_uy = np.isnan(sgdata_matrix_uy)
sgdata_matrix_uy[missing_entries_uy] = 0

# rescale to [0,1]
sgdata_matrix = sgdata_matrix/100. 
sgdata_matrix_ly = sgdata_matrix_ly/100. 
sgdata_matrix_uy = sgdata_matrix_uy/100.

# Filter for majors
sgdata_matrix = sgdata_matrix[sgDeptRowFilter,:]
sgdata_matrix_ly = sgdata_matrix_ly[sgDeptRowFilter,:]
sgdata_matrix_uy = sgdata_matrix_uy[sgDeptRowFilter,:]
sgDept_matrix = sgDept_matrix[sgDeptRowFilter,:][:,sgDeptColFilter]
missing_entries = missing_entries[sgDeptRowFilter,:]
sgMaj_matrix = sgMaj_matrix[sgDeptRowFilter,:][:,sgDeptColFilter]

np.save('sgdata_matrix',sgdata_matrix)
np.save('sgdata_matrix_ly',sgdata_matrix_ly)
np.save('sgdata_matrix_uy',sgdata_matrix_uy)
np.save('sgDept_matrix',sgDept_matrix)
np.save('missing_entries',missing_entries)
np.save('sgMaj_matrix',sgMaj_matrix)


#########
# tests
#########

# groupByDept = sgdata_raw[[\
# 					'DEPT','CREDIT']].groupby(['DEPT'])
# aggByDept = groupByDept.sum().add_suffix('_TOTAL').reset_index()
# aggByDept.sort('CREDIT_TOTAL',ascending=False)[:20]

# depts = sgDept[np.argsort(np.sum(sgMaj_matrix,axis=0))]

# maj_counts = np.sort(np.sum(sgMaj_matrix,axis=0)).astype(int)
# sgdata_raw.ix[sgdata_raw.DEPT == 'WDW','COURSE'].unique()

####################
# naive classifier
####################

# sgMaj_naive_matrix = naive_class(sgdata,sgDeptRowFilter,sgDept)
# naive_er = np.sum(sgMaj_naive_matrix[:,sgDeptColFilter] \
# 				- sgMaj_matrix) \
# 	/ sgMaj_matrix.shape[0]
# print 'Naive Classifier Error: ', naive_er







