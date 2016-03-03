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
sgdata_raw = pd.read_csv('allgradesanon2.csv')
sgdata_raw['CREDIT'] = 0.5
sgdata_raw.ix[sgdata_raw.WEIGHT == 'Y','CREDIT'] = 1
sgdata_raw.ix[sgdata_raw.GRADE == 0,'GRADE'] = 1
sgdata_raw['UPPER_YEAR'] = sgdata_raw['COURSE'].str[3].astype(int) >= 3

# filter for only math courses
sgdata_raw = sgdata_raw.ix[sgdata_raw.DEPT.isin(
	['MAT','STAT','PHY','CSC','ECO']
	# ['MAT','PHY']
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

groupById = sgdata_raw[['ID','GRADE']].groupby('ID')
aggById = groupById.count().add_suffix('_count').reset_index()

groupByIdCount = aggById.groupby('GRADE_count')
aggByIdCount = groupByIdCount.count().add_suffix('_count').reset_index()

aggByIdCount['c_pct'] = np.cumsum(aggByIdCount['ID_count']) / n_students

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

idFilter = aggById[aggById['GRADE_count']>=10]['ID'].unique()
courseFilter = aggByCourse[aggByCourse['ID_count']>=20]['COURSE'].\
				unique()

sgdata = sgdata_raw[ ( sgdata_raw['ID'].isin(idFilter) ) & \
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
# del groupByIdCount, aggByCourse, aggById, aggByCourseCount, aggByIdCount
# del n_students, n_students2, n_courses, n_courses2, n_grades, n_grades2

#################################
# find department representation

print '... aggregating by department'
groupByIdDept = sgdata[['ID','DEPT','CREDIT']].groupby(['ID','DEPT'])
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

#################################
# filter for repeated courses
# use group.last() assuming the last is most recent

print '... format and output'
sgGroup = sgdata[['ID','COURSE','GRADE']].groupby(['ID', 'COURSE'])
sgdataFilter = sgGroup.last().reset_index()
sgdata_pivot = sgdataFilter.pivot(index='ID', \
				columns='COURSE', values='GRADE')
sgdata_matrix = np.asarray(sgdata_pivot)

# include upper year courses only
sgGroup_uy = sgdata[['ID','COURSE','UPPER_YEAR']].groupby(['ID', 'COURSE'])
sgdataFilter_uy = sgGroup_uy.last().reset_index()
sgdata_pivot_uy = sgdataFilter_uy.pivot(index='ID', \
				columns='COURSE', values='UPPER_YEAR')
sgdata_matrix_uy = np.asarray(sgdata_pivot_uy)

# set to zero
missing_entries = np.isnan(sgdata_matrix)
sgdata_matrix[missing_entries] = 0
sgdata_matrix_uy[missing_entries] = False

sgdata_matrix = (sgdata_matrix * sgdata_matrix_uy).astype('float64')

# rescale to [0,1]
sgdata_matrix = sgdata_matrix/100. 

np.save('sgdata_matrix',sgdata_matrix)
np.save('sgDept_matrix',sgDept_matrix)
np.save('missing_entries',missing_entries)
np.save('sgMaj_matrix',sgMaj_matrix)


