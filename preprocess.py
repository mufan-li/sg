'''
Script to load and filter out data
	- prints out filter results
	- loads all relevant data into variable sgdata
	- saves variables idFilter, courseFilter
'''

## required imports
# import pandas as pd
# import numpy as np

# import .csv file of data
sgdata_raw = pd.read_csv('allgradesanon2.csv')

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

del sgdata_raw, groupByCourse, groupById, groupByCourseCount
del groupByIdCount, aggByCourse, aggById, aggByCourseCount, aggByIdCount
del n_students, n_students2, n_courses, n_courses2, n_grades, n_grades2


