import pandas as pd
import numpy as np

# import .csv file of data
sgdata = pd.read_csv('allgradesanon2.csv')

# size of data
n_students = len(sgdata['ID'].unique())
n_courses = len(sgdata['COURSE'].unique())
n_grades = len(sgdata)

print 'number of students: ' + str(n_students)
print 'number of courses: ' + str(n_courses)
print 'number of grades: ' + str(n_grades) + '\n'

#################################
# aggregated data by student
#  - count the number of courses taken per student

groupById = sgdata[['ID','GRADE']].groupby('ID')
aggById = groupById.count().add_suffix('_count').reset_index()

groupByIdCount = aggById.groupby('GRADE_count')
aggByIdCount = groupByIdCount.count().add_suffix('_count').reset_index()

aggByIdCount['c_pct'] = np.cumsum(aggByIdCount['ID_count']) / n_students

# print aggByIdCount

#################################
# aggregate data by course
#  - count the number of students enrolled per course

groupByCourse = sgdata[['ID','COURSE']].groupby('COURSE')
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

sgdata2 = sgdata[ ( sgdata['ID'].isin(idFilter) ) & \
		( sgdata['COURSE'].isin(courseFilter) ) ]

# size of data
n_students2 = len(sgdata2['ID'].unique())
n_courses2 = len(sgdata2['COURSE'].unique())
n_grades2 = len(sgdata2)

print 'after filter: '
print 'number of students: ' + str(n_students2) + ', ' + \
		str(n_students2*1.0 / n_students)
print 'number of courses: ' + str(n_courses2) + ', ' + \
		str(n_courses2*1.0 / n_courses)
print 'number of grades: ' + str(n_grades2) + ', ' + \
		str(n_grades2*1.0 / n_grades) + '\n'







