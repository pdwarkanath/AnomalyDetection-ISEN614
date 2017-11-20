import xlrd  # this library helps work with excel files
from math import log
import numpy #this library has methods to work with matrices
import matplotlib.pyplot as plt
from scipy.stats import chi2

book = xlrd.open_workbook("project_dataset.xlsx")  #open excel file
sh = book.sheet_by_index(0) 


n = sh.nrows
p = sh.ncols

def gmean(listA):
	result = 1
	for i in range(len(listA)):
		result *= listA[i]
	return (result)**(1/(len(listA)))

def amean(listA):
	result = 0
	for i in range(len(listA)):
		result += listA[i]
	return result/(len(listA))

# print(amean([4,8,16]))


def calculateMean(matrixA):
	mean = [] # list to store mean of all columns
	p = matrixA.shape[1]
	for i in range(p): #calculate means of all columns
		mean.append(numpy.mean(numpy.transpose(matrixA)[i]))
	meanArray = numpy.array(mean)
	return meanArray


def calculateS(matrixA):
	emptyPMatrix = []
	p = matrixA.shape[1]
	
	for i in range(p):
		emptyPMatrix.append([]) # add a list to store data from one column
		for j in range(p):
			emptyPMatrix[i].append(0) 

	MatrixS = numpy.matrix(emptyPMatrix) # S matrix in python format
	meanArray = calculateMean(matrixA)
	n = matrixA.shape[0]
	for i in range(n):
		MatrixS = numpy.add(MatrixS, numpy.dot(numpy.transpose(numpy.add(matrixA[i], -1*meanArray)),numpy.add(matrixA[i], -1*meanArray)))

	MatrixS = MatrixS/(n-1)

	return MatrixS


matrixData = [] # store data from excel file in a python readable format

for i in range(n):
	matrixData.append([]) # add a list to store data from one column
	for j in range(p):
		matrixData[i].append(sh.cell_value(i,j)) 

data = numpy.matrix(matrixData) # matrix of project data that can be used with numpy

"""
mean = [] # list to store mean of all columns



for i in range(p): #calculate means of all columns
	mean.append(numpy.mean(numpy.transpose(data)[i]))
"""

meanArray = calculateMean(data)


#print(matrixS.shape)

"""

emptyPMatrix = []

for i in range(p):
	emptyPMatrix.append([]) # add a list to store data from one column
	for j in range(p):
		emptyPMatrix[i].append(0) 


MatrixS = numpy.matrix(emptyPMatrix) # S matrix in python format

# print(MatrixS.shape)


for i in range(n):
	MatrixS = numpy.add(MatrixS, numpy.matmul(numpy.transpose(numpy.add(data[i], -1*meanArray)),numpy.add(data[i], -1*meanArray)))

"""
MatrixS = calculateS(data)

# print(MatrixS) # substitute for epsilon(0), matrix



eigvals = numpy.linalg.eig(MatrixS)


# print(eigvals[0][-1:-1:-1])

# MDL analysis and scree plot commented out below


"""

MDLmin = 100000000  #just some large number to start with
lmin = 0

MDL = []
for l in range(p):
	if l == 0:
		MDL.append(n*(p-l)*log(amean(eigvals[0][-1::-1])/gmean(eigvals[0][-1::-1])) + l*(2*p - l)*log(n)/2)
	else:
		MDL.append(n*(p-l)*log(amean(eigvals[0][-1:l-1:-1])/gmean(eigvals[0][-1:l-1:-1])) + l*(2*p - l)*log(n)/2)
	if MDL[l] < MDLmin:
		MDLmin = MDL[l]
		lmin = l

print(lmin)


plt.plot(range(p),MDL)
plt.xlabel('Principal Components')
plt.ylabel('MDL Values')
plt.show()


plt.plot(range(1,p+1),eigvals[0])
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

"""


lmin = 4 # After looking at scree plot

y = []

for i in range(n):
	y.append([])
	for j in range(lmin):
		y[i].append(numpy.asscalar(numpy.dot(eigvals[1][j],numpy.transpose(data[i]))))

yMatrix = numpy.matrix(y)

# print(yMatrix.shape)


"""
ymean = [] # list to store mean of all columns


for i in range(lmin): #calculate means of all columns
	ymean.append(numpy.mean(numpy.transpose(yMatrix)[i]))

"""

ymeanArray = calculateMean(yMatrix)

# print(ymeanArray)
"""

yemptyPMatrix = []

for i in range(lmin):
	yemptyPMatrix.append([]) # add a list to store data from one column
	for j in range(lmin):
		yemptyPMatrix[i].append(0) 


yMatrixS = numpy.matrix(yemptyPMatrix) # S matrix in python format

for i in range(n):
	yMatrixS = numpy.add(yMatrixS, numpy.matmul(numpy.transpose(numpy.add(yMatrix[i], -1*ymeanArray)),numpy.add(yMatrix[i], -1*ymeanArray)))


"""

yMatrixS = calculateS(yMatrix)

# print(yMatrixS)


def TSquaredArray(matrixA):
	meanArray = calculateMean(matrixA)
	MatrixS = calculateS(matrixA)
	MatrixSinv = numpy.linalg.inv(MatrixS)
	n = matrixA.shape[0]
	p = matrixA.shape[1]
	
	result = []
	for i in range(n):
		result.append(numpy.asscalar(numpy.dot(numpy.dot(numpy.add(matrixA[i],-1*meanArray),MatrixSinv),numpy.transpose(numpy.add(matrixA[i],-1*meanArray)))))
	return result

"""
			if result[i][j] > UCL:
				numpy.delete(matrixA,i)
				TSquaredMatrix(matrixA)
"""

def PhaseIAnalysis(matrixA):
	n = matrixA.shape[0]
	p = matrixA.shape[1]
	alpha = 0.05
	UCL = chi2.ppf(1 - alpha,p)
	Tsquare = TSquaredArray(matrixA)
	plt.plot(range(1,n+1),Tsquare,'bd',range(1,n+1),Tsquare,'b-')
	plt.axhline(y = UCL, color = 'b')
	plt.xlabel('n')
	plt.ylabel('T-squared Hoteling Statistic')
	plt.show()
	for i in range(n):
		if Tsquare[i] > UCL:
			matrixA = numpy.delete(matrixA,(i),axis = 0)
			return PhaseIAnalysis(matrixA)
		else:
			continue
	return Tsquare



yTsquare = PhaseIAnalysis(yMatrix)

print(len(yTsquare))
