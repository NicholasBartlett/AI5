###Routine to classify images###
from sys import argv
import Image
import numpy
import os, sys
import math
import shutil
from svmutil import *

if len(argv) != 2 :
  print "Usage: \n  svm.py [imageDirectory] \n\t imageDirectory: root directory containing images"
  exit(1)

script, rootdir = argv
categories = []
testSets = dict()
trainSets = dict()
models = dict()

#make a directory, deleting it if it exists already
def makeDir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	else :
		shutil.rmtree(path)
		os.makedirs(path)
	#print "made directory " + path

#initialize categories, tinyimages, and data structures
def initImages():
	if os.path.exists('tinyimages'):
		shutil.rmtree('tinyimages')
	makeDir('tinyimages')
	makeDir('tinyimages/all')
	for root, dirs, files in os.walk(rootdir):
		path = root.split('/');
		if len(path) > 1 and path[0]==rootdir :
			#looking at a folder category 
			categories.append(path[1])
			makeDir('tinyimages/' + path[1])
		if  len(path) > 1 and path[1] in categories:
			#in the files of a folder, get train and test sets for this category
			trainSets[path[1]] = []
			testSets[path[1]] = []
			fileCount = 0
			numpy.random.shuffle(files)
			for file in files:
				#count number of files
				if file.endswith(".jpg") and not file.startswith("."):
					fileCount += 1
			#calculate size of test and train sets
			trainSize = math.floor(fileCount*.7)
			testSize = fileCount - trainSize
			count = 0
			makeDir('tinyimages/' + path[1] + '/train')
			makeDir('tinyimages/' + path[1] + '/test')
			for file in files: 
				#add files to appropriate sets
				if file.endswith(".jpg") and not file.startswith("."):
					im = Image.open("images/" + path[1] + "/" + file)
					out = im.resize((32, 32))
					if count <= trainSize :
						trainSets[path[1]].append(file)
						out.save('tinyimages/' + path[1] + '/train/' + file)
						out.save('tinyimages/all/' + path[1] + "_" + file)
					else :
						testSets[path[1]].append(file)
						out.save('tinyimages/' + path[1] + '/test/' + file)
					count += 1

#build the models for each category of image
def buildModels():
	for root, dirs, files in os.walk('tinyimages'):
		path = root.split('/');
		if len(path) > 2 and path[2]=='train' and path[1] != 'all':
			trainCount = math.ceil(len(files)/2)
			numpy.random.shuffle(files)
			ctgryTrainFiles = [] 
			ctgryTestFiles = []
			otherTrainFiles = []
			count = 0
			for file in files:
				if count <= trainCount :
					im = Image.open(os.path.join(root, file))
					ivec = numpy.array(im).ravel().tolist()
					ctgryTrainFiles.append(ivec)
					count += 1
				else:
					im = Image.open(os.path.join(root, file))
					ivec = numpy.array(im).ravel().tolist()
					ctgryTestFiles.append(ivec)
			count = 0
			for root2, dirs2, files2 in os.walk('tinyimages/all'):
				numpy.random.shuffle(files2)
				for file in files2:
					if count <= trainCount :
						if not file.startswith(path[1]) :
							im = Image.open(os.path.join(root2, file))
							ivec = numpy.array(im).ravel().tolist()
							otherTrainFiles.append(ivec)
							count += 1
					else:
						break
			
			svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]

			ones = [1] * int(trainCount+1)
			notones = [-1] * int(trainCount+1)
			catType = ones + notones

			print path[1]
			prob = svm_problem(catType, ctgryTrainFiles + otherTrainFiles)

			#figure out best params for linear
			c = 10 ** -15
			maxCLinear = 10 ** -15
			maxAccLinear = -1;
			while c < 10**5 :
				m = svm_train(prob, '-q -t 0 -c ' + str(c))
				p_labels, p_acc, p_vals = svm_predict([1]*len(ctgryTestFiles), ctgryTestFiles, m)
				if(p_acc >= maxAccLinear):
					maxAccLinear = p_acc
					maxCLinear = c
				c = c * 10
			#print str(maxAccLinear[0]) + " with c of " + str(maxCLinear)

			#figure out best params for rbf
			c = 10 ** -15
			maxCRBF = 10 ** -15
			maxAccRBF = -1;
			g = 10 ** -15
			maxGRBF = 10 ** -15
			maxAccRBF = -1;
			while c < 10**5 :
				while g < 10**5 :
					m = svm_train(prob, '-q -t 2 -c ' + str(c) + ' -g ' + str(g))
					p_labels, p_acc, p_vals = svm_predict([1]*len(ctgryTestFiles), ctgryTestFiles, m)
					if(p_acc >= maxAccRBF):
						maxAccRBF = p_acc
						maxCRBF = c
						maxGRBF = g
					g = g * 10
				c = c * 10
			#print str(maxAccRBF[0]) + " with c of " + str(maxCRBF) + " and g of " + str(maxGRBF)

			#store best accuracy model given best parameters as the model to use when testing
			if maxAccRBF[0] > maxAccLinear[0]:
				models[path[1]] = svm_train(prob, '-q -t 2 -c ' + str(maxCRBF) + ' -g ' + str(maxGRBF))
			else :
				models[path[1]] = svm_train(prob, '-q -t 0 -c ' + str(maxCLinear))


#test the models you've created against the previously set aside test images
def testModels():
	print "Results:"
	for root, dirs, files in os.walk('tinyimages'):
		path = root.split('/');
		if len(path) > 2 and path[2]=='test' and path[1] != 'all':
			print "Accuracy for " + path[1] + ":"
			testCount = len(files)
			testFiles = []
			for file in files:
				im = Image.open(os.path.join(root, file))
				ivec = numpy.array(im).ravel().tolist()
				testFiles.append(ivec)
			for key in models.keys():
				catType = []
				if key != path[1]:
					catType = [-1]*testCount
				else:
					catType = [1]*testCount
				p_labels, p_acc, p_vals = svm_predict(catType, testFiles, models[key])

				print "	Accuracy of SVM testing " + str(path[1]) + " images with model " + str(key) + ": " + str(p_acc)









#main routine 
initImages()
buildModels()
testModels()
print models