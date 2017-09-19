from math import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import operator

class CreateTree:
	def createDataset(self):
		dataset = [[1,1,'y'],[1,1,'y'],[1,0,'n'],
					[0,1,'n'],[0,1,'n']]
		labels = ['nosurfacing','flippers']
		return dataset,labels

	def calcEnt(self,dataset):    # 计算分布的熵
		m = len(dataset)
		labelcount = {}
		for item in dataset:
			label = item[-1]
			if label not in labelcount.keys():
				labelcount[label] = 0
			labelcount[label] += 1
		ents = 0.0
		for keys in labelcount:
			prop = float(labelcount[keys])/m
			ents -= prop * log(prop,2)
		return ents

	def splitdata(self,dataset,axis,value):   # 按照给定的特征划分数据集,axis为划分的维度,value为划分的值
		retdata = []
		for item in dataset:
			if item[axis] == value:
				reducedata = item[:axis]
				reducedata.extend(item[axis+1:])
				retdata.append(reducedata)
		return retdata

	def bestfreature(self,dataset):   # 计算熵增益,选择我最好的分类特征
		numfreature = len(dataset[0])-1
		baseenstropy = self.calcEnt(dataset)
		best = 0.0
		for i in range(numfreature):
			featurelist = [example[i] for example in dataset]
			uniqe = set(featurelist)
			dens = 0.0
			for item in uniqe:
				retdata = self.splitdata(dataset,i,item)
				prob = float(len(retdata))/len(dataset)
				dens += prob * self.calcEnt(retdata)
			if baseenstropy - dens > best:
				best = baseenstropy - dens
				ibest = i
		return ibest


	def majority(self,classlist):   # 多数表决分类法
		count = {}
		for item in classlist:
			if item not in count.keys():
				count[item] = 0
			count[item] += 1
		sortcount = sorted(count.items(),key=operator.itemgetter(1),reverse=True)
		return sortcount[0][0]

	def creating(self,dataset,labels1):   # 创建树
		labels = labels1
		classlist = [example[-1] for example in dataset]
		if classlist.count(classlist[0]) == len(classlist):
			return classlist[0]
		if len(dataset[0]) == 1:
			return self.majority(classlist)
		bestfeat = self.bestfreature(dataset)
		bestlabel = labels[bestfeat]
		mytree = {bestlabel:{}}
		del(labels[bestfeat])
		featval = [example[bestfeat] for example in dataset]
		uniqeval = set(featval)
		for value in uniqeval:
			sublabel = labels[:]
			mytree[bestlabel][value] = self.creating(self.splitdata(dataset,bestfeat,value),sublabel)
		return mytree


	def getleafnums(self,mytree):
		num = 0
		firststr = list(mytree.keys())[0]
		seconddic = mytree[firststr]
		for key in seconddic.keys():
			if type(seconddic[key]).__name__ == 'dict':
				num += self.getleafnums(seconddic[key])
			else:
				num += 1
		return num

	def getdepth(self,mytree):
		depth = 0
		depth11 = 0
		firststr = list(mytree.keys())[0]
		seconddic = mytree[firststr]
		for key in seconddic:
			if type(seconddic[key]).__name__=='dict':
				depth11 += self.getdepth(seconddic[key])
			else:
				depth11 = 1
			if depth11>depth:
				depth = depth11
		return depth

		

		
decisionNode = dict(boxstyle="sawtooth", fc="2.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = CreateTree().getleafnums(myTree)  
    depth = CreateTree().getdepth(myTree)
    firstStr = list(myTree.keys())[0]     
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))        
        else:   
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    
    #createPlot.ax1 = plt.subplot(111, frameon=False) 
    plotTree.totalW = float(CreateTree().getleafnums(inTree))
    plotTree.totalD = float(CreateTree().getdepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


def classify(inputtree,featlabels,testvec):   #输入树进行分类
	firststr = list(inputtree.keys())[0]
	seconddic = inputtree[firststr]
	featindex = featlabels.index(firststr)
	for key in seconddic.keys():
		if key == testvec[featindex]:
			if type(seconddic[key]).__name__=='dict':
				classlabel = classify(seconddic[key],featlabels,testvec)
			else:
				classlabel = seconddic[key]
	return classlabel

def storetree(inputtree,filename):
	import pickle
	fw = open(filename,'w')
	pickle.dump(inputtree,fw)
	fw.close

def grabtree(inputtree,filename):
	import pickle
	fr = open(filename)
	return pickle.load(fr)








# myDat,mylab = CreateTree().createDataset()
# myDatG = copy.copy(myDat)
# mylabG = copy.copy(mylab)
# # # retdata = CreateTree().splitdata(myDat,0,0)
# # # i = CreateTree().bestfreature(myDat)
# mytree = CreateTree().creating(myDat,mylab)
# a = classify(mytree,mylabG,[1,1])
# print(a)

# # print(mytree)
fr = open('lenses.txt')
lenses = [line.strip().split('\t') for line in fr.readlines()]
lenselabels = ['age','prescript','astigmatic','tearrate']
lenselabelsg = copy.copy(lenselabels)
lensetree = CreateTree().creating(lenses,lenselabels)
createPlot(lensetree)
print(lensetree)

