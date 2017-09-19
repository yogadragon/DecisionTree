import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="2.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    


def createPlot():    # 绘制树节点
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # axprops = dict(xticks=[], yticks=[])
    # createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    createPlot.ax1 = plt.subplot(111, frameon=False) 
    plotNode('node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('leaf',(0.8,0.1),(0.3,0.8),leafNode)
    # plotTree.totalW = float(getNumLeafs(inTree))
    # plotTree.totalD = float(getTreeDepth(inTree))
    # plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    # plotTree(inTree, (0.5,1.0), '')
    plt.show()





createPlot()