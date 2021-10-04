import pandas
import numpy as np
from scipy import stats
from graphviz import Graph
import matplotlib.pyplot as plt
import warnings
import os
import seaborn as sns

def HoTdiagram(data, PlotName = None, significance_level = 0.99, Gaussian = True, boxplot_ylabel = None,
               NormalizeData = False, LowerAreBest = False, title = True, titleText = None):
    
    if not os.path.exists('Figures/'):
        os.makedirs('Figures/')
    
    # Normalize data
    if NormalizeData:
        data = data.sub(data.mean(1), axis=0).div(data.std(1), axis=0)
    
    # Construct and show the boxplot
    plt.clf()
    # make boxplot with Seaborn
    bplot = sns.boxplot(data=data,width=0.5,palette="colorblind")
    plt.setp(bplot.get_xticklabels(), rotation=90)
    
    if boxplot_ylabel is not None:
        plt.ylabel(boxplot_ylabel)
    
    plt.grid("on")
    plt.tight_layout()
    
    if (title is True) and (titleText is not None):
        plt.title(titleText)
    
    if PlotName == None:
        bplot.figure.savefig('Figures/BoxPlot.eps',dpi = 300)
    else:
        bplot.figure.savefig('Figures/BoxPlot_'+PlotName+'.pdf',dpi = 300)
    
    # If lower values are best - change the order of data.
    if LowerAreBest:
        data = -data
    
    # Construct the graph
    g1 = Graph(engine='dot', filename='Figures/HoTDiagram_'+PlotName, format = 'pdf')
    g1.attr(size='8,8')
    g1.node_attr.update(color='lightblue', style='filled')
    for method in data.columns:
        g1.node(method)
    
    alpha = 1-significance_level
        
    if Gaussian:
        print("Student's t-test (assumes normal distribution).")

        # Hypothesis Test;
        E1 = set()    
        for methodA in data.columns:
            for methodB in data.columns:
                htest = stats.ttest_1samp(data[methodB]-data[methodA],0)        
                if (htest.pvalue < 2*alpha) and (htest.statistic>0):
                    E1.add( (methodB,methodA) )
        if title == True:
            if titleText is not None:
                g1.attr(label=titleText)
            else:
                g1.attr(label="\nHasse diagram of paired Student's t-test\n(confidence level at "+str(100*significance_level)+"%)")
            
    else:
        print("Non-parametric Wilcoxon signed-rank test.")
    
        # Hypothesis Test;    
        E1 = set()
        for methodA in data.columns:
            for methodB in data.columns:
                dif = data[methodB]-data[methodA]
                if np.sum(dif)>0:
                    htest = stats.wilcoxon(dif)
                    if (htest.pvalue < 2*alpha) and (dif.median()>0):
                        E1.add( (methodB,methodA) )
        if title == True:
            if titleText is not None:
                g1.attr(label=titleText)
            else:
                g1.attr(label="\nHasse diagram of Wilcoxon signed-rank test\n(confidence level at "+str(100*significance_level)+"%)")
            
    
    # Determine edges that can be derived from transitivity
    E2 = set()
    for e0 in E1:
        for e1 in E1:
            if e0[1] == e1[0]:
                E2.add( (e0[0],e1[1]) )
    
    # Construc the Hasse diagram by removing edges derived from transitivity
    for e0 in E1 - E2:
        g1.edge(e0[0],e0[1])
    
    if E2-E1:
        for e0 in E2 - E1:
            g1.edge(e0[0],e0[1], color='red')
        warnings.warn("The hypothesis test yielded inconsistent results. The red edges could not be derived from transitivity.")
        
    g1.render()