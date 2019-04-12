
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import StratifiedKFold
from scipy.stats import binom


def discriminatePlot(X, y, cVal, titleStr='', figdir='.', Xcolname = None, plotFig = False, removeTickLabels = False, testInd = None):
    # Frederic's Robust Wrapper for discriminant analysis function.  Performs lda, qda and RF afer error checking, 
    # Generates nice plots and returns cross-validated
    # performance, stderr and base line.
    # X np array n rows x p parameters
    # y group labels n rows
    # rgb color code for each data point - should be the same for each data beloging to the same group
    # titleStr title for plots
    # figdir is a directory name (folder name) for figures
    # Xcolname is a np.array or list of strings with column names for printout display
    # returns: ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses

    # Global Parameters
    CVFOLDS = 10
    MINCOUNT = 10
    MINCOUNTTRAINING = 5
    # figdir = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/Figures Voice'

    # Initialize Variables and clean up data
    classes, classesCount = np.unique(y, return_counts = True)  # Classes to be discriminated should be same as ldaMod.classes_
    goodIndClasses = np.array([n >= MINCOUNT for n in classesCount])
    goodInd = np.array([b in classes[goodIndClasses] for b in y])
    if testInd is not None:
        # Check for goodInd - should be an np.array of dtype=bool
        # Transform testInd into an index inside xGood and yGood
        testIndx = testInd.nonzero()[0]
        goodIndx = goodInd.nonzero()[0]
        testInd = np.hstack([ np.where(goodIndx == testval)[0] for testval in testIndx])
        trainInd = np.asarray([i for i in range(len(goodIndx)) if i not in testInd])
        
    yGood = y[goodInd]
    XGood = X[goodInd]
    cValGood = cVal[goodInd]
        
    classes, classesCount = np.unique(yGood, return_counts = True) 
    nClasses = classes.size         # Number of classes or groups  

    # Do we have enough data?  
    if (nClasses < 2):
        print ('Error in ldaPLot: Insufficient classes with minimun data (%d) for discrimination analysis' % (MINCOUNT))
        return -1, -1, -1, -1 , -1, -1, -1, -1, -1
    
    if testInd is None:
        cvFolds = min(min(classesCount), CVFOLDS)
        if (cvFolds < CVFOLDS):
            print ('Warning in ldaPlot: Cross-validation performed with %d folds (instead of %d)' % (cvFolds, CVFOLDS))
    else:
        cvFolds = 1
   
    # Data size and color values   
    nD = XGood.shape[1]                 # number of features in X
    nX = XGood.shape[0]                 # number of data points in X
    cClasses = []   # Color code for each class
    for cl in classes:
        icl = (yGood == cl).nonzero()[0][0]
        cClasses.append(np.append(cValGood[icl],1.0))
    cClasses = np.asarray(cClasses)
    
    # Use a uniform prior 
    myPrior = np.ones(nClasses)*(1.0/nClasses)  

    # Perform a PCA for dimensionality reduction so that the covariance matrix can be fitted.
    nDmax = int(np.fix(np.sqrt(nX//5)))
    if nDmax < nD:
        print ('Warning: Insufficient data for', nD, 'parameters. PCA projection to', nDmax, 'dimensions.' )
    nDmax = min(nD, nDmax)
    pca = PCA(n_components=nDmax)
    Xr = pca.fit_transform(XGood)
    print ('Variance explained is %.2f%%' % (sum(pca.explained_variance_ratio_)*100.0))
    
    
    # Initialise Classifiers  
    ldaMod = LDA(n_components = min(nDmax,nClasses-1), priors = myPrior, shrinkage = None, solver = 'svd') 
    qdaMod = QDA(priors = myPrior)
    rfMod = RF()   # by default assumes equal weights

        
    # Perform CVFOLDS fold cross-validation to get performance of classifiers.
    ldaYes = 0
    qdaYes = 0
    rfYes = 0
    cvCount = 0
    
    if testInd is None:
        skf = StratifiedKFold(n_splits = cvFolds)
        skfList = skf.split(Xr, yGood)
    else:
        skfList = [(trainInd,testInd)]
    
    for train, test in skfList:
        
        # Enforce the MINCOUNT in each class for Training
        trainClasses, trainCount = np.unique(yGood[train], return_counts=True)
        goodIndClasses = np.array([n >= MINCOUNTTRAINING for n in trainCount])
        goodIndTrain = np.array([b in trainClasses[goodIndClasses] for b in yGood[train]])

        # Specity the training data set, the number of groups and priors
        yTrain = yGood[train[goodIndTrain]]
        XrTrain = Xr[train[goodIndTrain]]

        trainClasses, trainCount = np.unique(yTrain, return_counts=True) 
        ntrainClasses = trainClasses.size
        
        # Skip this cross-validation fold because of insufficient data
        if ntrainClasses < 2:
            continue
        goodInd = np.array([b in trainClasses for b in yGood[test]])    
        if (goodInd.size == 0):
            continue
           
        # Fit the data
        trainPriors = np.ones(ntrainClasses)*(1.0/ntrainClasses)
        ldaMod.priors = trainPriors
        qdaMod.priors = trainPriors
        ldaMod.fit(XrTrain, yTrain)
        qdaMod.fit(XrTrain, yTrain)        
        rfMod.fit(XrTrain, yTrain)
        
        
        ldaYes += np.around((ldaMod.score(Xr[test[goodInd]], yGood[test[goodInd]]))*goodInd.size)
        qdaYes += np.around((qdaMod.score(Xr[test[goodInd]], yGood[test[goodInd]]))*goodInd.size)
        rfYes += np.around((rfMod.score(Xr[test[goodInd]], yGood[test[goodInd]]))*goodInd.size)
        cvCount += goodInd.size


      
# Refit with all the data  for the plots
        
    ldaMod.priors = myPrior
    qdaMod.priors = myPrior
    Xrr = ldaMod.fit_transform(Xr, yGood)
    # Check labels
    for a, b in zip(classes, ldaMod.classes_):
        if a != b:
            print ('Error in ldaPlot: labels do not match')
            
# Check the within-group covariance in the rotated space 
#    covs = []
#    for group in classes:
#        Xg = Xrr[yGood == group, :]
#        covs.append(np.atleast_2d(np.cov(Xg,rowvar=False)))
#    withinCov = np.average(covs, axis=0, weights=myPrior)
  
    # Print the five largest coefficients of first 3 DFA
    MAXCOMP = 3        # Maximum number of DFA componnents
    MAXWEIGHT = 5     # Maximum number of weights printed for each componnent
    
    ncomp = min(MAXCOMP, nClasses-1)
    nweight = min(MAXWEIGHT, nD)
    
    # The scalings_ has the eigenvectors of the LDA in columns and the pca.componnents has the eigenvectors of PCA in columns
    weights = np.dot(ldaMod.scalings_[:,0:ncomp].T, pca.components_)
    
    print('LDA Weights:')
    for ic in range(ncomp):
        idmax = np.argsort(np.abs(weights[ic,:]))[::-1]
        print('DFA %d: '%ic, end = '')
        for iw in range(nweight):
            if Xcolname is None:
                colstr = 'C%d' % idmax[iw]
            else:
                colstr = Xcolname[idmax[iw]]
            print('%s %.3f; ' % (colstr, float(weights[ic, idmax[iw]]) ), end='')
        print()
        
    if plotFig:
        dimVal = 0.8    # Overall diming of background so that points can be seen
        # Obtain fits in this rotated space for display purposes   
        ldaMod.fit(Xrr, yGood)    
        qdaMod.fit(Xrr, yGood)
        rfMod.fit(Xrr, yGood)
    
        XrrMean = Xrr.mean(0)
                
        # Make a mesh for plotting
        x1, x2 = np.meshgrid(np.arange(-6.0, 6.0, 0.1), np.arange(-6.0, 6.0, 0.1))
        xm1 = np.reshape(x1, -1)
        xm2 = np.reshape(x2, -1)
        nxm = np.size(xm1)
        Xm = np.zeros((nxm, Xrr.shape[1]))
        Xm[:,0] = xm1
        if Xrr.shape[1] > 1 :
            Xm[:,1] = xm2
        
        for ix in range(2,Xrr.shape[1]):
            Xm[:,ix] = np.squeeze(np.ones((nxm,1)))*XrrMean[ix]
        
        XmcLDA = np.zeros((nxm, 4))  # RGBA values for color for LDA
        XmcQDA = np.zeros((nxm, 4))  # RGBA values for color for QDA
        XmcRF = np.zeros((nxm, 4))  # RGBA values for color for RF

    
        # Predict values on mesh for plotting based on the first two DFs     
        yPredLDA = ldaMod.predict_proba(Xm) 
        yPredQDA = qdaMod.predict_proba(Xm) 
        yPredRF = rfMod.predict_proba(Xm)

    
        # Transform the predictions in color codes
        maxLDA = yPredLDA.max()
        for ix in range(nxm) :
            cWeight = yPredLDA[ix,:]                               # Prob for all classes
            cWinner = ((cWeight == cWeight.max()).astype('float')) # Winner takes all 
            # XmcLDA[ix,:] = np.dot(cWeight, cClasses)/nClasses
            XmcLDA[ix,:] = np.dot(cWinner*cWeight, cClasses)
            XmcLDA[ix,3] = (cWeight.max()/maxLDA)*dimVal
    
        # Plot the surface of probability    
        plt.figure(facecolor='white', figsize=(10,4))
        plt.subplot(131)
        Zplot = XmcLDA.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
        plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
        if nClasses > 2:
            plt.scatter(Xrr[:,0], Xrr[:,1], c=cValGood, s=40, zorder=1)
        else:
            plt.scatter(Xrr,(np.random.rand(Xrr.size)-0.5)*12.0 , c=cValGood, s=40, zorder=1) 
        plt.title('%s: LDA %d/%d' % (titleStr, ldaYes, cvCount))
        plt.axis('square')
        plt.xlim((-6, 6))
        plt.ylim((-6, 6))    
        plt.xlabel('DFA 1')
        plt.ylabel('DFA 2')

        if removeTickLabels:
            ax = plt.gca()
        
            labels = [item.get_text() for item in ax.get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            ax.set_xticklabels(empty_string_labels)
            
            labels = [item.get_text() for item in ax.get_yticklabels()]
            empty_string_labels = ['']*len(labels)
            ax.set_yticklabels(empty_string_labels)
        
    
        # Transform the predictions in color codes
        maxQDA = yPredQDA.max()
        for ix in range(nxm) :
            cWeight = yPredQDA[ix,:]                               # Prob for all classes
            cWinner = ((cWeight == cWeight.max()).astype('float')) # Winner takes all 
            # XmcLDA[ix,:] = np.dot(cWeight, cClasses)/nClasses
            XmcQDA[ix,:] = np.dot(cWinner*cWeight, cClasses)
            XmcQDA[ix,3] = (cWeight.max()/maxQDA)*dimVal
    
        # Plot the surface of probability  

        plt.subplot(132)
        Zplot = XmcQDA.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
        plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
        if nClasses > 2:
            plt.scatter(Xrr[:,0], Xrr[:,1], c=cValGood, s=40, zorder=1)
        else:
            plt.scatter(Xrr,(np.random.rand(Xrr.size)-0.5)*12.0 , c=cValGood, s=40, zorder=1) 
        plt.title('%s: QDA %d/%d' % (titleStr, qdaYes, cvCount))
        plt.xlabel('DFA 1')
        plt.ylabel('DFA 2')
        plt.axis('square')
        plt.xlim((-6, 6))
        plt.ylim((-6, 6))
           
        if removeTickLabels:
            ax = plt.gca()
            labels = [item.get_text() for item in ax.get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            ax.set_xticklabels(empty_string_labels)
        
            labels = [item.get_text() for item in ax.get_yticklabels()]
            empty_string_labels = ['']*len(labels)
            ax.set_yticklabels(empty_string_labels)
   
        # Transform the predictions in color codes
        maxRF = yPredRF.max()
        for ix in range(nxm) :
            cWeight = yPredRF[ix,:]           # Prob for all classes
            cWinner = ((cWeight == cWeight.max()).astype('float')) # Winner takes all 
            # XmcLDA[ix,:] = np.dot(cWeight, cClasses)/nClasses  # Weighted colors does not work
            XmcRF[ix,:] = np.dot(cWinner*cWeight, cClasses)
            XmcRF[ix,3] = (cWeight.max()/maxRF)*dimVal
    
    # Plot the surface of probability    
        plt.subplot(133)
        Zplot = XmcRF.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
        plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
        if nClasses > 2:    
            plt.scatter(Xrr[:,0], Xrr[:,1], c=cValGood, s=40, zorder=1)
        else:
            plt.scatter(Xrr,(np.random.rand(Xrr.size)-0.5)*12.0 , c=cValGood, s=40, zorder=1) 
            
        plt.title('%s: RF %d/%d' % (titleStr, rfYes, cvCount))
        plt.xlabel('DFA 1')
        plt.ylabel('DFA 2')
        plt.axis('square')
        plt.xlim((-6, 6))
        plt.ylim((-6, 6))
        
        if removeTickLabels:
            ax = plt.gca()
                        
            labels = [item.get_text() for item in ax.get_xticklabels()]
            empty_string_labels = ['']*len(labels)
            ax.set_xticklabels(empty_string_labels)
        
            labels = [item.get_text() for item in ax.get_yticklabels()]
            empty_string_labels = ['']*len(labels)
            ax.set_yticklabels(empty_string_labels)
        
        plt.show()
        plt.savefig('%s/%s.png' % (figdir,titleStr), format='png', dpi=1000)


    # Results
    ldaYes = int(ldaYes)
    qdaYes = int(qdaYes)
    rfYes = int(rfYes)
    
    p = 1.0/nClasses
    ldaP = 0
    qdaP = 0
    rfP = 0
    
    for k in range(ldaYes, cvCount+1):
        ldaP += binom.pmf(k, cvCount, p)
        
    for k in range(qdaYes, cvCount+1):
        qdaP += binom.pmf(k, cvCount, p)
        
    for k in range(rfYes, cvCount+1):
        rfP += binom.pmf(k, cvCount, p)
        
    print ("Number of classes %d. Chance level %.2f %%" % (nClasses, 100.0/nClasses))
    print ("%s LDA: %.2f %% (%d/%d p=%.4f)" % (titleStr, 100.0*ldaYes/cvCount, ldaYes, cvCount, ldaP))
    print ("%s QDA: %.2f %% (%d/%d p=%.4f)" % (titleStr, 100.0*qdaYes/cvCount, qdaYes, cvCount, qdaP))
    print ("%s RF: %.2f %% (%d/%d p=%.4f)" % (titleStr, 100.0*rfYes/cvCount, rfYes, cvCount, rfP))
    return ldaYes, qdaYes, rfYes, cvCount, ldaP, qdaP, rfP, nClasses, weights

