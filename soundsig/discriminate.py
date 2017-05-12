import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn import cross_validation


def discriminatePlot(X, y, cVal, titleStr=''):
    # Frederic's Robust Wrapper for discriminant analysis function.  Performs lda, qda and RF afer error checking, 
    # Generates nice plots and returns cross-validated
    # performance, stderr and base line.
    # X np array n rows x p parameters
    # y group labels n rows
    # rgb color code for each data point - should be the same for each data beloging to the same group
    # titleStr title for plots
    # returns: ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses

    # Global Parameters
    CVFOLDS = 10
    MINCOUNT = 10
    MINCOUNTTRAINING = 5
    figdir = '/Users/frederictheunissen/Documents/Data/Julie/Acoustical Analysis/Figures Voice'

    # Initialize Variables and clean up data
    classes, classesCount = np.unique(y, return_counts = True)  # Classes to be discriminated should be same as ldaMod.classes_
    goodIndClasses = np.array([n >= MINCOUNT for n in classesCount])
    goodInd = np.array([b in classes[goodIndClasses] for b in y])
    yGood = y[goodInd]
    XGood = X[goodInd]
    cValGood = cVal[goodInd]


    classes, classesCount = np.unique(yGood, return_counts = True) 
    nClasses = classes.size         # Number of classes or groups  

    # Do we have enough data?  
    if (nClasses < 2):
        print 'Error in ldaPLot: Insufficient classes with minimun data (%d) for discrimination analysis' % (MINCOUNT)
        return -1, -1, -1, -1 , -1, -1, -1
    cvFolds = min(min(classesCount), CVFOLDS)
    if (cvFolds < CVFOLDS):
        print 'Warning in ldaPlot: Cross-validation performed with %d folds (instead of %d)' % (cvFolds, CVFOLDS)
   
    # Data size and color values   
    nD = XGood.shape[1]                 # number of features in X
    nX = XGood.shape[0]                 # number of data points in X
    cClasses = []   # Color code for each class
    for cl in classes:
        icl = (yGood == cl).nonzero()[0][0]
        cClasses.append(np.append(cValGood[icl],1.0))
    cClasses = np.asarray(cClasses)
    myPrior = np.ones(nClasses)*(1.0/nClasses)  

    # Perform a PCA for dimensionality reduction so that the covariance matrix can be fitted.
    nDmax = int(np.fix(np.sqrt(nX/5)))
    if nDmax < nD:
        print 'Warning: Insufficient data for', nD, 'parameters. PCA projection to', nDmax, 'dimensions.' 
    nDmax = min(nD, nDmax)
    pca = PCA(n_components=nDmax)
    Xr = pca.fit_transform(XGood)
    print 'Variance explained is %.2f%%' % (sum(pca.explained_variance_ratio_)*100.0)
    
    
    # Initialise Classifiers  
    ldaMod = LDA(n_components = min(nDmax,nClasses-1), priors = myPrior, shrinkage = None, solver = 'svd') 
    qdaMod = QDA(priors = myPrior)
    rfMod = RF()   # by default assumes equal weights

        
    # Perform CVFOLDS fold cross-validation to get performance of classifiers.
    ldaScores = np.zeros(cvFolds)
    qdaScores = np.zeros(cvFolds)
    rfScores = np.zeros(cvFolds)
    skf = cross_validation.StratifiedKFold(yGood, cvFolds)
    iskf = 0
    
    for train, test in skf:
        
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
        

        ldaScores[iskf] = ldaMod.score(Xr[test[goodInd]], yGood[test[goodInd]])
        qdaScores[iskf] = qdaMod.score(Xr[test[goodInd]], yGood[test[goodInd]])
        rfScores[iskf] = rfMod.score(Xr[test[goodInd]], yGood[test[goodInd]])

        iskf += 1
     
    if (iskf !=  cvFolds):
        cvFolds = iskf
        ldaScores.reshape(cvFolds)
        qdaScores.reshape(cvFolds)
        rfScores.reshape(cvFolds)
      
# Refit with all the data  for the plots
        
    ldaMod.priors = myPrior
    qdaMod.priors = myPrior
    Xrr = ldaMod.fit_transform(Xr, yGood)
    # Check labels
    for a, b in zip(classes, ldaMod.classes_):
        if a != b:
            print 'Error in ldaPlot: labels do not match'
  
    # Print the coefficients of first 3 DFA 
    print 'LDA Weights:'
    print 'DFA1:', ldaMod.coef_[0,:]
    if nClasses > 2:
        print 'DFA2:', ldaMod.coef_[1,:] 
    if nClasses > 3:
        print 'DFA3:', ldaMod.coef_[2,:] 
        
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
        XmcLDA[ix,:] = np.dot(cWinner, cClasses)
        XmcLDA[ix,3] = cWeight.max()/maxLDA
    
    # Plot the surface of probability    
    plt.figure(facecolor='white', figsize=(10,3))
    plt.subplot(131)
    Zplot = XmcLDA.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
    plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
    if nClasses > 2:
        plt.scatter(Xrr[:,0], Xrr[:,1], c=cValGood, s=40, zorder=1)
    else:
        plt.scatter(Xrr,(np.random.rand(Xrr.size)-0.5)*12.0 , c=cValGood, s=40, zorder=1) 
    plt.title('%s: LDA pC %.0f %%' % (titleStr, (ldaScores.mean()*100.0)))
    plt.axis('square')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))    
    plt.xlabel('DFA 1')
    plt.ylabel('DFA 2')

    
    # Transform the predictions in color codes
    maxQDA = yPredQDA.max()
    for ix in range(nxm) :
        cWeight = yPredQDA[ix,:]                               # Prob for all classes
        cWinner = ((cWeight == cWeight.max()).astype('float')) # Winner takes all 
        # XmcLDA[ix,:] = np.dot(cWeight, cClasses)/nClasses
        XmcQDA[ix,:] = np.dot(cWinner, cClasses)
        XmcQDA[ix,3] = cWeight.max()/maxQDA
    
    # Plot the surface of probability    
    plt.subplot(132)
    Zplot = XmcQDA.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
    plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
    if nClasses > 2:
        plt.scatter(Xrr[:,0], Xrr[:,1], c=cValGood, s=40, zorder=1)
    else:
        plt.scatter(Xrr,(np.random.rand(Xrr.size)-0.5)*12.0 , c=cValGood, s=40, zorder=1) 
    plt.title('%s: QDA pC %.0f %%' % (titleStr, (qdaScores.mean()*100.0)))
    plt.xlabel('DFA 1')
    plt.ylabel('DFA 2')
    plt.axis('square')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    plt.savefig('%s/%s.eps' % (figdir,titleStr))
    
    
    
    # Transform the predictions in color codes
    maxRF = yPredRF.max()
    for ix in range(nxm) :
        cWeight = yPredRF[ix,:]           # Prob for all classes
        cWinner = ((cWeight == cWeight.max()).astype('float')) # Winner takes all 
        # XmcLDA[ix,:] = np.dot(cWeight, cClasses)/nClasses  # Weighted colors does not work
        XmcRF[ix,:] = np.dot(cWinner, cClasses)
        XmcRF[ix,3] = cWeight.max()/maxRF
    
    # Plot the surface of probability    
    plt.subplot(133)
    Zplot = XmcRF.reshape(np.shape(x1)[0], np.shape(x1)[1],4)
    plt.imshow(Zplot, zorder=0, extent=[-6, 6, -6, 6], origin='lower', interpolation='none', aspect='auto')
    if nClasses > 2:    
        plt.scatter(Xrr[:,0], Xrr[:,1], c=cValGood, s=40, zorder=1)
    else:
        plt.scatter(Xrr,(np.random.rand(Xrr.size)-0.5)*12.0 , c=cValGood, s=40, zorder=1) 
    plt.title('%s: RF pC %.0f %%' % (titleStr, (rfScores.mean()*100.0)))
    plt.xlabel('DFA 1')
    plt.ylabel('DFA 2')
    plt.axis('square')
    plt.xlim((-6, 6))
    plt.ylim((-6, 6))
    
    plt.show()


    # Results
    ldaScore = ldaScores.mean()*100.0
    qdaScore = qdaScores.mean()*100.0
    rfScore = rfScores.mean()*100.0
    ldaScoreSE = ldaScores.std() * 100.0
    qdaScoreSE = qdaScores.std() * 100.0 
    rfScoreSE = rfScores.std() * 100.0 
    
    print ("Number of classes %d. Chance level %.2f %%") % (nClasses, 100.0/nClasses)
    print ("%s LDA: %.2f (+/- %0.2f) %%") % (titleStr, ldaScore, ldaScoreSE)
    print ("%s QDA: %.2f (+/- %0.2f) %%") % (titleStr, qdaScore, qdaScoreSE)
    print ("%s RF: %.2f (+/- %0.2f) %%") % (titleStr, rfScore, rfScoreSE)
    return ldaScore, ldaScoreSE, qdaScore, qdaScoreSE, rfScore, rfScoreSE, nClasses

