import cv2
import glob
import numpy as np
import Config as cfg
import matplotlib.pyplot as plt
import cPickle
import time
import random
from PIL import Image
import scipy.cluster.vq as vq
from sklearn import cross_validation
from sklearn import svm
from sklearn import neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.util import view_as_windows
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from joblib import Parallel, delayed
from joblib import load, dump
from scipy import interp
import ColorNaming as cnam


def prepareFiles(rootpath):
    current_GT_id=0
    filenames=[]
    GT_ids=[]
    GT_labels=[]
    classpath = sorted(glob.glob(rootpath+'*'))
    for i in classpath:
        filespath = sorted(glob.glob(i+'/*.jpg'))
        for j in filespath:
            filenames.append(j)
            GT_ids.append(current_GT_id)
            GT_labels.append(i.split('/')[-1])
        current_GT_id+=1
    return(filenames,GT_ids,GT_labels)

def getKeypointsDescriptors(filenames,detector_type,descriptor_type):
    detector=cv2.FeatureDetector_create(detector_type)
    if not(descriptor_type == 'color'):
        if not (descriptor_type == 'HOG' or descriptor_type == 'LBP'):
            descriptor = cv2.DescriptorExtractor_create(descriptor_type)
        K = []
        D = []
        print 'Extracting Local Descriptors'
        init=time.time()
        for filename in filenames:
            ima=cv2.imread(filename)
            gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
            if descriptor_type == 'HOG':
                des = extractHOGfeatures(gray, detector)
                D.append(des)
            elif descriptor_type == 'LBP':
                des = extractLBPfeatures(gray, detector)
                D.append(des)
            else:
                kpts=detector.detect(gray)
                kpts,des=descriptor.compute(gray,kpts)
                K.append(kpts)
                D.append(des)

        end=time.time()

        print 'Done in '+str(end-init)+' secs.'
    if(descriptor_type == 'color'):
        K = []
        print 'Extracting Local Descriptors'
        init=time.time()
        for filename in filenames:
            ima=cv2.imread(filename)
            gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
            kpts=detector.detect(gray)
            K.append(kpts)
        D = getLocalColorDescriptors(filenames, K, 0)
    return(K,D)




def extractLBPfeatures(img, detector):
    lbp = local_binary_pattern(img, cfg.lbp_n_points, cfg.lbp_radius, cfg.lbp_METHOD)
    lbp_windows = view_as_windows(lbp, window_shape=cfg.lbp_win_shape, step=cfg.lbp_win_step)
    features = []
    count = 0
    ws = cfg.lbp_win_shape[1]
    kpts = detector.detect(img)
    loc = [(int(x.pt[0]),int(x.pt[1])) for x in kpts]
    for poi in loc:
        window = Image.fromarray(img).crop((poi[0]-ws,poi[1]-ws,poi[0]+ws,poi[1]+ws))
        window = np.array(window)
 #   for windows_list in lbp_windows:
#       for window in windows_list:
        lbp_hist, bin_edges = np.histogram(window, bins=cfg.lbp_n_bins)
        lbp_hist_norm = sum(abs(lbp_hist))
        lbp_hist_l1sqrtnorm = np.sqrt(lbp_hist/float(lbp_hist_norm))
        features.append(lbp_hist_l1sqrtnorm)
        count += 1
    #features_flatten = [item for sublist in features for item in sublist]
    features = np.asarray(features)
    return features

def extractHOGfeatures(img, detector):
    #winSize = (64,64)
    #blockSize = (16,16)
    #blockStride = (8,8)
    #cellSize = (8,8)
    #nbins = 9
    hog = cv2.HOGDescriptor()#winSize,blockSize,blockStride,cellSize,nbins)
    kpts = detector.detect(img)
    loc = [(int(x.pt[0]),int(x.pt[1])) for x in kpts]
    loc = tuple(loc)
    fd = hog.compute(img,hog.blockStride,hog.cellSize,loc)
    #fd = hog(img,
    #         orientations=cfg.hog_orientations,
    #         pixels_per_cell=cfg.hog_pixels_per_cell,
    #         cells_per_block=cfg.hog_cells_per_block,
    #         visualise=False,
    return fd

def getLocalColorDescriptors(filenames, keypoints, colormode):
    CD=[]
    area=4
    n_bins=16
    print 'Extracting Local Color Descriptors'
    init=time.time()
    cont=0
    for filename in filenames:
        if colormode == 0:
            ima=cv2.imread(filename)
            hls=cv2.cvtColor(ima,cv2.COLOR_BGR2HLS)
            cdesc = cnam.getColorNamingDescriptor(hls)
        else:
            kpts=keypoints[cont]
            cdesc=np.zeros((len(kpts),n_bins),dtype=np.float32)
            ima=cv2.imread(filename)
            hls=cv2.cvtColor(ima,cv2.COLOR_BGR2HLS)
            cnam.getColorNamingDescriptor(hls)
            hue=hls[:,:,0]
            w,h=hue.shape
            cont2=0
            for k in kpts:
                patch=hue[max(0,k.pt[0] - area*k.size):min(w,k.pt[0] + area*k.size),max(0,k.pt[1] - area*k.size):min(h,k.pt[1] + area*k.size)]
                hist,bin_edges=np.histogram(patch,bins=n_bins,range=(0,180))
                cdesc[cont2,:]=hist
                cont2+=1
            cont+=1
        CD.append(cdesc)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return(CD)

def computePCA(data,experimentType, pca_ncomponents):
    #data is the input data to be reduced
    #experimentType is the type of experiment: 	1 = plots the PCA variance depending on the number of components.
    # 											0 = performs a PCA with n_components

    init=time.time()
    if experimentType == 1:
        print 'Looking for the PCA variance depending on the number of items'
        my_model = PCA()
        my_model.fit_transform(data)
        end=time.time()
        plt.figure()
        components = [index for index in range(0, len(my_model.explained_variance_ratio_.cumsum()))]
        plt.plot(components,my_model.explained_variance_ratio_.cumsum())
        plt.xlabel('Components')
        plt.ylabel('Acc. Variance')
        plt.title('PCA variance depending on the number of components')
        plt.show()
    else:
        print 'Computing PCA using '+str(pca_ncomponents)+' components.'
        my_model = PCA(pca_ncomponents)
        my_model.fit_transform(data)
        end=time.time()
    print 'END PCA: Done in '+str(end-init)+' secs.'

    return my_model.components_ # Return the eigenvalues

def getAndSaveCodebook(descriptors,num_samples,k,filename,doPCA, pca_ncomponents):
    size_descriptors=descriptors[0].shape[1]
    A=np.zeros((num_samples,size_descriptors),dtype=np.float32)
    for i in range(num_samples):
        A[i,:]=random.choice(random.choice(descriptors))

    #Perform a dimensionality reduction before K Means
    if doPCA == True:
        A = computePCA(A,0, pca_ncomponents)

    print 'Computing kmeans on '+str(num_samples)+' samples with '+str(k)+' centroids'
    init=time.time()
    A = vq.whiten(A)
    codebook,v=vq.kmeans(A,k,1)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    cPickle.dump(codebook, open(filename, "wb"))
    return codebook

def getAndSaveBoVWRepresentation(descriptors,k,codebook,filename):
    print 'Extracting visual word representations'
    init=time.time()
    visual_words=np.zeros((len(descriptors),k),dtype=np.float32)
    for i in xrange(len(descriptors)):
        words,distance=vq.vq(descriptors[i],codebook)
        visual_words[i,:]=np.bincount(words,minlength=k)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    cPickle.dump(visual_words, open(filename, "wb"))
    return visual_words

def getAndSaveBoVW_SPMRepresentation(descriptors,keypoints,k,codebook,filename,files):
    print 'Extracting visual word representations with SPM'
    init=time.time()
    visual_words=np.zeros((len(descriptors),k*21),dtype=np.float32)
    for i in xrange(len(descriptors)):
        ima=cv2.imread(files[i])
        w,h,_=ima.shape
        words,distance=vq.vq(descriptors[i],codebook)
        idx_bin1=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(0*w/2)) & (x[0]<(1*w/2)) & (x[1]>=(0*h/2)) & (x[1]<(1*h/2)) )]
        idx_bin2=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(1*w/2)) & (x[0]<(2*w/2)) & (x[1]>=(0*h/2)) & (x[1]<(1*h/2)) )]
        idx_bin3=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(0*w/2)) & (x[0]<(1*w/2)) & (x[1]>=(1*h/2)) & (x[1]<(2*h/2)) )]
        idx_bin4=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(1*w/2)) & (x[0]<(2*w/2)) & (x[1]>=(1*h/2)) & (x[1]<(2*h/2)) )]

        idx_bin5=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(0*w/4)) & (x[0]<(1*w/4)) & (x[1]>=(0*h/4)) & (x[1]<(1*h/4)) )]
        idx_bin6=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(1*w/4)) & (x[0]<(2*w/4)) & (x[1]>=(0*h/4)) & (x[1]<(1*h/4)) )]
        idx_bin7=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(2*w/4)) & (x[0]<(3*w/4)) & (x[1]>=(0*h/4)) & (x[1]<(1*h/4)) )]
        idx_bin8=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(3*w/4)) & (x[0]<(4*w/4)) & (x[1]>=(0*h/4)) & (x[1]<(1*h/4)) )]

        idx_bin9=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(0*w/4)) & (x[0]<(1*w/4)) & (x[1]>=(1*h/4)) & (x[1]<(2*h/4)) )]
        idx_bin10=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(1*w/4)) & (x[0]<(2*w/4)) & (x[1]>=(1*h/4)) & (x[1]<(2*h/4)) )]
        idx_bin11=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(2*w/4)) & (x[0]<(3*w/4)) & (x[1]>=(1*h/4)) & (x[1]<(2*h/4)) )]
        idx_bin12=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(3*w/4)) & (x[0]<(4*w/4)) & (x[1]>=(1*h/4)) & (x[1]<(2*h/4)) )]

        idx_bin13=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(0*w/4)) & (x[0]<(1*w/4)) & (x[1]>=(2*h/4)) & (x[1]<(3*h/4)) )]
        idx_bin14=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(1*w/4)) & (x[0]<(2*w/4)) & (x[1]>=(2*h/4)) & (x[1]<(3*h/4)) )]
        idx_bin15=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(2*w/4)) & (x[0]<(3*w/4)) & (x[1]>=(2*h/4)) & (x[1]<(3*h/4)) )]
        idx_bin16=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(3*w/4)) & (x[0]<(4*w/4)) & (x[1]>=(2*h/4)) & (x[1]<(3*h/4)) )]

        idx_bin17=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(0*w/4)) & (x[0]<(1*w/4)) & (x[1]>=(3*h/4)) & (x[1]<(4*h/4)) )]
        idx_bin18=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(1*w/4)) & (x[0]<(2*w/4)) & (x[1]>=(3*h/4)) & (x[1]<(4*h/4)) )]
        idx_bin19=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(2*w/4)) & (x[0]<(3*w/4)) & (x[1]>=(3*h/4)) & (x[1]<(4*h/4)) )]
        idx_bin20=[j for j,x in enumerate([keypoints[i][m].pt for m in range(len(keypoints[i]))]) if ((x[0]>=(3*w/4)) & (x[0]<(4*w/4)) & (x[1]>=(3*h/4)) & (x[1]<(4*h/4)) )]


        visual_words[i,:]=np.hstack((np.bincount(words,minlength=k), np.bincount(words[idx_bin1],minlength=k), np.bincount(words[idx_bin2],minlength=k), np.bincount(words[idx_bin3],minlength=k), np.bincount(words[idx_bin4],minlength=k), np.bincount(words[idx_bin5],minlength=k), np.bincount(words[idx_bin6],minlength=k), np.bincount(words[idx_bin7],minlength=k), np.bincount(words[idx_bin8],minlength=k), np.bincount(words[idx_bin9],minlength=k), np.bincount(words[idx_bin10],minlength=k), np.bincount(words[idx_bin11],minlength=k), np.bincount(words[idx_bin12],minlength=k) , np.bincount(words[idx_bin13],minlength=k), np.bincount(words[idx_bin14],minlength=k), np.bincount(words[idx_bin15],minlength=k), np.bincount(words[idx_bin16],minlength=k), np.bincount(words[idx_bin17],minlength=k), np.bincount(words[idx_bin18],minlength=k), np.bincount(words[idx_bin19],minlength=k) , np.bincount(words[idx_bin20],minlength=k)  ))

    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    cPickle.dump(visual_words, open(filename, "wb"))
    return visual_words

def trainAndTestKNeighborsClassifier(train,test,GT_train,GT_test,k):
    #http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
    print 'Training and Testing a KNN'
    init 		= time.time()
    stdSlr 		= StandardScaler().fit(train)
    train 		= stdSlr.transform(train)
    neigh 		= neighbors.KNeighborsClassifier(n_neighbors=k).fit(train, GT_train)
    accuracy 	= 100*neigh.score(stdSlr.transform(test), GT_test)
    assert isinstance(test, object)
    pred 		= neigh.predict(test)
    cm 			= confusion_matrix(GT_test, pred)
    fpr, tpr, thresholds = roc_curve(np.asarray(GT_test).ravel(), pred.ravel(), pos_label=2)
    end 		= time.time()
    print 'Done in '+str(end-init)+' secs.'
    return accuracy, cm, fpr, tpr

def trainAndTestKNeighborsClassifier_withfolds(train,test,GT_train,GT_test,folds,k):
    print 'Training and Testing a KNN (with folds)'
    init 				= time.time()
    stdSlr 				= StandardScaler().fit(train)
    train 				= stdSlr.transform(train)
    kernelMatrix 		= histogramIntersection(train, train)
    tuned_parameters 	= [{'n_neighbors': [k]}]
    neigh 				= GridSearchCV(neighbors.KNeighborsClassifier(),tuned_parameters,cv	= folds,scoring='accuracy')
    neigh.fit(kernelMatrix, GT_train)
    print(neigh.best_params_)
    predictMatrix 		= histogramIntersection(stdSlr.transform(test), train)
    NNpredictions 		= neigh.predict(predictMatrix)
    correct 			= sum(1.0 * (NNpredictions == GT_test))
    accuracy 			= correct / len(GT_test)
    assert isinstance(test, object)
    cm 					= confusion_matrix(GT_test, NNpredictions)
    fpr, tpr, thresholds= roc_curve(np.asarray(GT_test).ravel(), NNpredictions.ravel(), pos_label=2)
    end					= time.time()
    print 'Done in '+str(end-init)+' secs.'
    return accuracy, cm, fpr, tpr

def trainAndTestLinearSVM(train,test,GT_train,GT_test,c):
    print 'Training and Testing a linear SVM'
    init=time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    clf = svm.SVC(kernel='linear', C=c, decision_function_shape='ovr').fit(train, GT_train)
    pred = clf.predict(test)
    cm = confusion_matrix(GT_test, pred)
    sc = clf.score(stdSlr.transform(test), GT_test)
    accuracy = 100*sc
    end=time.time()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = clf.decision_function(stdSlr.transform(test))

    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(np.asarray(GT_test), y_score[:,i],pos_label = i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(8)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(8):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 8

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return accuracy,cm,fpr,tpr,roc_auc

def trainAndTestLinearSVM_withfolds(train,test,GT_train,GT_test,folds,start,end,numparams):
    print 'Training and Testing a Linear SVM'
    init=time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = histogramIntersection(train, train)
    tuned_parameters = [{'kernel': ['linear'], 'C':np.linspace(start,end,num=numparams)}]
    clf = GridSearchCV(svm.SVC(kernel='linear',decision_function_shape='ovr'), tuned_parameters, cv=folds,scoring='accuracy',n_jobs = 6)
    clf.fit(kernelMatrix, GT_train)
    print(clf.best_params_)
    predictMatrix = histogramIntersection(stdSlr.transform(test), train)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    cm = confusion_matrix(GT_test, SVMpredictions)
    end=time.time()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = clf.decision_function(predictMatrix)

    for i in range(8):
        fpr[i], tpr[i], _ = roc_curve(np.asarray(GT_test), y_score[:,i],pos_label = i)
        roc_auc[i] = auc(fpr[i], tpr[i])

        # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(8)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(8):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 8

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    print 'Done in '+str(end-init)+' secs.'
    return accuracy,cm,fpr,tpr,roc_auc

def histogramIntersection(M, N):
    m = M.shape[0]
    n = N.shape[0]
    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = np.sum(np.minimum(M[i], N[j]))
            result[i][j] = temp
    return result

def SPMKernel(M, N,k):
    m = M.shape[0]
    n = N.shape[0]

    result = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            temp = ((.25*np.sum(np.minimum(M[i,0:k], N[j,0:k]))) + (.25*np.sum(np.minimum(M[i,k:k*5], N[j,k:k*5]))) + (.5*np.sum(np.minimum(M[i,k*5:k*21], N[j,k*5:k*21]))))
            result[i][j] = temp
    return result

def trainAndTestHISVM(train,test,GT_train,GT_test,c):
    print 'Training and Testing a HI SVM'
    init=time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = histogramIntersection(train, train)
    clf = svm.SVC(kernel='precomputed',C=c)
    clf.fit(kernelMatrix, GT_train)
    predictMatrix = histogramIntersection(stdSlr.transform(test), train)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return accuracy

def trainAndTestHISVM_withfolds(train,test,GT_train,GT_test,folds):
    print 'Training and Testing a HI SVM'
    init=time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = histogramIntersection(train, train)
    tuned_parameters = [{'kernel': ['precomputed'], 'C':np.linspace(0.0001,0.2,num=10)}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=folds,scoring='accuracy')
    clf.fit(kernelMatrix, GT_train)
    print(clf.best_params_)
    predictMatrix = histogramIntersection(stdSlr.transform(test), train)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return accuracy

def trainAndTestSPMSVM(train,test,GT_train,GT_test,c,k):
    print 'Training and Testing a SPMKernel SVM'
    init=time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = SPMKernel(train, train,k)
    clf = svm.SVC(kernel='precomputed',C=c)
    clf.fit(kernelMatrix, GT_train)
    predictMatrix =SPMKernel(stdSlr.transform(test), train,k)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return accuracy

def trainAndTestSPMSVM_withfolds(train,test,GT_train,GT_test,k,folds):
    print 'Training and Testing a SPMKernel SVM'
    init=time.time()
    stdSlr = StandardScaler().fit(train)
    train = stdSlr.transform(train)
    kernelMatrix = SPMKernel(train, train,k)
    tuned_parameters = [{'kernel': ['precomputed'], 'C':np.linspace(0.0001,0.2,num=10)}]
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=folds,scoring='accuracy')
    clf.fit(kernelMatrix, GT_train)
    print(clf.best_params_)
    predictMatrix =SPMKernel(stdSlr.transform(test), train,k)
    SVMpredictions = clf.predict(predictMatrix)
    correct = sum(1.0 * (SVMpredictions == GT_test))
    accuracy = correct / len(GT_test)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return accuracy
 
def plot_confusion_matrix(cm, names, savename, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
#    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(savename)
    plt.close()

def unique_elements(seq): 
   # order preserving
   checked = []
   for e in seq:
       if e[5:] not in checked:
           checked.append(e[5:])
   return checked
