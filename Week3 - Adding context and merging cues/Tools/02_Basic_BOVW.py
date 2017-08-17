from BOVW_functions import *
from sklearn.metrics import roc_curve, auc

detector='Dense'
descriptor='SIFT'
num_samples=70000

k=64 # KNN parameter
C=1 # LinearSVM parameter

doPCA = False # Do PCA before K Means
pca_components = 40
classifier='LinearSVM' # Choose between KNN and LinearSVM

codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../MIT_split/train/')
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)
CB=getAndSaveCodebook_GMM(DSC_train, num_samples, k, codebook_filename, doPCA,pca_components)
#CB=cPickle.load(open(codebook_filename,'r'))

VW_train=getAndSaveBoVWRepresentation_GMM(DSC_train,k,CB,visual_words_filename_train)
#VW_train=cPickle.load(open(visual_words_filename_train,'r'))

filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../MIT_split/test/')
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)
VW_test=getAndSaveBoVWRepresentation_GMM(DSC_test,k,CB,visual_words_filename_test)

if classifier == 'KNN':
	ac_BOVW_L = trainAndTestKNeighborsClassifier(VW_train,VW_test,GT_ids_train,GT_ids_test,k)
elif classifier == 'LinearSVM':
	ac_BOVW_L,cm,fpr,tpr,roc_auc = trainAndTestLinearSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,C)

names = unique_elements(GT_labels_test)
savename = 'cm.png'
plot_confusion_matrix(cm,names,savename)
print 'Accuracy BOVW: '+str(ac_BOVW_L)

plt.figure()

plt.figure()
plt.plot(fpr["macro"], tpr["macro"],
     label='macro-average ROC curve (area = {0:0.2f})'
           ''.format(roc_auc["macro"]),
     linewidth=2)

for i in range(8):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()