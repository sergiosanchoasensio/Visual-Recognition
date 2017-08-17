from BOVW_functions import *
from sklearn.metrics import roc_curve, auc

detector='SIFT'
descriptor='LBP'
num_samples=50000
k=32 # KNN parameter
folds = 5
start=0.01
end=10
numparams=30
classifier='LinearSVM' # Choose between KNN and linearSVM
text_file = open("Output_lbp_lin.txt", "a")

codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../MIT_split/train/')
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)
CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)
#CB=cPickle.load(open(codebook_filename,'r'))

VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)
#VW_train=cPickle.load(open(visual_words_filename_train,'r'))

filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../MIT_split/test/')
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)
VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)

#ac_BOVW_L = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

if classifier == 'KNN':
    ac_BOVW_L = trainAndTestKNeighborsClassifier_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,k)
elif classifier == 'LinearSVM':
    ac_BOVW_L,cm,fpr,tpr = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

names = unique_elements(GT_labels_test)
savename = str(folds) + 'lbp_cm_lin.png'
plot_confusion_matrix(cm,names,savename)
s = 'For lbp_lin The accuracy is ' + str(ac_BOVW_L)
text_file.write(s)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr,
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc))
#for i in range(8):w
#    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
#                                   ''.format(i, roc_auc[i]))
save = str(folds) +'lbp_lin.png'
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig(save)
plt.close()
