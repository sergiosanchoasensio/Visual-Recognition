from BOVW_functions import *
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':

    detector=['FAST','Dense']
    descriptor=['SIFT']
    num_samples= 60000

    k= [16,32,64,80,128]

    folds = 5
    start=0.001
    end=10
    numparams=30

    doPCA = False # Do PCA before K Means

    # Name of the accuracy file
    for i in range(len(detector)):
        for j in range(len(descriptor)):
            for n in range(len(k)):
                codebook_filename='CB_'+detector[i]+'_'+descriptor[j]+'_'+str(num_samples)+'samples_'+str(k[n])+'centroids.dat'
                visual_words_filename_train='VW_train_'+detector[i]+'_'+descriptor[j]+'_'+str(num_samples)+'samples_'+str(k[n])+'centroids.dat'
                visual_words_filename_test='VW_test_'+detector[i]+'_'+descriptor[j]+'_'+str(num_samples)+'samples_'+str(k[n])+'centroids.dat'

                filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../MIT_split/train/')
                KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector[i],descriptor[j])
                pca_components = int(round(0.9 *  DSC_train[0].shape[1]))

                # Perform PCA
                if doPCA == True:
                    DSC_train, PCAModel  = computePCA(DSC_train,0,pca_components,False)

                CB=getAndSaveCodebook_GMM(DSC_train, num_samples, k[n], codebook_filename, doPCA, 0)

                #CB=cPickle.load(open(codebook_filename,'r'))
                VW_train=getAndSaveBoVWRepresentation_GMM(DSC_train,k[n],CB,visual_words_filename_train)
                #VW_train=cPickle.load(open(visual_words_filename_train,'r'))


                filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../MIT_split/test/')
                KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector[i],descriptor[j])
                # Perform PCA

                if doPCA == True:
                    DSC_test, _  = computePCA(DSC_test,2,pca_components,PCAModel)


                VW_test=getAndSaveBoVWRepresentation_GMM(DSC_test,k[n],CB,visual_words_filename_test)

                ac_BOVW_L,cm,fpr,tpr,roc_auc = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

                names = unique_elements(GT_labels_test)
                # Name of the confusion matrix file
                savename = detector[i] + '+' + descriptor[j] + '_' + str(k[n]) + '_CN.png'
                #Name of the ROC Curve file
                save = detector[i] + '+' + descriptor[j] + '_' + str(k[n]) + '_ROC.png'
                plot_confusion_matrix(cm,names,savename)
                print  'For ' + detector[i] +  '+' + descriptor[j] + ' and ' + str(k[n]) +  ' components, the accuracy is ' + str(ac_BOVW_L) + '\n'

                plt.figure()
                plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     linewidth=2)

                for z in range(8):
                    plt.plot(fpr[z], tpr[z], label='ROC curve of class {0} (area = {1:0.2f})'
                                                   ''.format(i, roc_auc[z]))

                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc="lower right")
                plt.savefig(save)
                plt.close()
                print 'End of iteration.'