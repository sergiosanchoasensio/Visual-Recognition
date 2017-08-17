from BOVW_functions import *

# Funcion para normalizar el array (norma l2).
def arrayNormalization(input):
    return input / np.linalg.norm(input, axis=-1)[:, np.newaxis]

if __name__ == '__main__':

    # Parametros de experimento
    detector='FAST'
    descriptor=['SIFT','color']
    num_samples         = 60000
    k                   = 80
    doPCA               = False
    pca_components      = 16
    weighting           = 0.35 # Valor entre 0 y 1. Representa la importancia del primer descriptor respecto el segundo (solo se aplica si len(descriptors) == 2). Por defecto usar 0.5.
    colormode           = 0 # Tipo de color para el descriptor.
    spm                 = True # Realizar Spatial pyramid
    folds = 5
    start=0.001
    end=10
    numparams=30

    # Extraer hist. del vocabulario
    dlist = []
    for current_descriptor in descriptor:
        print 'Descriptor: '+current_descriptor
        codebook_filename='CB_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_filename_train='VW_train_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_filename_test='VW_test_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

        if spm:
            visual_words_SPM_filename_train='VWSPM_train_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
            visual_words_SPM_filename_test='VWSPM_test_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

        filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../MIT_split/train/')
        KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,current_descriptor)

        if current_descriptor == 'color':
            DSC_train = getLocalColorDescriptors(filenames_train,KPTS_train,colormode)

        CB=getAndSaveCodebook_GMM(DSC_train, num_samples, k, codebook_filename, doPCA, pca_components)

        VW_train=getAndSaveBoVWRepresentation_GMM(DSC_train,k,CB,visual_words_filename_train)

        if spm:
            VWSPM_train=getAndSaveBoVW_SPMRepresentation_GMM(DSC_train,KPTS_train,k,CB,visual_words_SPM_filename_train,filenames_train)

        if len(descriptor)>1:
            dlist.append(VW_train)
            if spm:
                dlist.append(VWSPM_train)

        filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../MIT_split/test/')
        KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,current_descriptor)

        if current_descriptor == 'color':
            DSC_test = getLocalColorDescriptors(filenames_test,KPTS_test,colormode)

        VW_test=getAndSaveBoVWRepresentation_GMM(DSC_test,k,CB,visual_words_filename_test)
        if spm:
            VWSPM_test=getAndSaveBoVW_SPMRepresentation_GMM(DSC_test,KPTS_test,k,CB,visual_words_SPM_filename_test,filenames_test)

        if len(descriptor)>1:
            dlist.append(VW_test)
            if spm:
                dlist.append(VWSPM_test)

    # Concatenar descriptores
    if len(descriptor)>1:
        if spm:
            VW_train        = None
            VWSPM_train     = None
            VW_test         = None
            VWSPM_test      = None

            # Asignar peso a cada descriptor
            offset          = len(dlist)/len(descriptor)
            dlist[0]        = dlist[0] * weighting
            dlist[0+offset] = dlist[0+offset] * (1-weighting)

            dlist[1]        = dlist[1] * weighting
            dlist[1+offset] = dlist[1+offset] * (1-weighting)

            # Normalizar valores
            dlist[0]        = arrayNormalization(dlist[0])
            dlist[1]        = arrayNormalization(dlist[1])
            dlist[2]        = arrayNormalization(dlist[2])
            dlist[3]        = arrayNormalization(dlist[3])

            # Unir histogramas de los vocabularios
            VW_train        = np.hstack((dlist[0],dlist[0+offset]))
            VWSPM_train     = np.hstack((dlist[1],dlist[1+offset]))
            VW_test         = np.hstack((dlist[2],dlist[2+offset]))
            VWSPM_test      = np.hstack((dlist[3],dlist[3+offset]))
        else:
            VW_train        = None
            VW_test         = None

            # Asignar peso a cada descriptor
            offset          = len(dlist)/len(descriptor)
            dlist[0]        = dlist[0] * weighting
            dlist[0+offset] = dlist[0+offset] * (1-weighting)

            # Normalizar valores
            dlist[0]        = arrayNormalization(dlist[0])
            dlist[1]        = arrayNormalization(dlist[1])

            # Unir histogramas de los vocabularios
            VW_train        = np.hstack((dlist[0],dlist[0+offset]))
            VW_test         = np.hstack((dlist[1],dlist[1+offset]))

    ac_BOVW_L = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)
    print 'Accuracy BOVW with LinearSVM: '+str(ac_BOVW_L)
    ac_BOVW_HI = trainAndTestHISVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds)
    print 'Accuracy BOVW with HISVM: '+str(ac_BOVW_HI)
    # Clasificar
    if spm:
        ac_BOVW_SPM_L = trainAndTestLinearSVM_withfolds(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)
        print 'Accuracy BOVW with SPM with LinearSVM:'+str(ac_BOVW_SPM_L)
        ac_BOVW_SPM_HI = trainAndTestHISVM_withfolds(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,folds)
        print 'Accuracy BOVW with SPM with HISVM: '+str(ac_BOVW_SPM_HI)
        ac_BOVW_SPM_SPMK = trainAndTestSPMSVM_withfolds(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,k,folds)
        print 'Accuracy BOVW with SPM with SPMKernelSVM: '+str(ac_BOVW_SPM_SPMK)