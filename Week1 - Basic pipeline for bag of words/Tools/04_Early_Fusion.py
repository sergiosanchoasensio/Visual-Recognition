from BOVW_functions import *

##########################
#PAR�METROS DE EJECUCION
##########################

# M�todo usado para detectar los puntos de inter�s
detector='Dense'
# M�todo usado para obtener la descripci�n de los puntos de inter�s
descriptor='SIFT'
# N�mero de puntos de inter�s que se utilizan para obtener el vocabulario visual
num_samples=50000
# N�mero de palabras en el vocabulario visual
k=32
# Factor de regularizaci�n C para entrenar el clasificador SVM
C=1
# Directorio raiz donde se encuentran todas las im�genes de aprendizaje
dataset_folder_train='../../Databases/MIT_split/train/'
# Directorio raiz donde se encuentran todas las im�genes de test
dataset_folder_test='../../Databases/MIT_split/test/'

##############################################


# Preparaci�n de los nombres de los ficheros necesarios para guardar el vocabulario y las palabras visuales de las im�genes de aprendizaje y test
# Se generan tres vocabularios y, por lo tanto tres conjuntos de palabras visuales utilizando tres configuraciones diferentes de descriptor:
# 1. S�lo SIFT, 2. Color, 3. Concatenaci�n de SIFT + color

codebook_filename_SIFT='CB_S_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
codebook_filename_COLOR='CB_C_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
codebook_filename_EARLY='CB_E_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

visual_words_filename_train_SIFT='VW_train_SIFT_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train_COLOR='VW_train_COLOR_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train_EARLY='VW_train_EARLY_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

visual_words_filename_test_SIFT='VW_test_SIFT_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test_COLOR='VW_test_COLOR_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test_EARLY='VW_test_EARLY_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

# C�lculo de puntos de inter�s para todas las im�genes del conjunto de aprendizaje. La descripci�n se obtiene tanto con SIFT como con el descriptor de color
filenames_train,GT_ids_train,GT_labels_train = prepareFiles(dataset_folder_train)
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)
CDSC_train = getLocalColorDescriptors(filenames_train,KPTS_train)

# Early fusion de las dos descripciones para cada punto de inter�s: SIFT + color. Simplemente se concatenan los dos descriptores
FDSC_train=[]
for i in range(len(DSC_train)):
	FDSC_train.append(np.hstack((DSC_train[i],CDSC_train[i])))

# Construcci�n de los 3 vocabularios visuales: SIFT, Color y Early fusion. Los vocabularios quedan guardados en disco.
# Comentar estas l�neas si los vocabularios ya est�n creados y guardados en disco de una ejecuci�n anterior
CB_SIFT=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename_SIFT)
CB_COLOR=getAndSaveCodebook(CDSC_train, num_samples, k, codebook_filename_COLOR)
CB_EARLY=getAndSaveCodebook(FDSC_train, num_samples, k, codebook_filename_EARLY)

# Carga de los vocabularios visuales previamente creados y guardados en disco en una ejecuci�n anterior.
# Comentar estas l�neas si se quiere re-calcular los vocabularios o si los vocabularios todav�a no se han creado
#CB_SIFT=cPickle.load(open(codebook_filename_SIFT,'r'))
#CB_COLOR=cPickle.load(open(codebook_filename_COLOR,'r'))
#CB_EARLY=cPickle.load(open(codebook_filename_EARLY,'r'))

# Obtiene la descripci�n BoW de las im�genes del conjunto de aprendizaje para las tres descripciones: SIFT, Color y Early fusion
VW_SIFT_train=getAndSaveBoVWRepresentation(DSC_train,k,CB_SIFT,visual_words_filename_train_SIFT)
VW_COLOR_train=getAndSaveBoVWRepresentation(CDSC_train,k,CB_COLOR,visual_words_filename_train_COLOR)
VW_FUSION_train=getAndSaveBoVWRepresentation(FDSC_train,k,CB_EARLY,visual_words_filename_train_EARLY)

# Carga de las 3 descripciones BoW del conjunto de aprendizaje previamente creadas y guardadas en disco en una ejecuci�n anterior.
# Comentar estas l�neas si se quiere re-calcular la representaci�n o si la representaci�n todav�a no se ha creado
#VW_SIFT_train=cPickle.load(open(visual_words_filename_train_SIFT,'r'))
#VW_COLOR_train=cPickle.load(open(visual_words_filename_train_COLOR,'r'))
#VW_FUSION_train=cPickle.load(open(visual_words_filename_train_EARLY,'r'))

# C�lculo de puntos de inter�s para todas las im�genes del conjunto de test. Obtiene las dos descripciones (SIFT y color) y tambi�n las concatena
# para obtener la descripci�n Early Fusion
filenames_test,GT_ids_test,GT_labels_test = prepareFiles(dataset_folder_test)
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)
CDSC_test = getLocalColorDescriptors(filenames_test,KPTS_test)
FDSC_test=[]
for i in range(len(DSC_test)):
	FDSC_test.append(np.hstack((DSC_test[i],CDSC_test[i])))

# Obtiene las 3 descripciones BoW (SIFT, color y early fusion) de las im�genes del conjunto de test
VW_SIFT_test=getAndSaveBoVWRepresentation(DSC_test,k,CB_SIFT,visual_words_filename_test_SIFT)
VW_COLOR_test=getAndSaveBoVWRepresentation(CDSC_test,k,CB_COLOR,visual_words_filename_test_COLOR)
VW_FUSION_test=getAndSaveBoVWRepresentation(FDSC_test,k,CB_EARLY,visual_words_filename_test_EARLY)

# Entrena un clasificador SVM con las im�genes del conjunto de aprendizaje y lo eval�a utilizando las im�genes del conjunto de test
# para las 3 descripciones (SIFT, color y early fusion)
# Devuelve la accuracy como medida del rendimiento del clasificador
ac_BOVW_SIFT = trainAndTestLinearSVM(VW_SIFT_train,VW_SIFT_test,GT_ids_train,GT_ids_test,C)
ac_BOVW_COLOR = trainAndTestLinearSVM(VW_COLOR_train,VW_COLOR_test,GT_ids_train,GT_ids_test,C)
ac_BOVW_EF = trainAndTestLinearSVM(VW_FUSION_train,VW_FUSION_test,GT_ids_train,GT_ids_test,C)


print 'Accuracy BOVW with LinearSVM SIFT: '+str(ac_BOVW_SIFT)
print 'Accuracy BOVW with LinearSVM Color: '+str(ac_BOVW_COLOR)
print 'Accuracy BOVW with LinearSVM Early Fusion: '+str(ac_BOVW_EF)