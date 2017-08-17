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
# Se generan dos vocabularios y, por lo tanto dos conjuntos de palabras visuales utilizando dos configuraciones diferentes de descriptor:
# 1. S�lo SIFT, 2. Color
codebook_filename_SIFT='CB_S_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
codebook_filename_COLOR='CB_C_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

visual_words_filename_train_SIFT='VW_train_SIFT_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train_COLOR='VW_train_COLOR_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

visual_words_filename_test_SIFT='VW_test_SIFT_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test_COLOR='VW_test_COLOR_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'


# C�lculo de puntos de inter�s para todas las im�genes del conjunto de aprendizaje. La descripci�n se obtiene tanto con SIFT como con el descriptor de color
filenames_train,GT_ids_train,GT_labels_train = prepareFiles(dataset_folder_train)
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)
CDSC_train = getLocalColorDescriptors(filenames_train,KPTS_train)

# Construcci�n de los 2 vocabularios visuales: SIFT, Color. Los vocabularios quedan guardados en disco.
# Comentar estas l�neas si los vocabularios ya est�n creados y guardados en disco de una ejecuci�n anterior
CB_SIFT=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename_SIFT)
CB_COLOR=getAndSaveCodebook(CDSC_train, num_samples, k, codebook_filename_COLOR)

# Carga de los vocabularios visuales previamente creados y guardados en disco en una ejecuci�n anterior.
# Comentar estas l�neas si se quiere re-calcular los vocabularios o si los vocabularios todav�a no se han creado
#CB_SIFT=cPickle.load(open(codebook_filename_SIFT,'r'))
#CB_COLOR=cPickle.load(open(codebook_filename_COLOR,'r'))

# Obtiene la descripci�n BoW de las im�genes del conjunto de aprendizaje para las dos descripciones: SIFT, Color
VW_SIFT_train=getAndSaveBoVWRepresentation(DSC_train,k,CB_SIFT,visual_words_filename_train_SIFT)
VW_COLOR_train=getAndSaveBoVWRepresentation(CDSC_train,k,CB_COLOR,visual_words_filename_train_COLOR)

# Carga de las 2 descripciones BoW del conjunto de aprendizaje previamente creadas y guardadas en disco en una ejecuci�n anterior.
# Comentar estas l�neas si se quiere re-calcular la representaci�n o si la representaci�n todav�a no se ha creado
#VW_SIFT_train=cPickle.load(open(visual_words_filename_train_SIFT,'r'))
#VW_COLOR_train=cPickle.load(open(visual_words_filename_train_COLOR,'r'))

# C�lculo de puntos de inter�s para todas las im�genes del conjunto de test. Obtiene las dos descripciones (SIFT y color)
filenames_test,GT_ids_test,GT_labels_test = prepareFiles(dataset_folder_test)
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)
CDSC_test = getLocalColorDescriptors(filenames_test,KPTS_test)

# Obtiene las 2 descripciones BoW (SIFT, color) de las im�genes del conjunto de test
VW_SIFT_test=getAndSaveBoVWRepresentation(DSC_test,k,CB_SIFT,visual_words_filename_test_SIFT)
VW_COLOR_test=getAndSaveBoVWRepresentation(CDSC_test,k,CB_COLOR,visual_words_filename_test_COLOR)

# Entrena un clasificador SVM con las im�genes del conjunto de aprendizaje y lo eval�a utilizando las im�genes del conjunto de test
# para las 2 descripciones (SIFT, color)
# Devuelve la accuracy como medida del rendimiento del clasificador
ac_BOVW_SIFT = trainAndTestLinearSVM(VW_SIFT_train,VW_SIFT_test,GT_ids_train,GT_ids_test,C)
ac_BOVW_COLOR = trainAndTestLinearSVM(VW_COLOR_train,VW_COLOR_test,GT_ids_train,GT_ids_test,C)

# Entrena un clasificador SVM con la descripci�n SIFT.
# Al incluir el par�metro "probability=True", podremos luego recuperar la probabilidad asociada al resultado de la clasificaci�n para poder aplicar un esquema de late fusion.
# De forma previa al aprendizaje del clasificador, los datos se re-escalan para normalizarlos a media 0 y desviaci�n est�ndar 1.
stdSlr = StandardScaler().fit(VW_SIFT_train)
VW_SIFT_train_scaled = stdSlr.transform(VW_SIFT_train)
VW_SIFT_test_scaled = stdSlr.transform(VW_SIFT_test)
clf_SIFT = svm.SVC(kernel='linear', C=1,probability=True).fit(VW_SIFT_train_scaled, GT_ids_train)
ac_BOVW_SIFT = clf_SIFT.score(VW_SIFT_test_scaled, GT_ids_test)

# Entrena un clasificador SVM con la descripci�n de color.
# Al incluir el par�metro "probability=True", podremos luego recuperar la probabilidad asociada al resultado de la clasificaci�n para poder aplicar un esquema de late fusion.
# De forma previa al aprendizaje del clasificador, los datos se re-escalan para normalizarlos a media 0 y desviaci�n est�ndar 1.
stdSlr = StandardScaler().fit(VW_COLOR_train)
VW_COLOR_train_scaled = stdSlr.transform(VW_COLOR_train)
VW_COLOR_test_scaled = stdSlr.transform(VW_COLOR_test)
clf_COLOR = svm.SVC(kernel='linear', C=1,probability=True).fit(VW_COLOR_train_scaled, GT_ids_train)
ac_BOVW_COLOR = clf_COLOR.score(VW_COLOR_test_scaled, GT_ids_test)

# Eval�a los dos clasificadores entrenados previamente (para SIFT y color) con la funci�n "predict_proba".
# La funci�n "predict_proba" devuelve una probabilidad de clasificaci�n para cada una de las clases
# Para la representaci�n late fusion se concatenan los vectores de probabilidad con la confianza de cada una de las clases para ambos descriptores
late_train = np.hstack(( clf_SIFT.predict_proba(VW_SIFT_train_scaled),clf_COLOR.predict_proba(VW_COLOR_train_scaled)))
late_test =  np.hstack(( clf_SIFT.predict_proba(VW_SIFT_test_scaled),clf_COLOR.predict_proba(VW_COLOR_test_scaled)))

# Entrena y eval�a el clasificador final a partir de la representaci�n late fusion
stdSlr = StandardScaler().fit(late_train)
late_train_scaled = stdSlr.transform(late_train)
late_test_scaled =  stdSlr.transform(late_test)
clf_LATE = svm.SVC(kernel='linear', C=1).fit(late_train_scaled, GT_ids_train)
ac_BOVW_LF = clf_LATE.score(late_test_scaled,GT_ids_test)


print 'Accuracy BOVW with LinearSVM SIFT: '+str(ac_BOVW_SIFT)
print 'Accuracy BOVW with LinearSVM Color: '+str(ac_BOVW_COLOR)
print 'Accuracy BOVW with LinearSVM Late Fusion: '+str(ac_BOVW_LF)