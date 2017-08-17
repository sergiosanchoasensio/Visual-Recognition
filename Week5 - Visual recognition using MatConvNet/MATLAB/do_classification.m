%alexNet = load('../data/nets/imagenet-alex.mat');
% classes = alexNet.classes.description;
% alexNet = dagnn.DagNN.fromSimpleNN(alexNet, 'canonicalNames', true);
% alexNet.meta.classes.description = classes;
% alexNet.conserveMemory=0
% addpath('/home/joost/work//matlab/libraries/fast-additive-svms/libsvm-mat-3.0-1/');

[train_data,test_data,train_labels,test_labels]=CNNfeat('./../data/MIT_split',alexNet);
train_data=normalize(train_data,2);
test_data=normalize(test_data,2);

display('Classifying');

cc=20;

options=sprintf('-t 0 -c %f -b 1',cc);
model=svmtrain(train_labels',train_data,options);
[predict_label, accuracy , dec_values] = svmpredict(double(test_labels'),double(test_data), model,'-b 1');
