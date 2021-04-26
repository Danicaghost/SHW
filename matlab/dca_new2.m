load('/DATA/shihaowei/new/pycharmgive/label/FISR/FIlabel.mat');
load('/DATA/shihaowei/new/pycharmgive/label/FISR/FItest.mat');
test_1 = FItest;
%test_label =load('C:/Users/66/Documents/WeChat Files/alberthyne/Files/dcaFuse-master/����/Gist.mat');
% test_x=load('C:\Users\Administrator\Desktop\���ݼ�\����\SIFT.mat');
% test_y=load('C:\Users\Administrator\Desktop\���ݼ�\����\VGG.mat');

% Inputs:
%       X       :   pxn matrix containing the first set of training feature vectors
%                   p:  dimensionality of the first feature set
%                   n:  number of training samples
% 
%       Y       :   qxn matrix containing the second set of training feature vectors
%                   q:  dimensionality of the second feature set
% 
%       label   :   1xn row vector of length n containing the class labels

% X=double(train_x.resnet152_2);Y=double(train_y.resnet50_4);label=train_label.Lable;

%  X = reshape(X,882,2048);
%  Y = reshape(Y,882,2048);
label = FIlabel';
% create saving path
% save_path= 'E:\study\my01\clustercca\55\sum\2\';
save_path= '/DATA/shihaowei/new/pycharmgive/FISR/dca/';
% save_path= 'E:\study\my01\DCAF\CCCASUM\1\CLUT\';
if ~exist(save_path,'dir')
    mkdir(save_path);
end

load('/DATA/shihaowei/new/pycharmgive/FISR/seresnet50R.mat');
load('/DATA/shihaowei/new/pycharmgive/FISR/seresnet101R.mat');
load('/DATA/shihaowei/new/pycharmgive/FISR/seresnet152R.mat');
load('/DATA/shihaowei/new/pycharmgive/FISR/seresnetxt50R.mat');
load('/DATA/shihaowei/new/pycharmgive/FISR/seresnetxt101R.mat');
% R152 = resnet152R';
SR152 = double(seresnet152R)';
SR50 = double(seresnet50R)';
SR101 = double(seresnet101R)';
SRXT50 = double(seresnetxt50R)';
SRXT101 = double(seresnetxt101R)';

test_index = double(find(ismember(test_1,'TRUE')==1));
train_index = double(find(ismember(test_1,'FALSE')==1));
trainLable = label(train_index);
testLable = label(test_index);



feature_list = {'SR50','SRXT50'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');

feature_list = {'SR101','SRXT101'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');

feature_list = {'SR50','SR101'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');


feature_list = {'SR50','SR152'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');


feature_list = {'SR50','SRXT101'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');


feature_list = {'SR152','SR101'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');


feature_list = {'SR152','SRXT50'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');


feature_list = {'SR152','SRXT101'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');


feature_list = {'SR101','SRXT50'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');


feature_list = {'SRXT50','SRXT101'};

new_feature_name = ['dca_' feature_list{1} feature_list{2}];
X= eval(feature_list{1});
Y = eval(feature_list{2});
trainX = X(:,train_index);
trainY = Y(:,train_index);
testX = X(:,test_index);
testY = Y(:,test_index);
[Ax, Ay, trainXdca, trainYdca] = dcaFuse(trainX,trainY,trainLable);
testXdca = Ax * testX;
testYdca = Ay * testY;
all= [trainXdca ; trainYdca]';
test = [testXdca ; testYdca]';
savepath = [save_path new_feature_name ];
save(savepath,'-v7.3','all','test','trainLable','testLable');