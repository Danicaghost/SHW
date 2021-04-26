%%
clc;
clear;
global unq_a_label
global unq_b_label
global card_a
global card_b
global Cxx
global Cyy
global train_a_mean
global train_b_mean
global Cxy
global Cyx
global Wx
global Wy
global r
global m
global c
global d
% global U
% global V
% global r
% global A
% global B


load('E:\study\hj\MLABELDCA.mat');
load('E:\study\hj\mtest1.mat');
test = mtest1;

label = mmmm';
% create saving path
save_path= 'E:\study\my01\clustercca\55\sum\2\';

if ~exist(save_path,'dir')
    mkdir(save_path);
end
% label = importdata('./INbreast387/label&train/Label.csv')';
% test = importdata('./INbreast387/label&train/Train.csv');
% S = importdata('E:\study\jzlpaper1\MAT\sift/sift.csv')';
% G = importdata('E:\study\jzlpaper1\MAT\gist\gist.csv')';
% H = importdata('E:\study\jzlpaper1\MAT\hog/hog.csv')';
% V =  importdata('E:\study\jzlpaper1\MAT\lbp/vgg16.csv')';
% R = importdata('E:\study\jzlpaper1\MAT\sd\/resnet50.csv')';
% load('E:\study\hj\f\new\help\ALLSiftVectors.mat','countVectors');
% load('E:\study\hj\f\new\help\ALLGistFea.mat','GistFeats');
% load('E:\study\hj\f\new\help\hog_right_43.mat','hog');
% load('E:\study\hj\f\new\help\ALLLBPFea.mat','LBPFeats');
% % load('E:\study\hj\f\new\help\resnet152R.mat','resnet152R');
% load('E:\study\hj\f\new\help\resnet50.mat','resnet50');
% load('E:\study\hj\f\new\help\densenet121.mat','densenet121');
% % load('E:\study\hj\f\new\help\densenet161R.mat','densenet161R');
% load('E:\study\hj\f\new\help\vgg16a.mat','vgg16a');
% load('E:\study\hj\f\new\help\vgg16b.mat','vgg16b');

% load('E:\study\hj\ma\ALLSiftVectors.mat','countVectors');
% load('E:\study\hj\ma\ALLGistFea.mat','GistFeats');
% load('E:\study\hj\ma\hog_right_43.mat','hog');
% % load('E:\study\hj\ma\ALLLBPFea.mat','LBPFeats');
% % load('E:\study\hj\ma\resnet152R.mat','resnet152R');
% % load('E:\study\hj\ma\resnet50R.mat','resnet50R');
% % load('E:\study\hj\ma\densenet121R.mat','densenet121R');
% % load('E:\study\hj\ma\densenet161R.mat','densenet161R');
% % load('E:\study\hj\ma\vgg16aR.mat','vgg16aR');
% % load('E:\study\hj\ma\vgg16R.mat','vgg16R');
% S1 = countVectors';
% G = GistFeats';
% % V16VA16R50D121 = importdata('E:\study\hj\ma\restart\ma_result\dca\4\V16VA16R50D121.csv')';
% % HS = importdata('E:\study\hj\ma\dca\55\HS.csv')';
% H = hog';
% L = LBPFeats';
% % R152 = resnet152';
% R50 = resnet50R';
% D121 = densenet121R';
% % D161 = densenet161';
% VA16 = vgg16aR';
% V16 = vgg16R';
% load('E:\study\my01\mat\REAL\resnet50R.mat');
% load('E:\study\my01\mat\REAL\resnet152R2.mat');
% load('E:\study\my01\mat\REAL\resnet101R2.mat');
% load('E:\study\my01\mat\REAL\resnetxt50R.mat');
% load('E:\study\my01\mat\REAL\resnetxt101R.mat');
load('E:\study\my01\mat\REAL\senet154R.mat');
load('E:\study\my01\mat\REAL\seresnet50Rn2.mat');
load('E:\study\my01\mat\REAL\seresnet101R2n.mat');
load('E:\study\my01\mat\REAL\seresnet152R2n.mat');
load('E:\study\my01\mat\REAL\seresnetxt50R.mat');
load('E:\study\my01\mat\REAL\seresnetxt101R.mat');
% R152 = resnet152R';
% R50 = resnet50R';
% R101 = resnet101R';
% RXT50 = resnetxt50R';
% RXT101 = resnetxt101R';
SE154 = senet154R';
SR152 = seresnet152R';
SR50 = seresnet50Rn2';
SR101 = seresnet101R';
SRXT50 = seresnetxt50R';
SRXT101 = seresnetxt101R';
D1D2 = importdata('E:\study\my01\DCAF\CCCASUM\1\D1D2.csv')';
feature_list = {'D1D2','SR152'};
len = length(feature_list);
for i=1:len-1
    for j=i+1:len
        fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
        new_feature_name = [feature_list{i} feature_list{j}];
        X= eval(feature_list{i});
        Y = eval(feature_list{j});
        fused_feature = getclustercca(X,Y,label,test)';
        eval([new_feature_name '=fused_feature;']);
        savepath = [save_path new_feature_name ];
        fprintf("save path : %s\n",savepath);
        save(savepath,new_feature_name);
        csvwrite([savepath '.csv'],eval(new_feature_name));
        %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
    end
end
SR50SRXT50 = importdata('E:\study\my01\clustercca\55\sum\2\SR50SRXT50.csv')';
feature_list = {'SR101','SR50SRXT50'};
len = length(feature_list);
for i=1:len-1
    for j=i+1:len
        fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
        new_feature_name = [feature_list{i} feature_list{j}];
        X= eval(feature_list{i});
        Y = eval(feature_list{j});
        fused_feature = getclustercca(X,Y,label,test)';
        eval([new_feature_name '=fused_feature;']);
        savepath = [save_path new_feature_name ];
        fprintf("save path : %s\n",savepath);
        save(savepath,new_feature_name);
        csvwrite([savepath '.csv'],eval(new_feature_name));
        %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
    end
end
SR101SR50SRXT50 = importdata('E:\study\my01\clustercca\55\sum\2\SR101SR50SRXT50.csv')';
feature_list = {'SR101SR50SRXT50','SR152'};
len = length(feature_list);
for i=1:len-1
    for j=i+1:len
        fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
        new_feature_name = [feature_list{i} feature_list{j}];
        X= eval(feature_list{i});
        Y = eval(feature_list{j});
        fused_feature = getclustercca(X,Y,label,test)';
        eval([new_feature_name '=fused_feature;']);
        savepath = [save_path new_feature_name ];
        fprintf("save path : %s\n",savepath);
        save(savepath,new_feature_name);
        csvwrite([savepath '.csv'],eval(new_feature_name));
        %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
    end
end
SR101SR50SRXT50SR152 = importdata('E:\study\my01\clustercca\55\sum\2\SR101SR50SRXT50SR152.csv')';
feature_list = {'SR101SR50SRXT50SR152','SRXT101'};
len = length(feature_list);
for i=1:len-1
    for j=i+1:len
        fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
        new_feature_name = [feature_list{i} feature_list{j}];
        X= eval(feature_list{i});
        Y = eval(feature_list{j});
        fused_feature = getclustercca(X,Y,label,test)';
        eval([new_feature_name '=fused_feature;']);
        savepath = [save_path new_feature_name ];
        fprintf("save path : %s\n",savepath);
        save(savepath,new_feature_name);
        csvwrite([savepath '.csv'],eval(new_feature_name));
        %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
    end
end
SR101SR50SRXT50SR152SRXT101 = importdata('E:\study\my01\clustercca\55\sum\2\SR101SR50SRXT50SR152SRXT101.csv')';
feature_list = {'SR101SR50SRXT50SR152SRXT101','SE154'};
len = length(feature_list);
for i=1:len-1
    for j=i+1:len
        fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
        new_feature_name = [feature_list{i} feature_list{j}];
        X= eval(feature_list{i});
        Y = eval(feature_list{j});
        fused_feature = getclustercca(X,Y,label,test)';
        eval([new_feature_name '=fused_feature;']);
        savepath = [save_path new_feature_name ];
        fprintf("save path : %s\n",savepath);
        save(savepath,new_feature_name);
        csvwrite([savepath '.csv'],eval(new_feature_name));
        %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
    end
end

%     dcaResult = dcaResult';
% save(savepath,new_feature_name);
% csvwrite([savepath '.csv'],eval(new_feature_name));
