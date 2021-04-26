%%
clc;
clear;
% global train_a
% global train_b
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
global mu_x
global mu_y
global trainX
global trainY
global testX
global testY
global test_index
global train_index
global trainZ1
global testZ1
global r
% global U
% global V
% global r
% global A
% global B

tic;
load('../pycharmgive/label/twSR/tweenterSRlabel.mat');
load('../pycharmgive/label/twSR/tweenterSRtest.mat');
test = tweenterSRtest;

label = tweenterSRlabel';
% create saving path
% save_path= 'E:\study\my01\clustercca\55\sum\2\';
save_path= '../pycharmgive/tweenter1SR/test/';
% save_path= 'E:\study\my01\DCAF\CCCASUM\1\CLUT\';
if ~exist(save_path,'dir')
    mkdir(save_path);
end



% load('../PYCHARM/fusionmat/resnet50R.mat');
% load('../PYCHARM/fusionmat/resnet152R2.mat');
% load('../PYCHARM/fusionmat/resnet101R2.mat');
% load('../PYCHARM/fusionmat/resnetxt50R.mat');
% load('../PYCHARM/fusionmat/resnetxt101R.mat');
% load('../PYCHARM/fusionmat/senet154R.mat');
load('../pycharmgive/tweenter1SR/seresnet50R.mat');
load('../pycharmgive/tweenter1SR/seresnet101R.mat');
load('../pycharmgive/tweenter1SR/seresnet152R.mat');
load('../pycharmgive/tweenter1SR/seresnetxt50R.mat');
load('../pycharmgive/tweenter1SR/seresnetxt101R.mat');
% R152 = resnet152R';
% R50 = resnet50R';
% R101 = resnet101R';
% RXT50 = resnetxt50R';
% RXT101 = resnetxt101R';
% SE154 = senet154R';
% seresnet1522n = reshape(seresnet1522n,943,2048)';
% seresnet50 = reshape(seresnet50,943,2048)';
% seresnet1012n = reshape(seresnet1012n,943,2048)';
% seresnetxt50 = reshape(seresnetxt50,943,2048)';
% seresnetxt101R = reshape(seresnetxt101,943,2048)';
SR152 = double(seresnet152R)';
SR50 = double(seresnet50R)';
SR101 = double(seresnet101R)';
SRXT50 = double(seresnetxt50R)';
SRXT101 = double(seresnetxt101R)';
% D1D2 = importdata('E:\study\my01\DCAF\CCCACON\1\D1D2_0.csv')';



feature_list = {'SR50','SRXT50'};
len = length(feature_list);
for i=1:len-1
    for j=i+1:len
        fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
        new_feature_name = [feature_list{i} feature_list{j}];
        X= eval(feature_list{i});
        Y = eval(feature_list{j});
        fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
        eval([new_feature_name '=fused_feature;']);
        savepath = [save_evalpath new_feature_name ];
        fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
        %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
    end
end
toc;
% feature_list = {'SR101','SRXT101'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end
% 
% feature_list = {'SR50','SR101'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end
% feature_list = {'SR50','SR152'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end
% feature_list = {'SR50','SRXT101'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end
% 
% feature_list = {'SR152','SR101'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end
% 
% feature_list = {'SR152','SRXT50'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end
% 
% feature_list = {'SR152','SRXT101'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end
% 
% feature_list = {'SR101','SRXT50'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end
% 
% feature_list = {'SRXT50','SRXT101'};
% len = length(feature_list);
% for i=1:len-1
%     for j=i+1:len
%         fprintf("dca fuse %s : %s  \n",feature_list{i},feature_list{j});
%         new_feature_name = [feature_list{i} feature_list{j}];
%         X= eval(feature_list{i});
%         Y = eval(feature_list{j});
%         fused_feature = getclusterccaCnewsvd(X,Y,label,test)';
%         eval([new_feature_name '=fused_feature;']);
%         savepath = [save_path new_feature_name ];
%         fprintf("save path : %s\n",savepath);
%         save(savepath,new_feature_name);
%         csvwrite([savepath '.csv'],eval(new_feature_name));
%         %     fprintf("i:%.0f len: %.0f\n",kk, len_kk);
%     end
% end