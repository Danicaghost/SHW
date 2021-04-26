clc;clear;



train_label=load('/home/cvnlp/wujinpeng/image-text-emotion/wujinpeng/Flickr&Instagram/dataset_5/Lable.mat');
trainX=load('/home/cvnlp/wujinpeng/image-text-emotion/wujinpeng/Flickr&Instagram/dataset_5/deepfashion_vgg19_1.mat');
trainY=load('/home/cvnlp/wujinpeng/image-text-emotion/wujinpeng/Flickr&Instagram/dataset_5/deepfeature_5_vgg16_1.mat');

% testX=load('C:\Users\Administrator\Desktop\����\ccaFuse-master\����\SIFT+RGB.mat');
% testY=load('C:\Users\Administrator\Desktop\����\ccaFuse-master\����\LBP.mat');
X=double(trainX.vgg19_1);Y=double(trainY.vgg16_1);label=train_label.Lable;
length(X)
length(Y)
length(label)
   

class_label=[1,0];
all={X,Y,label};
use_less=[];
X_1=[];Y_1=[];label_1=[];
X_0=[];Y_0=[];label_0=[];
X_2=[];Y_2=[];label_2=[];
X_3=[];Y_3=[];label_3=[];
X_4=[];Y_4=[];label_4=[];
X_5=[];Y_5=[];label_5=[];
X_6=[];Y_6=[];label_6=[];
X_7=[];Y_7=[];label_7=[];
[X_hang,X_lie]=size(X);
for j=1:X_hang
   if label(j)==1
       X_1=[X_1;X(j,:)];
       Y_1=[Y_1;Y(j,:)];
       label_1=[label_1;label(j)];
   end
   if label(j)==0
       X_0=[X_0;X(j,:)];
       Y_0=[Y_0;Y(j,:)];
       label_0=[label_0;label(j)];
   end
   if label(j)==2
       X_2=[X_2;X(j,:)];
       Y_2=[Y_2;Y(j,:)];
       label_2=[label_2;label(j)];
   end
   if label(j)==3
       X_3=[X_3;X(j,:)];
       Y_3=[Y_3;Y(j,:)];
       label_3=[label_3;label(j)];
   end
   if label(j)==4
       X_4=[X_4;X(j,:)];
       Y_4=[Y_4;Y(j,:)];
       label_4=[label_4;label(j)];
   end
   if label(j)==5
       X_5=[X_5;X(j,:)];
       Y_5=[Y_5;Y(j,:)];
       label_5=[label_5;label(j)];
   end
   if label(j)==6
       X_6=[X_6;X(j,:)];
       Y_6=[Y_6;Y(j,:)];
       label_6=[label_6;label(j)];
   end
   if label(j)==7
       X_7=[X_7;X(j,:)];
       Y_7=[Y_7;Y(j,:)];
       label_7=[label_7;label(j)];
   end
end

% 7-3��
train_1_label=[];train_0_label=[];train_2_label=[];train_3_label=[];train_4_label=[];train_5_label=[];train_6_label=[];train_7_label=[];
test_1_label=[];test_0_label=[];test_2_label=[];test_3_label=[];test_4_label=[];test_5_label=[];test_6_label=[];test_7_label=[];
% 1
train_x_1=[];train_y_1=[];
test_x_1=[];test_y_1=[];
[X_1_hang,X_1_lie]=size(X_1);
for i=1:X_1_hang
    if i<=X_1_hang*0.7
        train_x_1=[train_x_1;X_1(i,:)];
        train_y_1=[train_y_1;Y_1(i,:)];
        train_1_label=[train_1_label;1];
    else
        test_x_1=[test_x_1;X_1(i,:)];
        test_y_1=[test_y_1;Y_1(i,:)];
        test_1_label=[test_1_label;1];
    end
end
% 0
train_x_0=[];train_y_0=[];
test_x_0=[];test_y_0=[];
[X_0_hang,X_0_lie]=size(X_0);
for i=1:X_0_hang
    if i<=X_0_hang*0.7
        train_x_0=[train_x_0;X_0(i,:)];
        train_y_0=[train_y_0;Y_0(i,:)];
        train_0_label=[train_0_label;0];
    else
        test_x_0=[test_x_0;X_0(i,:)];
        test_y_0=[test_y_0;Y_0(i,:)];
        test_0_label=[test_0_label;0];
    end
end
% 2
train_x_2=[];train_y_2=[];
test_x_2=[];test_y_2=[];
[X_2_hang,X_2_lie]=size(X_2);
for i=1:X_2_hang
    if i<=X_2_hang*0.7
        train_x_2=[train_x_2;X_2(i,:)];
        train_y_2=[train_y_2;Y_2(i,:)];
        train_2_label=[train_2_label;2];
    else
        test_x_2=[test_x_2;X_2(i,:)];
        test_y_2=[test_y_2;Y_2(i,:)];
        test_2_label=[test_2_label;2];
    end
end
% 3
train_x_3=[];train_y_3=[];
test_x_3=[];test_y_3=[];
[X_3_hang,X_3_lie]=size(X_3);
for i=1:X_3_hang
    if i<=X_3_hang*0.7
        train_x_3=[train_x_3;X_3(i,:)];
        train_y_3=[train_y_3;Y_3(i,:)];
        train_3_label=[train_3_label;3];
    else
        test_x_3=[test_x_3;X_3(i,:)];
        test_y_3=[test_y_3;Y_3(i,:)];
        test_3_label=[test_3_label;3];
    end
end
% 4
train_x_4=[];train_y_4=[];
test_x_4=[];test_y_4=[];
[X_4_hang,X_4_lie]=size(X_4);
for i=1:X_4_hang
    if i<=X_4_hang*0.7
        train_x_4=[train_x_4;X_4(i,:)];
        train_y_4=[train_y_4;Y_4(i,:)];
        train_4_label=[train_4_label;4];
    else
        test_x_4=[test_x_4;X_4(i,:)];
        test_y_4=[test_y_4;Y_4(i,:)];
        test_4_label=[test_4_label;4];
    end
end
% 5
train_x_5=[];train_y_5=[];
test_x_5=[];test_y_5=[];
[X_5_hang,X_5_lie]=size(X_5);
for i=1:X_5_hang
    if i<=X_5_hang*0.7
        train_x_5=[train_x_5;X_5(i,:)];
        train_y_5=[train_y_5;Y_5(i,:)];
        train_5_label=[train_5_label;5];
    else
        test_x_5=[test_x_5;X_5(i,:)];
        test_y_5=[test_y_5;Y_5(i,:)];
        test_5_label=[test_5_label;5];
    end
end
% 6
train_x_6=[];train_y_6=[];
test_x_6=[];test_y_6=[];
[X_6_hang,X_6_lie]=size(X_6);
for i=1:X_6_hang
    if i<=X_6_hang*0.7
        train_x_6=[train_x_6;X_6(i,:)];
        train_y_6=[train_y_6;Y_6(i,:)];
        train_6_label=[train_6_label;6];
    else
        test_x_6=[test_x_6;X_6(i,:)];
        test_y_6=[test_y_6;Y_6(i,:)];
        test_6_label=[test_6_label;6];
    end
end
% 7
train_x_7=[];train_y_7=[];
test_x_7=[];test_y_7=[];
[X_7_hang,X_7_lie]=size(X_7);
for i=1:X_7_hang
    if i<=X_7_hang*0.7
        train_x_7=[train_x_7;X_7(i,:)];
        train_y_7=[train_y_7;Y_7(i,:)];
        train_7_label=[train_7_label;7];
    else
        test_x_7=[test_x_7;X_7(i,:)];
        test_y_7=[test_y_7;Y_7(i,:)];
        test_7_label=[test_7_label;7];
    end
end
% ƴ��
train_X=[];train_Y=[];test_X=[];test_Y=[];
train_X=[train_x_0;train_x_1;train_x_2;train_x_3;train_x_4;train_x_5;train_x_6;train_x_7];
train_Y=[train_y_0;train_y_1;train_y_2;train_y_3;train_y_4;train_y_5;train_y_6;train_y_7];
test_X=[test_x_0;test_x_1;test_x_2;test_x_3;test_x_4;test_x_5;test_x_6;test_x_7];
test_Y=[test_y_0;test_y_1;test_y_2;test_y_3;test_y_4;test_y_5;test_y_6;test_y_7];
train_label=[train_0_label;train_1_label;train_2_label;train_3_label;train_4_label;train_5_label;train_6_label;train_7_label];
test_label=[test_0_label;test_1_label;test_2_label;test_3_label;test_4_label;test_5_label;test_6_label;test_7_label];

[trainZ,testZ]=ccaFuse(train_X, train_Y,test_X,test_Y, 'concat');
size(trainZ)
save('/home/cvnlp/wujinpeng/image-text-emotion/wujinpeng/Flickr&Instagram/dataset_5/cca/deep/cca_vgg19_vgg16.mat','-v7.3','trainZ','testZ','train_label','test_label');