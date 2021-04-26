function [trainZ,testZ] = ccaFuse(trainX, trainY, testX, testY, mode)
% CCAFUSE applies feature level fusion using a method based on Canonical
% Correlation Analysis (CCA). 
% Feature fusion is the process of combining two feature vectors to obtain 
% a single feature vector, which is more discriminative than any of the 
% input feature vectors. 
% CCAFUSE gets the train and test data matrices from two modalities X & Y,
% and consolidates them into a single feature set Z.
% 
% 
%   Details can be found in:
%   
%   M. Haghighat, M. Abdel-Mottaleb, W. Alhalabi, "Fully Automatic Face 
%   Normalization and Single Sample Face Recognition in Unconstrained 
%   Environments," Expert Systems With Applications, vol. 47, pp. 23-34, 
%   April 2016.
%   http://dx.doi.org/10.1016/j.eswa.2015.10.047
% 
% 
% Inputs:
%       trainX	:	nxp matrix containing the first set of training data
%                   n:  number of training samples
%                   p:  dimensionality of the first feature set
% 
%       trainY	:	nxq matrix containing the second set of training data
%                   q:  dimensionality of the second feature set
% 
%       testX	:	mxp matrix containing the first set of test data
%                   m:  number of test samples
% 
%       testY	:	mxq matrix containing the second set of test data
% 
%       mode    :   fusion mode: 'concat' or 'sum' (default: 'sum')
% 
% Outputs:
%       trainZ  :   matrix containing the fused training data
%       testZ   :   matrix containing the fused test data
% 
% 
% Sample use:
% [trainZ,testZ] = ccaFuse(trainX, trainY, testX, testY, 'sum');
% 
% 
% (C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       PLEASE CITE THE ABOVE PAPER IF YOU USE THIS CODE.


[n,p] = size(trainX);
if size(trainY,1) ~= n
    error('trainX and trainY must have the same number of samples.');
elseif n == 1
    error('trainX and trainY must have more than one sample.');
end
q = size(trainY,2);


if size(testX,2) ~= p
    error('trainX and testX must have the same dimensions.');
end

if size(testY,2) ~= q
    error('trainY and testY must have the same dimensions.');
end

if size(testX,1) ~= size(testY,1)
    error('testX and testY must have the same number of samples.');
end

if ~exist('mode', 'var')
    mode = 'sum';	% Default fusion mode
end


%% Center the variables

% 对trainX 和 trainY 去均值, 

meanX = mean(trainX);
meanY = mean(trainY);
trainX = bsxfun(@minus, trainX, meanX);  
testX  = bsxfun(@minus, testX,  meanX);
trainY = bsxfun(@minus, trainY, meanY);
testY  = bsxfun(@minus, testY,  meanY);


%% Dimensionality reduction using PCA for the first data X  降维


% Calculate the covariance matrix 计算协方差矩阵
if n >= p       %  [n,p] = size(trainX) n: sample numbers, 列向量个数。
    C = trainX' * trainX;	% pxp 
else
    C = trainX  * trainX';	% nxn
end

% Perform eigenvalue decomposition 特征值分解
[eigVecs, eigVals] = eig(C); % 得到特征向量和特征值
eigVals = abs(diag(eigVals)); % 得到对角矩阵


% Ignore zero eigenvalues 忽略零特征值
maxEigVal = max(eigVals);  % 得到最大的特征值
zeroEigIdx = find((eigVals/maxEigVal)<1e-6); 
% zeroEigIdx = find(eigVals<2)

% (1.0e+03)
 
eigVals(zeroEigIdx) = [];
eigVecs(:,zeroEigIdx) = [];  % 将特征值不满足的对应的特征向量置为0

% Sort in descending order
[~,index] = sort(eigVals,'descend'); % ~表示不取该变量的值，只取index
eigVals = eigVals(index); % 将特征值降序排列





eigVecs = eigVecs(:,index);  % 特征向量降序排列

% Obtain the projection matrix % 获得投影矩阵
if n >= p   %  [n,p] = size(trainX)
    Wxpca = eigVecs;
else
    Wxpca = trainX' * eigVecs * diag(1 ./ sqrt(eigVals));  % 该步骤对应的公式不理解数学含义
    % Ax = lamda*x  x:lamda对应的特征向量  lamda： 特征值
    
end
clear C eigVecs eigVals maxEigVal zeroEigIndex

% Update the first train and test data
trainX = trainX * Wxpca;
testX = testX * Wxpca;


%% Dimensionality reduction using PCA for the second data Y

% Calculate the covariance matrix
if n >= q        % q = size(trainY,2);
    C = trainY' * trainY;	% qxq
else
    C = trainY  * trainY';	% nxn
end

% Perform eigenvalue decomposition
[eigVecs, eigVals] = eig(C);
eigVals = abs(diag(eigVals));

% Ignore zero eigenvalues
maxEigVal = max(eigVals);
zeroEigIndex = find((eigVals/maxEigVal)<1e-6);
eigVals(zeroEigIndex) = [];
eigVecs(:,zeroEigIndex) = [];

% Sort in descending order
[~,index] = sort(eigVals,'descend');
eigVals = eigVals(index);
eigVecs = eigVecs(:,index);

% Obtain the projection matrix
if n >= q
    Wypca = eigVecs;
else
    Wypca = trainY' * eigVecs * diag(1 ./ sqrt(eigVals));
end
clear C eigVecs eigVals maxEigVal zeroEigIndex

% Update the second train and test data
trainY = trainY * Wypca;
testY = testY * Wypca;


%% Fusion using Canonical Correlation Analysis (CCA)

[Wxcca,Wycca] = canoncorr(trainX,trainY);

trainXcca = trainX * Wxcca;
trainYcca = trainY * Wycca;
testXcca = testX * Wxcca;
testYcca = testY * Wycca;

if strcmp(mode, 'concat')	% Fusion by concatenation (Z1)
    trainZ = [trainXcca, trainYcca];
    testZ  = [testXcca, testYcca];
else                        % Fusion by summation (Z2)
    trainZ = trainXcca + trainYcca;
    testZ  = testXcca + testYcca;
end
