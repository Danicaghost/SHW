function meanccaResult = getclusterccaSnewsvd(X,Y,label,test)
% global U1
% global V1
% global A
% global B
% global k
global trainX
global trainY
global testX
global testY
global test_index
global train_index
global trainZ1
global testZ1
global r

% Inputs:
%       X       :   pxn matrix containing the first set of all feature vectors
%                   p:  dimensionality of the first feature set
%                   n:  number of all samples
% 
%       Y       :   qxn matrix containing the second set of all feature vectors
%                   q:  dimensionality of the second feature set
% 
%       label   :   1xn row vector of length n containing the class labels
%       test    :   nx1 row vector of length n containing the logical value of test index 
%   
    test_index = double(find(ismember(test,'TRUE')==1));
    train_index = double(find(ismember(test,'FALSE')==1));
    trainLable1 = label(train_index);
    trainLable2 = label(train_index);
    trainX = X(:,train_index);
    trainY = Y(:,train_index);


    testX = X(:,test_index);
    testY = Y(:,test_index);
%     [A,B,r,U,V] = mean_cca_by_svd(trainX',trainY',trainLable1,trainLable2)
    [Wx,Wy,r] = cluster_cca_by_svd(trainX,trainY,trainLable1,trainLable2,1)
    trainx = real(Wx.'*trainX);
    trainy = real(Wy.'*trainY);
%     testXmeanccaa = r*A'*testX;
%     testYmeanccaa = r*B'*testY;
%     trainx = real(r*Wx'*trainX);
%     trainy = real(r*Wy'*trainY);
%     testXmeanccaa = real(r*Wx'*testX);
%     testYmeanccaa = real(r*Wy'*testY);
%     trainx = r*Wx'*trainX;
%     trainy = r*Wy'*trainY;
    testXmeanccaa = real(Wx.'*testX);
    testYmeanccaa = real(Wy.'*testY);
% global trainZ1
% global testZ1

%MEANCCA
    trainZ1 = [trainx + trainy];
    testZ1  = [testXmeanccaa + testYmeanccaa];

%CLUSTERCCA     
%     trainZ1 = [trainx + trainy];
%     testZ1  = [testXmeanccaa + testYmeanccaa];
    
    k = size(trainZ1,1);


    meanccaResult = ones(k,(size(trainZ1,k)+size(testZ1,k)));
    meanccaResult(:,test_index) = testZ1;
    meanccaResult(:,train_index) = trainZ1;
%     dcaResult = dcaResult';
    
end