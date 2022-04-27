%**Modifyable values**
%-------------------------------------------------------------------------
load bodyfatInputs; %replace with relevant location
load bodyfatTargets; %replace with relevant location

input = transpose(bodyfatInputs); %replace with real input
target = transpose(bodyfatTargets); %replace with real target

layer_sizes = [7, 4]; %Only hidden layers, Changing the amount of indexes affects nr of layers, changing the value changes nr of nodes.

mode = 1; %training or classification mode 0 = training, 1 = classification

%------------------------------------------------------------------------

if mode == 0 %if in training
    R = rmmissing([input target]); %remove missing values
    X = R(:,1:end-1);
    Y = R(:,end);

    %part data into training set and test set
    c = cvpartition(length(Y),"Holdout",0.20); 
    trainingIdx = training(c); % Indices for the training set
    XTrain = X(trainingIdx,:);
    YTrain = Y(trainingIdx);
    testIdx = test(c); % Indices for the test set
    XTest = X(testIdx,:);
    YTest = Y(testIdx);

    %train network
    network = fitrnet(XTrain,YTrain,"Standardize",true, ...
        "LayerSizes", layer_sizes);

    %evalulate loss with MSE, lower value = better 
    testMSE = loss(network,XTest,YTest);
    
    trainedNet = network;
    save trainedNet;

else %if in classification mode
    load trainedNet;
    input = rmmissing(input); %remove missing values
    testPredictions = predict(network, input);
end