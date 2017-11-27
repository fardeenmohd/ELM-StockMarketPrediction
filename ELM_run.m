%%Introduction to the ELM run script
%Author : Fardin Mohammed
%Date of revision : 27.11.2017
%Please the change the arguments according to your needs

%% important arguments for the script
activationFunction = 'linear';
dataFile = 'newstocks.txt';
rowsToSkip = 1;
columnsToSkipFromLeft = 1; 
columnsToSkipFromRight = 1;
hiddenLayerSize = 10;
trainingPercentage = 50;
daysToPredict = 5;

%% data loading and preprocessing
% load data
pureData = csvread(dataFile,rowsToSkip,columnsToSkipFromLeft);

% acquire X
X = pureData(daysToPredict:end,1:end - columnsToSkipFromRight);

% get number of entries and features
[nEntries, nFeatures] = size(X);

% acquire Y 
Y = pureData(1:nEntries,1:nFeatures);

% finding split points in data
percTraining = trainingPercentage/100; 
endTraining  = ceil(percTraining * nEntries);

% dividing X and Y
% training data 
trainX = X(1:endTraining,:); 
trainY = Y(1:endTraining,:);

% testing data
testX = X(endTraining+1:end,:);
testY = Y(endTraining+1:end,:);

%% creation and training of ELM model

% create ELM
ELM = ELM_MatlabClass(nFeatures,hiddenLayerSize,activationFunction);

% train ELM on the training dataset
ELM = train(ELM,trainX,trainY);

%% validation of ELM model
predictionTest = predict(ELM,testX);
disp('Statistics when predicting on testing data');
fprintf('Testing Rsquare(close to 1 means nice prediction) = %3.3f\n',computeR2(testX,predictionTest));
fprintf('Testing Root mean square error of open price = %3.3f\n',computeRMSE(testX(:,1),predictionTest(:,1)));
fprintf('Testing Root mean square error of high price = %3.3f\n',computeRMSE(testX(:,2),predictionTest(:,2)));
fprintf('Testing Root mean square error of low price = %3.3f\n',computeRMSE(testX(:,3),predictionTest(:,3)));
fprintf('Testing Root mean square error of close price = %3.3f\n',computeRMSE(testX(:,4),predictionTest(:,4)));

% compute and report accuracy on training dataset
predictionTrain = predict(ELM,trainX);
disp('Statistics when predicting on training data');
fprintf('Training Rsquare(close to 1 means nice prediction) = %3.3f\n',computeR2(trainX,predictionTrain));
fprintf('Training Root mean square error of open price = %3.3f\n',computeRMSE(trainX(:,1),predictionTrain(:,1)));
fprintf('Training Root mean square error of high price = %3.3f\n',computeRMSE(trainX(:,2),predictionTrain(:,2)));
fprintf('Training Root mean square error of low price = %3.3f\n',computeRMSE(trainX(:,3),predictionTrain(:,3)));
fprintf('Training Root mean square error of close price = %3.3f\n',computeRMSE(trainX(:,4),predictionTrain(:,4)));

%% sensitivity analysis on number of hidden neurons
hiddenLayerSize    = 1:10:100;
trainR2   = zeros(size(hiddenLayerSize));
testR2   = zeros(size(hiddenLayerSize));
for i = 1 : numel(hiddenLayerSize)
    % create ELM for classification
    ELM = ELM_MatlabClass(nFeatures,hiddenLayerSize(i),activationFunction);
    % train ELM on the training dataset
    ELM = train(ELM,trainX,trainY);
    Yhat = predict(ELM,trainX);
    trainR2(i) = computeR2(trainX,Yhat);
    % validation of ELM model
    Yhat = predict(ELM,testX);
    testR2(i) = computeR2(testX,Yhat);
end

% plot results of accuracy with different sizes of the hidden layer
figure;
plot(hiddenLayerSize,[trainR2;testR2],'-o');
xlabel('Number of Hidden Neurons');
ylabel('R square error');
legend({'training','testing'},'Location','southeast');



