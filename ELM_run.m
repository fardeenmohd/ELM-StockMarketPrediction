%% Important arguments for the script
actFun = 'tanh';
dataFile = 'stocks.csv';
rowsToSkip = 0;
columnsToSkip = 0;
numberOfHiddenNeurons = 10;
trainingPercentage = 50;
daysToPredict = 10;

%% data loading and preprocessing
% load data
pureData = csvread(dataFile,rowsToSkip,columnsToSkip);
futureData = pureData(1:daysToPredict,:);
data = pureData(daysToPredict+1:end,:);

% get number of inputs and patterns
[nEntries, nInputs] = size(data);

% divide datasets
percTraining = trainingPercentage/100; 
endTraining  = ceil(percTraining * nEntries);

trainData = data(1:endTraining,:); 
testData = data(endTraining+1:end,:);

%% creation and training of ELM model

% create ELM for classification
ELM = ELM_MatlabClass(nInputs,numberOfHiddenNeurons,actFun);

% train ELM on the training dataset
ELM = train(ELM,trainData,futureData);

% compute and report accuracy on training dataset
Yhat = predict(ELM,trainData);
fprintf('Training Rsquare = %3.3f\n',computeR2(futureData,Yhat));



%% validation of ELM model
Yhat = predict(ELM,testData);
fprintf('Testing Rsquare = %3.3f\n',computeR2(futureData,Yhat));


% %% sensitivity analysis on number of hidden neurons
% nHidden    = 1:10:100;
% trainR2   = zeros(size(nHidden));
% testR2   = zeros(size(nHidden));
% for i = 1 : numel(nHidden)
%     % create ELM for classification
%     ELM = ELM_MatlabClass(nInputs,nHidden(i),actFun);
%     % train ELM on the training dataset
%     ELM = train(ELM,trainData);
%     Yhat = predict(ELM,trainData(:,1:end));
%     trainR2(i) = computeR2(trainData(:,1:end),Yhat);
%     % validation of ELM model
%     Yhat = predict(ELM,testData(:,1:end));
%     testR2(i) = computeR2(testData(:,1:end),Yhat);
% end
% 
% % plot results
% plot(nHidden,[trainR2;testR2],'-o');
% xlabel('Number of Hidden Neurons');
% ylabel('R square error');
% legend({'training','testing'},'Location','southeast')
% 
% 

