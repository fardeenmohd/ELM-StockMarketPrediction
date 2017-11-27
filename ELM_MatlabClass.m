classdef ELM_MatlabClass
% Extreme Learning Machine (ELM) class for regression.
%   Author: Fardin Mohammed
%   Date of revision : 27.11.2017
%   ELM = ELM_MatlabClass(nInputs,nHidden,activationFunction)
%
%   where:
%           nInputs     -> number of inputs in the dataset (single output) 
%           nHidden     -> number of hidden neurons of the ELM 
%           activationFunction -> 
%
%                          Options:
%                          'tanh'      hyperbolic tangent (default),
%                          'logsig'    log-sigmoid,
%                          'linear'    linear transfer function, 
%                          'sine'      sine transfer function,
%                          'harlim' *  positive hard limit transfer function
%                          'tribas' *  triangular basis transfer function
%                          'radbas' *  radial basis transfer function *,
%                          
    
    properties (GetAccess = private)
        nFeatures         % number of features
        nHidden         % number of hidden nodes                
        activationFunction          % activation function     
        IW              % input weights
        OW              % output weights
        bias            % bias                
    end
    
    % public methods
    methods
       
    % constructor class for the ELM class
    function self = ELM_MatlabClass(nFeatures,nHidden,activationFunctionString)
        % inputs and hidden neurons
        self.nFeatures = nFeatures;
        self.nHidden = nHidden;
        
            switch activationFunctionString
                case 'tanh'
                    self.activationFunction = @(x) (1-2./(exp(2*x)+1));
                case 'sig'
                    self.activationFunction = @(x)(1./(1+exp(-x)));
                case 'linear'
                    self.activationFunction = @(x) (x);
                case 'radbas'
                    self.activationFunction = @radbas;
                case 'sine'
                    self.activationFunction = @sin;
                case 'hardlim'
                    self.activationFunction = @(x) (double(hardlim(x)));
                case 'tribas'
                    self.activationFunction = @tribas;
                otherwise
                    self.activationFunction = activationFunctionString;   % custom activation fucntion
            end
    end
    
    % train the ELM
    function self = train(self,trainX,trainY)
        % get output and inputs and number of entries n
        X = trainX';
        Y = trainY';
        [~,n] = size(X);
        % initialize input weights and bias randomly
        self.IW   = rand(self.nHidden,self.nFeatures);
        self.bias = rand(self.nHidden,1);
        % compute activation field F
        H = self.IW * X + repmat(self.bias,1,n);    
        % compute H
        H = self.activationFunction(H);
        % find OW from matrix H pseudo-inversion
        Hinv    = pinv(H');
        %finding the outerweights which is betas
        self.OW = Hinv * Y';                    
    end    
    
    % predict using ELM
    function Yhat = predict(self,testX)
        testX = testX';
        % get length of test dataset
        [~,n] = size(testX);
        % compute activation field F
        H = self.IW * testX + repmat(self.bias,1,n);
        % compute H
        H = self.activationFunction(H);
        % compute output
        Yhat = (H' * self.OW)';
        Yhat = Yhat';       
    end
    end

end
