classdef ELM_MatlabClass
% Extreme Learning Machine (ELM) class for regression.
%
%   
%   ELM = ELM_MatlabClass(nInputs,nHidden,actFun)
%
%   where:
%           nInputs     -> number of inputs in the dataset (single output) 
%           nHidden     -> number of hidden neurons of the ELM 
%           actFun -> 
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
        nInputs         % number of inputs
        nHidden         % number of hidden nodes                
        actFun          % activation function     
        IW              % input weights
        OW              % output weights
        bias            % bias                
    end
    
    % public methods
    methods
       
    % constructor
    function self = ELM_MatlabClass(nInputs,nHidden,actFunString)
        % # inputs and hidden neurons
        self.nInputs = nInputs;
        self.nHidden = nHidden;
            switch actFunString
                case 'tanh'
                    self.actFun = @(x) (1-2./(exp(2*x)+1));
                case 'sig'
                    self.actFun = @(x)(1./(1+exp(-x)));
                case 'linear'
                    self.actFun = @(x) (x);
                case 'radbas'
                    self.actFun = @radbas;
                case 'sine'
                    self.actFun = @sin;
                case 'hardlim'
                    self.actFun = @(x) (double(hardlim(x)));
                case 'tribas'
                    self.actFun = @tribas;
                otherwise
                    self.actFun = actFunString;   % custom activation fucntion
            end
  
    end
    
    % train the ELM
    function self = train(self,trainData,futureData)
    
        % get output and inputs and number of patterns n
        X = trainData';
        Y = futureData';
        [~,n] = size(X);
        % initialize inputs and bias randomly
        self.IW   = rand(self.nHidden,self.nInputs);
        self.bias = ones(self.nHidden,1);
        % compute activation field F
        H = self.IW * X + repmat(self.bias,1,n);    
        % compute H
        %H = self.actFun(F);
        % find OW from matrix H pseudo-inversion
        Hinv    = pinv(H');
        self.OW = Hinv' * Y';                    
    end    
    
    % predict using ELM
    function Yhat = predict(self,X)
        % get length of test dataset
        X = X';
        [~,n] = size(X);
        % compute activation field F
        F = self.IW * X + repmat(self.bias,1,n);
        % compute H
        H = self.actFun(F);
        % compute output
        Yhat = (H * self.OW)';
        Yhat = Yhat';       
    end
    end

end
