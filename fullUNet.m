% U-NET
clear;clc;close all

% DATASET SETTINGS
pathDataset = 'MerchData'; % path of the folder containing the images to be learnt, subdivided into folders per class
fileExtensions = {'.jpg'}; % image extensions to be accepted
trainingPercentage = 0.75; % data percentage for training set
balanceDataset = true; %true/false if your dataset is unbalanced you can balance it

% DATA AUGMENTATION/TRANSFORMATION SETTINGS
dataAugmentation = true; % true/false if you need to perform data augmentation or data transformation (rgb2gray, resize etc). If false, you can ignore next settings (DATA AUGMENTATION SETTINGS)
imageSize = [64 64]; % images output size, width and height must be a multiple of 2^encoderDepth
outputSizeMode = 'resize'; % 'resize', 'centercrop' or 'randcrop'
colorPreprocessing = 'none'; % 'none', 'gray2rgb' or 'rgb2gray'

RandXReflection = false;
RandYReflection = false;
RandRotation = [-15 15]; % default [0 0]
RandScale = [0.8 1.2]; % default [1 1]
RandXShear = [0 0]; % default [0 0]
RandYShear = [0 0]; % default [0 0]
RandXTranslation = [0 0]; % default [0 0]
RandYTranslation = [0 0];  % default [0 0]

% NETWORK SETTINGS
encoderDepth = 4; % default 4, depth of the encoder/decoder structure in U-Net, i.e. number of images downsampling/upsampling
numFirstEncoderFilters = 32; % default 32, number of output channels for the first encoder stage, in each subsequent encoder stage the number of output channels doubles
filterSize = 3; % default 3, convolutional layer filter size (it can be also defined as [hieght width])
convolutionPadding = 'same'; % 'same' for zero padding applied to the inputs, 'valid' otherwise

% TRAINING SETTINGS
solverName = 'sgdm'; % 'sgdm', 'rmsprop' or 'adam', solver for training network
maxEpochs = 30; % Maximum number of epochs
miniBatchSize = 128; % Size of mini-batch
shuffleMode = 'once'; % 'once', 'never' or 'every-epoch', option for data shufflind during training/validation
validationFrequency = 5; % Frequency of network validation
initialLearnRate = 1e-3; % default 0.001, Initial learning rate
learnRateSchedule = 'none'; % 'none' for constant learning rate, 'piecewise' otherwise
learnRateDropFactor = 0.1; % default 0.1, [0, 1], factor for dropping the learning rate
l2Regularization = 0.0001; % default 0.0001, nonnegative scalar, factor for L2 regularization
momentum = 0.9; % default 0.9, [0, 1], contribution of previous step
% Other options for solver algorithms, hardware and checkpoints are available

rng('default') % for reproducibility

% DATASTORE CREATION
disp('Datastore creation...');
if (isempty(pathDataset) || ~isfolder(pathDataset)) % use MNIST as default
    disp("Dataset path doesn't exist. You will use MNIST!")
    pathDataset = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
end
datastore = imageDatastore(pathDataset, 'FileExtensions', fileExtensions, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainDatastore, validationDatastore] = splitEachLabel(datastore,trainingPercentage);

%histogram(trainDatastore.Labels); title('Training set label frequency');
%histogram(validationDatastore.Labels); title('Validation set label frequency');
if balanceDataset
    trainDatastore = balanceDatastore(trainDatastore);
    validationDatastore = balanceDatastore(validationDatastore);
    %histogram(trainDatastore.Labels); title('Training set label frequency after data balancing');
    %histogram(validationDatastore.Labels); title('Validation set label frequency after data balancing');
end

if dataAugmentation
    imageAugmenter = imageDataAugmenter(... % it allows to handle the image augmenter settings
        'RandXReflection', RandXReflection, ...
        'RandYReflection', RandYReflection, ...
        'RandRotation', RandRotation, ... % default [0 0]
        'RandScale', RandScale, ... % default [1 1]
        'RandXShear', RandXShear, ... % default [0 0]
        'RandYShear', RandYShear, ... % default [0 0]
        'RandXTranslation', RandXTranslation, ... % default [0 0]
        'RandYTranslation', RandYTranslation ... % default [0 0]
        );
    
    augmentedTrainDatastore = augmentedImageDatastore(imageSize, trainDatastore, 'DataAugmentation', imageAugmenter, ...
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
    
    augmentedValidationDatastore = augmentedImageDatastore(imageSize, validationDatastore, 'DataAugmentation', imageAugmenter, ...
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
else
    augmentedTrainDatastore = augmentedImageDatastore(imageSize, trainDatastore, 'DataAugmentation', 'none', ...
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
    
    augmentedValidationDatastore = augmentedImageDatastore(imageSize, validationDatastore, 'DataAugmentation', 'none', ...
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
end

%{
% display some of the training images
clf
imshow(imtile(preview(augmentedTrainDatastore).input));
%}

% NETWORK CREATION
disp('Network creation...');
inputSize = size(preview(augmentedTrainDatastore).input{1});
numClasses = numel(unique(datastore.Labels));

unet = unetLayers(inputSize, numClasses, 'EncoderDepth', encoderDepth, 'NumFirstEncoderFilters', numFirstEncoderFilters, 'FilterSize', filterSize, 'ConvolutionPadding', convolutionPadding);

% modify U-Net in order to perform classification instead of segmentation
unet = removeLayers(unet, {'Softmax-Layer','Segmentation-Layer'});
unet = addLayers(unet, [reluLayer('Name','Final-ReLU') fullyConnectedLayer(numClasses,'Name','Final-FullyConnected') softmaxLayer("Name","Softmax-Layer") classificationLayer("Name","Classification-Layer")]);
unet = connectLayers(unet,'Final-ConvolutionLayer','Final-ReLU');

% TRAINING
disp('Training...');
options = trainingOptions( ...
    solverName, ... % 'sgdm', 'rmsprop' or 'adam'
    'Plots', 'training-progress', ...
    'MaxEpochs', maxEpochs, ... % Maximum number of epochs
    'MiniBatchSize', miniBatchSize, ... % Size of mini-batch
    'Shuffle', shuffleMode, ... % 'once', 'never' or 'every-epoch'
    'ValidationData', augmentedValidationDatastore, ...
    'ValidationFrequency', validationFrequency, ... % Frequency of network validation
    'InitialLearnRate', initialLearnRate, ... % default 0.001, Initial learning rate
    'LearnRateSchedule', learnRateSchedule, ... % 'none' for constant learning rate, 'piecewise' otherwise
    'LearnRateDropFactor', learnRateDropFactor, ... % default 0.1, [0, 1], factor for dropping the learning rate
    'L2Regularization', l2Regularization, ... % default 0.0001, nonnegative scalar, factor for L2 regularization
    'Momentum', momentum ... % default 0.9, [0, 1], contribution of previous step
);
% Other options for solver algorithms, hardware and checkpoints are available

net = trainNetwork(augmentedTrainDatastore,unet,options);

y = classify(net, augmentedValidationDatastore);
plotconfusion(readall(augmentedValidationDatastore).response, y);