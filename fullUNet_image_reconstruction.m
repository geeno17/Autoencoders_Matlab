% U-NET
clear; clc; close all

% DATASET SETTINGS
%pathDataset = 'D:\Documenti\Matlab_Workspace\datasets\shape_dataset_reduced'; % path of the folder containing the images to be learnt, subdivided into folders per class
pathDataset = '/home/gaetano/Matlab_Workspace/Datasets/shape_dataset'; % path of the folder containing the images to be learnt, subdivided into folders per class
fileExtensions = {'.png'}; % image extensions to be accepted
trainingPercentage = 0.7; % data percentage for training set
valPercentage = 0.15; % data percentage for validation set
testPercentage = 0.15; % data percentage for test set
balanceDataset = true; % true/false if your dataset is unbalanced you can balance it
trainDesiredNumObservation = "max"; % max, min, mean, median, or you can specify a number. It is the number of samples per class that you want in the balanced dataSet
validationDesiredNumObservation = "max";
testDesiredNumObservation = "max";

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
encoderDepth = 3; % default 4, depth of the encoder/decoder structure in U-Net, i.e. number of images downsampling/upsampling
numFirstEncoderFilters = 16; % default 32, number of output channels for the first encoder stage, in each subsequent encoder stage the number of output channels doubles
filterSize = 7; % default 3, convolutional layer filter size (it can be also defined as [hieght width])
convolutionPadding = 'same'; % 'same' for zero padding applied to the inputs, 'valid' otherwise
numOutputChannels = 3; %number of output channels (for image reconstruction must be equal to input channels)

% TRAINING SETTINGS
solverName = 'sgdm'; % 'sgdm', 'rmsprop' or 'adam', solver for training network
maxEpochs = 30; % Maximum number of epochs
miniBatchSize = 128; % Size of mini-batch
shuffleMode = 'every-epoch'; % 'once', 'never' or 'every-epoch', option for data shufflind during training/validation
validationFrequency = 5; % Frequency of network validation
initialLearnRate = 1e-1; % default 0.001, Initial learning rate
learnRateSchedule = 'piecewise'; % 'none' for constant learning rate, 'piecewise' otherwise
learnRateDropFactor = 0.1; % default 0.1, [0, 1], factor for dropping the learning rate
l2Regularization = 0.0001; % default 0.0001, nonnegative scalar, factor for L2 regularization
momentum = 0.9; % default 0.9, [0, 1], contribution of previous step
learnRateDropPeriod = 10;
gradientThresholdMethod = 'l2norm';
gradientThreshold = 0.01;

% Other options for solver algorithms, hardware and checkpoints are available
useGPU = true;

rng('default') % for reproducibility

% DATASTORE CREATION
disp('Datastore creation...');
if (isempty(pathDataset) || ~isfolder(pathDataset)) % use MNIST as default
    disp("Dataset path doesn't exist. You will use MNIST!")
    pathDataset = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
end

datastore = imageDatastore(pathDataset, 'FileExtensions', fileExtensions, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

datastore = shuffle(datastore);

[trainDatastore, validationDatastore, testDatastore] = splitEachLabel(datastore,trainingPercentage, valPercentage, testPercentage);

%histogram(trainDatastore.Labels); title('Training set label frequency');
%histogram(validationDatastore.Labels); title('Validation set label frequency');
if balanceDataset
    trainDatastore = balanceDatastore(trainDatastore, trainDesiredNumObservation);
    validationDatastore = balanceDatastore(validationDatastore, validationDesiredNumObservation);
    testDatastore = balanceDatastore(testDatastore, testDesiredNumObservation);
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
    
    augmentedTestDatastore = augmentedImageDatastore(imageSize, testDatastore, 'DataAugmentation', imageAugmenter, ...
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
else
    augmentedTrainDatastore = augmentedImageDatastore(imageSize, trainDatastore, 'DataAugmentation', 'none', ...
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
    
    augmentedValidationDatastore = augmentedImageDatastore(imageSize, validationDatastore, 'DataAugmentation', 'none', ...
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
    
     augmentedTestDatastore = augmentedImageDatastore(imageSize, testDatastore, 'DataAugmentation', 'none', ...
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
end

%augmentedTrainDatastore = randomPatchExtractionDatastore(trainDatastore, trainDatastore, 64, ....
%'PatchesPerImage',1);
%augmentedValidationDatastore = randomPatchExtractionDatastore(validationDatastore, validationDatastore, 64, ....
%'PatchesPerImage',1);


augmentedTrainDatastore = transform(trainDatastore,@(x) resize(x, imageSize));
augmentedValidationDatastore = transform(validationDatastore, @(x) resize(x, imageSize));
augmentedTestDatastore = transform(testDatastore, @(x) resize(x, imageSize));

%responseTrainDatastore = transform(trainDatastore,@(x) normalize(x, imageSize));
%responseValidationDatastore = transform(validationDatastore, @(x) normalize(x, imageSize));
%responseTestDatastore = transform(testDatastore, @(x) normalize(x, imageSize));

augmentedTrainDatastore = combine(augmentedTrainDatastore,augmentedTrainDatastore);
augmentedValidationDatastore = combine(augmentedValidationDatastore,augmentedValidationDatastore);
augmentedTestDatastore = combine(augmentedTestDatastore,augmentedTestDatastore);


%{
% display some of the training images
clf
imshow(imtile(preview(augmentedTrainDatastore).input));
%}

% NETWORK CREATION
disp('Network creation...');
example_img = read(augmentedTrainDatastore);
inputSize = size(example_img{1,1});
%inputSize = [64,64,3];

unet = unetLayers(inputSize, 3, 'EncoderDepth', encoderDepth, 'NumFirstEncoderFilters', numFirstEncoderFilters, 'FilterSize', filterSize, 'ConvolutionPadding', convolutionPadding);
%unet = unetLayers(inputSize, 3, 'EncoderDepth', encoderDepth);
% modify U-Net in order to perform classification instead of segmentation
unet = removeLayers(unet, {'Final-ConvolutionLayer','Softmax-Layer','Segmentation-Layer'});
lastLayer = unet.Layers(end).Name;
unet = addLayers(unet, [convolution2dLayer([1,1], numOutputChannels, "Name","Final-ConvolutionLayer", "Padding", "same", "WeightsInitializer", "he") regressionLayer("Name","Regression-Layer")]);
unet = connectLayers(unet,lastLayer, 'Final-ConvolutionLayer');

%unet = unet.removeLayers({'Softmax-Layer','Segmentation-Layer'});
%unet = unet.addLayers(regressionLayer("Name","Regression-Layer"));
%unet = unet.connectLayers('Final-ConvolutionLayer','Regression-Layer');


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
    'Momentum', momentum, ... % default 0.9, [0, 1], contribution of previous step    
    'LearnRateSchedule', learnRateSchedule, ...
    'LearnRateDropPeriod', learnRateDropPeriod, ...
    'GradientThresholdMethod', gradientThresholdMethod, ...
    'GradientThreshold', gradientThreshold, ...
    'CheckpointPath', './checkpoints', ...
    'ExecutionEnvironment', 'auto' ...
);
% Other options for solver algorithms, hardware and checkpoints are available

net = trainNetwork(augmentedTrainDatastore, unet, options);
unet0 = net;
save unet0;

clear; clc; close all

%y = classify(net, augmentedValidationDatastore);
%plotconfusion(readall(augmentedValidationDatastore).response, y);


function dataOut = resize(dataIn, imageSize)
        temp = dataIn;
        temp = rescale(temp);
        dataOut = {imresize(temp,imageSize,'method','bilinear')};
end

function dataOut = normalize(dataIn, imageSize)
        temp = dataIn;
        temp = {imresize(temp,imageSize,'method','bilinear')};
        %dataOut = {(temp - 127.5)/127.5};
        dataOut = {rescale(temp, -1, 1)}
end