% VANILLA AUTOENCODER

clear;clc;close all
% DATASET SETTINGS

pathDataset = 'MerchData'; % path of the folder containing the images to be learnt, subdivided into folders per class. If empty you will use MNIST
fileExtensions = {'.png'}; % image extensions to be accepted
trainingPercentage = 0.75; % data percentage for training set
balanceDataset = true; %true/false if your dataset is unbalanced you can balance it.
trainDesiredNumObservation = "max" % max, min, mean, median, or you can specify a number. It is the number of samples per class that you want in the balanced dataSet
testDesiredNumObservation = "max"

%DATA AUGMENTATION SETTINGS 
dataAugmentation = false; % true/false if you need to perform data augmentation or data transformation (rgb2gray, resize etc). If false, you can ignore next settings (DATA AUGMENTATION SETTINGS)
imageSize = [50 50]; % images output size
nDataset = 3; % how many times you want to replicate the dataset
outputSizeMode = 'resize'; %'resize', 'centercrop' or 'randcrop'
colorPreprocessing = 'rgb2gray'; % 'gray2rgb' or 'rgb2gray'

RandXReflection = false;
RandYReflection = false;
RandRotation = [-15 15]; % default [0 0]
RandScale = [0.8 1.2]; % default [1 1]
RandXShear = [0 0]; % default [0 0]
RandYShear = [0 0]; % default [0 0]
RandXTranslation = [0 0]; % default [0 0]
RandYTranslation = [0 0];  % default [0 0]

% AUTOENCODER SETTINGS
layersSize = [100, 50]; % each element of layersSize refers to the size of the ith autoencoder layer
l2WeightRegularization = [0.004, 0.002]; % default 0.001, accepts positive values
sparsityRegularization = [4, 4]; % default 1, accepts positive values
sparsityProportion = [0.15, 0.1]; % default 0.05, accepts values between 0 and 1
maxEpochs = [400, 100, 25]; % max number of epochs to train every layer (the last value refers to Softmax layer)
scaleData = [false, false]; % default true, accepts boolean
%encoderTransferFunction = ['logsig','logsig']; % 'logsig' o 'satlin'
%decoderTransferFunction = ['logsig','logsig']; % 'logsig', 'satlin' o 'purelin'
%lossFunction = ['mseparse', 'mseparse', 'mseparse'];
%trainingAlgorithm = ['trainscg', 'trainscg', trainscg'];

rng('default') % for reproducibility

%%CREATE DATASTORE
if (isempty(pathDataset) || ~isfolder(pathDataset)) % use MNIST as default
    disp("Dataset path doesn't exist. You will use MNIST!")
    pathDataset = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
end
datastore = imageDatastore(pathDataset, 'FileExtensions', fileExtensions, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainDatastore, testDatastore] = splitEachLabel(datastore, trainingPercentage);


%histogram(trainDatastore.Labels); title('Training set label frequency');
%histogram(testDatastore.Labels); title('Test set label frequency');
if balanceDataset
    trainDatastore = balanceDatastore(trainDatastore, trainDesiredNumObservation);
    testDatastore = balanceDatastore(testDatastore, testDesiredNumObservation);
    
    %histogram(trainDatastore.Labels); title('Training set label frequency after data balancing');
    %histogram(testDatastore.Labels); title('Test set label frequency after data balancing');
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

    augmentedTestDatastore = augmentedImageDatastore(imageSize, testDatastore, 'DataAugmentation', imageAugmenter, ... 
        'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);
    
    aug = table('Size', [nDataset * augmentedTrainDatastore.NumObservations, 2], 'VariableNames', {'input', 'response'}, 'VariableTypes', {'cell', 'categorical'});
    for i = 1 : augmentedTrainDatastore.NumObservations : nDataset * augmentedTrainDatastore.NumObservations
        aug(i : i + augmentedTrainDatastore.NumObservations - 1, :) = readall(augmentedTrainDatastore);
    end
    trainImages = aug.input';
    trainLabels = dummyvar(aug.response)';
    
    aug = table('Size', [nDataset * augmentedTestDatastore.NumObservations, 2], 'VariableNames', {'input', 'response'}, 'VariableTypes', {'cell', 'categorical'});
    for i = 1 : augmentedTestDatastore.NumObservations : nDataset * augmentedTestDatastore.NumObservations
        aug(i : i + augmentedTestDatastore.NumObservations - 1, :) = readall(augmentedTestDatastore);
    end
    testImages = aug.input';
    testLabels = dummyvar(aug.response)';
else
    trainImages = readall(trainDatastore)';
    trainLabels = dummyvar(trainDatastore.Labels)';

    testImages = readall(testDatastore)';
    testLabels = dummyvar(testDatastore.Labels)'
end



%{
% display some of the training images
clf
for i = 1:20
    subplot(4,5,i);
    imshow(trainImages{i});
end
%}

% TRAINING

layers = cell(1, numel(layersSize) + 1);
layerInput = trainImages;

for i = 1 : numel(layersSize)
    layers{i} = trainAutoencoder(layerInput, layersSize(i), ...
        'L2WeightRegularization', l2WeightRegularization(i), ...
        'SparsityRegularization', sparsityRegularization(i), ...
        'SparsityProportion', sparsityProportion(i), ...
        'MaxEpochs', maxEpochs(i), ...
        'ScaleData', scaleData(i));
    layerInput = encode(layers{i}, layerInput);
end
layers{end} = trainSoftmaxLayer(layerInput,trainLabels, ...
    'MaxEpochs', maxEpochs{end} ... % epochs to train the last soft max layer
    );

network = stack(layers{:});

% FINE TUNING

xTrain = zeros(prod(imageSize), numel(trainImages));
for i = 1 : numel(trainImages)
    xTrain(:,i) = trainImages{i}(:);
end

% It is possible to modify the fine-tuning settings from network.trainParam

network = train(network, xTrain, trainLabels);

% display the weights of the ith autoencoder layer
%plotWeights(layers{i});

% display the full network
%view(network);

xTest = zeros(prod(imageSize), numel(testImages));
for i = 1 : numel(testImages)
    xTest(:,i) = testImages{i}(:);
end

y = network(xTest);
plotconfusion(testLabels, y);