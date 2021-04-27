% VANILLA AUTOENCODER

% DATASET SETTINGS

pathDataset = 'MerchData'; % path of the folder containing the images to be learnt, subdivided into folders per class
fileExtensions = {'.jpg'}; % image extensions to be accepted
trainingPercentage = 0.75; % data percentage for training set

datastore = imageDatastore(pathDataset, 'FileExtensions', fileExtensions, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainDatastore, testDatastore] = splitEachLabel(datastore,trainingPercentage);

dataAugmentation = true; % true/false if you need to perform data augmentation or data transformation (rgb2gray, resize etc)
if dataAugmentation
    nDataset = 3; % how many times you want to replicate the dataset
    imageSize = [50 50]; % images output size
    outputSizeMode = 'resize'; %'resize', 'centercrop' or 'randcrop'
    colorPreprocessing = 'rgb2gray'; % 'gray2rgb' or 'rgb2gray'
    
    imageAugmenter = imageDataAugmenter(... % it allows to handle the image augmenter settings 
        'RandXReflection', false, ...
        'RandYReflection', false, ...
        'RandRotation', [-15 15], ... % default [0 0]
        'RandScale', [0.8 1.2], ... % default [1 1]
        'RandXShear', [0 0], ... % default [0 0]
        'RandYShear', [0 0], ... % default [0 0]
        'RandXTranslation', [0 0], ... % default [0 0]
        'RandYTranslation', [0 0] ... % default [0 0]
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

% AUTOENCODER SETTINGS

layersSize = [100, 50]; % each element of layersSize refers to the size of the ith autoencoder layer
l2WeightRegularization = [0.004, 0.002]; % default 0.001, accepts positive values
sparsityRegularization = [4, 4]; % default 1, accepts positive values
sparsityProportion = [0.15, 0.1]; % default 0.05, accepts values between 0 and 1
maxEpochs = [400, 100];
scaleData = [false, false]; % default true, accepts boolean
%encoderTransferFunction = ['logsig','logsig']; % 'logsig' o 'satlin'
%decoderTransferFunction = ['logsig','logsig']; % 'logsig', 'satlin' o 'purelin'
%lossFunction = ['mseparse', 'mseparse', 'mseparse'];
%trainingAlgorithm = ['trainscg', 'trainscg', trainscg'];

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
    'MaxEpochs', 25 ... % epochs to train the last soft max layer
    );

network = stack(layers{:});

% display the weights of the ith autoencoder layer
%plotWeights(layers{i});

% display the full network
%view(network);

% FINE TUNING

xTrain = zeros(prod(imageSize), numel(trainImages));
for i = 1 : numel(trainImages)
    xTrain(:,i) = trainImages{i}(:);
end

% It is possible to modify the fine-tuning settings from network.trainParam

network = train(network, xTrain, trainLabels);

xTest = zeros(prod(imageSize), numel(testImages));
for i = 1 : numel(testImages)
    xTest(:,i) = testImages{i}(:);
end

y = network(xTest);
plotconfusion(testLabels,y);