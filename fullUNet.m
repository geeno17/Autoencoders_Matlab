% U-NET

% DATASET SETTINGS

pathDataset = 'MerchData'; % path of the folder containing the images to be learnt, subdivided into folders per class
fileExtensions = {'.png'}; % image extensions to be accepted
trainingPercentage = 0.75; % data percentage for training set

datastore = imageDatastore(pathDataset, 'FileExtensions', fileExtensions, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainDatastore, validationDatastore] = splitEachLabel(datastore,trainingPercentage);

imageSize = [64 64]; % images output size, width and height must be a multiple of 2^encoderDepth
outputSizeMode = 'resize'; % 'resize', 'centercrop' or 'randcrop'
colorPreprocessing = 'none'; % 'none', 'gray2rgb' or 'rgb2gray'

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

augmentedValidationDatastore = augmentedImageDatastore(imageSize, validationDatastore, 'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', colorPreprocessing, 'OutputSizeMode', outputSizeMode);

%{
% display some of the training images
clf
imshow(imtile(preview(augmentedTrainDatastore).input));
%}

% NETWORK SETTINGS

encoderDepth = 4; % default 4, depth of the encoder/decoder structure in U-Net, i.e. number of images downsampling/upsampling
numFirstEncoderFilters = 32; % default 32, number of output channels for the first encoder stage, in each subsequent encoder stage the number of output channels doubles
filterSize = 3; % default 3, convolutional layer filter size (it can be also defined as [hieght width])
convolutionPadding = 'same'; % 'same' for zero padding applied to the inputs, 'valid' otherwise

inputSize = size(preview(augmentedTrainDatastore).input{1});
numClasses = numel(unique(datastore.Labels));

unet = unetLayers(inputSize, numClasses, 'EncoderDepth', encoderDepth, 'NumFirstEncoderFilters', numFirstEncoderFilters, 'FilterSize', filterSize, 'ConvolutionPadding', convolutionPadding);

% modify U-Net in order to perform classification instead of segmentation
unet = removeLayers(unet, {'Softmax-Layer','Segmentation-Layer'});
unet = addLayers(unet, [reluLayer('Name','Final-ReLU') fullyConnectedLayer(numClasses,'Name','Final-FullyConnected') softmaxLayer("Name","Softmax-Layer") classificationLayer("Name","Classification-Layer")]);
unet = connectLayers(unet,'Final-ConvolutionLayer','Final-ReLU');

% TRAINING

options = trainingOptions( ...
    'sgdm', ... % 'sgdm', 'rmsprop' or 'adam'
    'Plots', 'training-progress', ...
    'MaxEpochs', 30, ... % Maximum number of epochs
    'MiniBatchSize', 128, ... % Size of mini-batch
    'Shuffle', 'once', ... % 'once', 'never' or 'every-epoch'
    'ValidationData', augmentedValidationDatastore, ...
    'ValidationFrequency', 5, ... % Frequency of network validation
    'InitialLearnRate', 1e-3, ... % default 0.001, Initial learning rate
    'LearnRateSchedule', 'none', ... % 'none' for constant learning rate, 'piecewise' otherwise
    'LearnRateDropFactor', 0.1, ... % default 0.1, [0, 1], factor for dropping the learning rate
    'L2Regularization', 0.0001, ... % default 0.0001, nonnegative scalar, factor for L2 regularization
    'Momentum', 0.9 ... % default 0.9, [0, 1], contribution of previous step
);
% Other options for solver algorithms, hardware and checkpoints are available

net = trainNetwork(augmentedTrainDatastore,unet,options);

y = classify(net, augmentedValidationDatastore);
plotconfusion(readall(augmentedValidationDatastore).response, y);