% LOADING EXAMPLE DATA

[xTrainImages,tTrain] = digitTrainCellArrayData;
[xTestImages,tTest] = digitTestCellArrayData;

imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;

% 3Dtrain = cell(1,10);
% for i = 1 : 10
%    3Dtrain{i} = zeros(28,28,3);
%    3Dtrain{i}(:,:,1) = xTrainImages{1};
%    3Dtrain{i}(:,:,2) = xTrainImages{1};
%    3Dtrain{i}(:,:,3) = xTrainImages{1};    
% end
% xTrain = 3Dtrain;
% tTrain = tTrain(:,1:10);

rng('default');

% SETTING PARAMETERS

layersSize = [100, 50];
l2WeightRegularization = [0.004, 0.002]; %default 0.001, accepts positive values
sparsityRegularization = [4, 4]; %default 1, accepts positive values
sparsityProportion = [0.15, 0.1]; %default 0.05, accepts values between 0 and 1
maxEpochs = [400, 100];
%encoderTransferFunction = ['logsig','logsig']; %'logsig' o 'satlin'
%decoderTransferFunction = ['logsig','logsig']; %'logsig', 'satlin' o 'purelin'
scaleData = [false, false]; %default true, accepts boolean
%lossFunction = ['mseparse', 'mseparse', 'mseparse'];
%trainingAlgorithm = ['trainscg', 'trainscg', trainscg'];

% TRAINING

layers = cell(1, numel(layersSize) + 1);
layerInput = xTrainImages;

for i = 1 : numel(layersSize)
    layers{i} = trainAutoencoder(layerInput, layersSize(i), ...
        'L2WeightRegularization', l2WeightRegularization(i), ...
        'SparsityRegularization', sparsityRegularization(i), ...
        'SparsityProportion', sparsityProportion(i), ...
        'MaxEpochs', maxEpochs(i), ...
        'ScaleData', scaleData(i));
    layerInput = encode(layers{i}, layerInput);
end
layers{end} = trainSoftmaxLayer(layerInput,tTrain, 'MaxEpochs', 25);

network = stack(layers{:});

% FINE-TUNING

xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

%E' possibile modificare le impostazioni di fine-tuning da network.trainParam

network = train(network,xTrain,tTrain);

[xTestImages,tTest] = digitTestCellArrayData;
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end

y = network(xTest);
plotconfusion(tTest,y);