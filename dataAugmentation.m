datastore = imageDatastore('MerchData', 'FileExtensions', {'.jpg'}, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

imageAugmenter = imageDataAugmenter(...
    'RandXReflection', false, ...
    'RandYReflection', false, ...
    'RandRotation', [-45 45], ... %default [0 0]
    'RandScale', [0.8 1.2], ... %default [1 1]
    'RandXShear', [0 45], ... %default [0 0]
    'RandYShear', [0 45], ... %default [0 0]
    'RandXTranslation', [-5 5], ... %default [0 0]
    'RandYTranslation', [-5 5] ... %default [0 0]
    );

augmentedDatastore = augmentedImageDatastore([50 50], datastore, ...
    'DataAugmentation', imageAugmenter, ... %default 'none'
    'ColorPreprocessing', 'none', ... %'gray2rgb' or 'rgb2gray'
    'OutputSizeMode', 'randcrop' ... %'resize' or 'centercrop'
    );

p = preview(augmentedDatastore);
imshow(imtile(p.input));


%{
%per ottenere un dataset aumentato n volte
n = 3;

aug = table('Size', [n * augmentedDatastore.NumObservations, 2], 'VariableNames', {'input', 'response'}, 'VariableTypes', {'cell', 'categorical'});

for i = 1 : augmentedDatastore.NumObservations : n * augmentedDatastore.NumObservations
    aug(i : i + augmentedDatastore.NumObservations - 1, :) = readall(augmentedDatastore);
end

trainImages = aug.input';
trainLabels = dummyvar(aug.response)';
%}

%{
[trainDatastore, testDatastore] = splitEachLabel(datastore,0.75);

trainImages = readall(trainDatastore)';
trainLabels = dummyvar(trainDatastore.Labels)';

testImages = readall(testDatastore)';
testLabels = dummyvar(testDatastore.Labels)';
%}
    

