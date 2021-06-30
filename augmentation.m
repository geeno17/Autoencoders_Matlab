function dataOut = augmentation(dataIn)
    global imageSize outputSizeMode colorPreprocessing RandXReflection RandYReflection RandRotation ...
        RandScale RandXShear RandYShear RandXTranslation RandYTranslation
    
    temp = dataIn;
    
    % Color preprocessing
    if strcmp(colorPreprocessing, 'rgb2gray')
        temp = rgb2gray(temp);
    end
    
    % X and Y reflection
    if RandXReflection && randi([0 1])
        temp = flip(temp, 1);
    end
    if RandYReflection && randi([0 1])
        temp = flip(temp, 2);
    end
    
    % Rotation
    temp = imrotate(temp, randi(RandRotation));
    
    % Scaling
    temp = imresize(temp, diff(RandScale)*rand() + RandScale(1));
    
    % Shearing
    temp = imwarp(temp, affine2d([1 0 0; diff(RandXShear)*rand() + RandXShear(1) 1 0; 0 0 1]));
    temp = imwarp(temp, affine2d([1 0 0; diff(RandYShear)*rand() + RandYShear(1) 1 0; 0 0 1]'));
    
    % Traslating
    temp = imtranslate(temp,[diff(RandXTranslation)*rand() + RandXTranslation(1), diff(RandYTranslation)*rand() + RandYTranslation(1)]);
    
    % Resizing
    if strcmp(outputSizeMode, 'resize')
        temp = imresize(temp, imageSize);
    elseif strcmp(outputSizeMode, 'centercrop')
        tempSize = size(temp);
        tempCenter = [fix(tempSize(1) / 2) fix(tempSize(2) / 2)];
        cropRectangle = [ ...
            (tempCenter(1) - fix(imageSize(1) / 2)) ...
            (tempCenter(2) - fix(imageSize(2) / 2)) ...
            (imageSize(1) - 1) ...
            (imageSize(2) - 1) ...
            ];
        temp = imcrop(temp, cropRectangle);
    elseif strcmp(outputSizeMode, 'randcrop')
        tempSize = size(temp);
        tempCenter = [ ...
            randi([fix(imageSize(1) / 2) tempSize(1) - fix(imageSize(1) / 2)]) ...
            randi([fix(imageSize(2) / 2) tempSize(2) - fix(imageSize(2) / 2)]) ...
            ];
        disp(tempCenter);
        cropRectangle = [ ...
            (tempCenter(1) - fix(imageSize(1) / 2)) ...
            (tempCenter(2) - fix(imageSize(2) / 2)) ...
            (imageSize(1) - 1) ...
            (imageSize(2) - 1) ...
            ];
        temp = imcrop(temp, cropRectangle);
    end
    
    dataOut = {temp};
    %dataOut = temp;
end

