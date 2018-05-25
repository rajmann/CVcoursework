%% RecogniseFace - intro

%RecogniseFace returns a matrix P representing the people present in an RGB image I. 
%P is a matrix of size Nx4, where N is the number of people detected in the number of image. 
%The first three columns represent:
% 1. id, a unique number for each person that matches the training data
% 2. x, the x location of the person detected in the image (central face region)
% 3. y, the y location of the person detected in the image (central face region)

%The fourth column contains an emotion classification for 
%each image (happy = 0; sad = 1; surprised = 2; angry = 3)

% The function accepts three input parameters:
% 1. Input Image - an RGB image of any size
% 2. featureType - HOG (for HOG features), BAG (for bag of SURF features).  Not applicable for CNN 
% 3. classifierName - CNN (convolutional NN), SVM or MLP (non-convolutional NN)

%The function also accepts an optional fourth parameter to display diagnostics:
%'verbose' = 0(default - off), 1(display progress)

%% RecogniseFace - file structure

%This file consists of functions as follows.  

% 1. RecogniseFace - manage workflow through face detection and recognition
%for different input parameters
% 2. DetectFace - find bounding boxes of faces in image
% 3. PreProcess - Pre-process face images for classifying
% 4. ClassifyFace - extract features and classify face identity against selected pre-trained classifiers
% 5. ClassifyEmotion - classify face emotion against pre-trained emotion classifier
% 6. DataCleansing - final data preparation


%% Function 1:   1. RecogniseFace - manage workflow through face detection and recognition

function [P] = RecogniseFace(I, featureType, classifierName, verbose)

    %default for 'verbose' options is zero (no diagnostics)
    if ~exist('verbose','var')
        verbose=0;
    end
    
    %check valid function parameters
    if ~any(strcmp({'CNN', 'CNX', 'SVM', 'MLP'},classifierName)) | ~any(strcmp({'RAW', 'HOG', 'BAG'},featureType)) | strcmp('RAWSVM', strcat(featureType,classifierName))
        error('invalid function parameters featureType or classiferName pair')
        return
    end
    
    P=[];
    bbox=[];
    mergeLevel = 6;

    %try different face detectors and increase sensitivity until a valid face match found
    while size(P,1)==0 & mergeLevel>2
        faceModelAttempt = 1;
        while size(P,1)==0 & faceModelAttempt<=3
            if faceModelAttempt==1
                [bbox, xy, faceArrayGray] = DetectFace(I, mergeLevel, 'FrontalFaceCART', verbose); %detect faces and resize             %if none found, look for profile faces
            elseif faceModelAttempt==2
                [bbox, xy, faceArrayGray] = DetectFace(I, mergeLevel, 'ProfileFace', verbose); %detect faces and resize
            %final attempt, upperBody search and crop 
            elseif faceModelAttempt==3 & mergeLevel<6
                if verbose, display('Upper Body Detection'), end;
                [bbox, xy, faceArrayGray] = DetectFace(I, mergeLevel, 'UpperBody', verbose); %detect faces and resize 
            end
    
            %only continue processing if face found
            if size(bbox,1)>0     
                faceArrayMask = PreProcess(faceArrayGray, verbose); %apply pre-processing mask
                personPredict = ClassifyFace(faceArrayMask, featureType, classifierName, verbose)';
                emotionPredict = zeros(size(personPredict));
                emotionPredict = ClassifyEmotion(faceArrayGray, verbose);
                [P,faceArrayClean] = DataCleansing(personPredict, xy, emotionPredict, faceArrayGray, verbose);

            end
            faceModelAttempt=faceModelAttempt+1;
        end
        mergeLevel = mergeLevel-1;
    end
        
    %if no faces in clean data, return nul result 
    if size(P,1)==0
        %if no faces found
        P=[];
        faceArrayClean=[]; %for diagnostics
    end
    
    if verbose>0
        figure, montage(faceArrayClean),title('All faces found - final phase')
    end

end

%% Function 2:   DetectFace - find bounding boxes of faces in image
function [bbox, xy, faceArrayGray] = DetectFace(I, mergeLevel, faceModel, verbose)
    
    % check size of image
    imageSize = size(I); 

    % Create a cascade detector object
    faceDetector = vision.CascadeObjectDetector(faceModel);
 
    % Set detector parameters (default 40-400 range but can scale with image)
    minscale = max(22, ceil(imageSize(1)/100)); %model trained on 20x20 images
    maxscale = max(800, ceil(imageSize(1)/5));
    faceDetector.MinSize = [minscale minscale];
    faceDetector.MaxSize = [maxscale maxscale];
    faceDetector.MergeThreshold = mergeLevel;

    % Run detector
    bbox = step(faceDetector, I);
    faceArrayGray=uint8(zeros(100,100,3,size(bbox,1)));
    % Crop faces from image and convert to gray
    for j=size(bbox,1):-1:1 %step backwards so deleting box doesn't change index
        xbox=bbox(j,:);
        if strcmp(faceModel,'UpperBody')
            x=xbox(1);y=xbox(2);w=xbox(3);h=xbox(4);
            xnew=x+w*0.3;ynew=y+h*0.2;wnew=w*0.4;hnew=h*0.45;
            xbox=[xnew,ynew,wnew,hnew];
            bbox(j,:)=xbox;
        end
        face = imcrop(I, xbox); %crop to bounding box
        if strcmp(faceModel,'UpperBody')
            %Check if eyes included in upper body
            faceDetector = vision.CascadeObjectDetector('EyePairSmall');
            % Set detector parameters (default 40-400 range but can scale with image)
            maxscale = max(800, ceil(imageSize(1)/5));
            faceDetector.MinSize = [5 22];
            faceDetector.MaxSize = [maxscale maxscale];
            faceDetector.MergeThreshold = mergeLevel;
            bboxmini = step(faceDetector, face);
            % if no eyes found, delete bbox
            if size(bboxmini,1)==0
                bbox(j,:)=[];
            end
        end
        face = imresize(face,[100 100]); %size 100,100
        face(:,:,1) = rgb2gray(face); %convert to gray
        face(:,:,2) = rgb2gray(face); %convert to gray
        face(:,:,3) = rgb2gray(face); %convert to gray
        faceArrayGray(:,:,:,j) = face; %store in results array
    end

    xy = [bbox(:,1)+bbox(:,3)/2, bbox(:,2)+bbox(:,4)/2];
    
    %Optional diagnostics
    if verbose>0
        % Display region on image.
        imageOut = insertObjectAnnotation(I,'rectangle',bbox,'Face');
        imageOut = insertMarker(imageOut,xy,'*', 'color', 'red', 'size', 200);
        figure, imshow(imresize(imageOut,min(1,500/size(imageOut,1)))), title('Detected face');
        figure, montage(faceArrayGray),title('All faces found - first phase')
    end
end

%% Function 3. Pre-process face images for classifying
function [faceArrayMask] = PreProcess(faceArrayGray, verbose)

    %Create mask to remove left/right extremes and v-shape neck line
    cornerMask = ~(flip(tril(ones(100,100),65).*rot90(tril(ones(100,100),65)))); %cut corners
    cornerMask(:,1:15,:)=1; %cut left side
    cornerMask(:,86:end,:)=1; %cut right side

    %Loop through faces and apply mask to image image
    nBoxes = size(faceArrayGray,4);
    faceArrayMask = uint8(zeros(100,100,3,nBoxes));
    for j=1:nBoxes
        j=j;
        face = faceArrayGray(:,:,:,j);
        face = imoverlay(face,cornerMask,'black'); %apply mask    
        faceArrayMask(:,:,:,j)=face;
        
    end
    
    %Optional diagnostics
    if verbose>0
        figure,    montage(faceArrayMask),    title('Masked faces from group')
    end
end

%% Function 4. ClassifyFace - classify face identity against selected pre-trained classifiers
function [personPredict] = ClassifyFace(faceArray, featureType, classifierName, verbose)
    
    %load appropriate model
    if strcmp(classifierName,'CNN') 
        load('Models/faces5CNN4x4x32', 'convnet')
    elseif strcmp(classifierName,'CNX') 
        load('Models/faces5CNX4x4x32', 'convnet227')
    elseif strcmp(classifierName,'MLP') & strcmp(featureType,'HOG')
        load('Models/faces5MLPHOG68', 'mlpnet')
    elseif strcmp(classifierName,'MLP') & strcmp(featureType,'BAG')
        load('Models/faces5MLPBAG10008', 'mlpnet')
        load('Models/faces5MLPBAG10008bag', 'bag')

    elseif strcmp(classifierName,'MLP') & strcmp(featureType,'RAW')
        load('Models/faces5MLPRAW150', 'mlpnet')
    
    elseif strcmp(classifierName,'SVM') & strcmp(featureType,'HOG')
        faceClassifier = loadCompactModel('Models/face5SVMHOG988');
    elseif strcmp(classifierName,'SVM') & strcmp(featureType,'BAG')
        load('Models/faces5SVMBAG100010', 'faceClassifierBag')
        %load('Models/faces5SVMBAG100010bag', 'bag')%note:SVM BAG classifier takes image input not features
    end

    %Extract features for dataset
    if strcmp(featureType,'HOG')
        %Note: different HOG paramaters chosen for optimization of MLP v SVM
        if strcmp(classifierName, 'MLP')
            testFeatures=extractFeaturesHOG(faceArray, 6,[8 8], verbose);
            testFeatures=permute(reshape(testFeatures, size(testFeatures,1), 1, 1, size(testFeatures,2)),[4,3,2,1]);

        elseif strcmp(classifierName, 'SVM')
            testFeatures=extractFeaturesHOG(faceArray, 9,[8 8], verbose);
        end
    
    elseif strcmp(featureType,'BAG') & strcmp(classifierName, 'MLP') %note:SVM BAG classifier takes image input not features
        testFeatures=extractFeaturesBAG(faceArray, bag, verbose);
        %reshape to fit first layer of MLP which expects an image
        testFeatures=permute(reshape(testFeatures, size(testFeatures,1), 1, 1, size(testFeatures,2)),[4,3,2,1]);

    elseif strcmp(classifierName, 'CNN') & strcmp(classifierName, 'CNX')  
        testFeatures=faceArray; %CNN takes raw image data

    else
        testFeatures=reshape(faceArray, 100*100*3, 1, 1, size(faceArray,4)); %RAW model takes image input as 30000x1 array
        
    end
    
    %Loop through images and classify faces
    nBoxes = size(faceArray,4);
    for i=1:nBoxes
        if strcmp(classifierName,'CNN') 
            personPredict(i) = cellstr(classify(convnet,faceArray(:,:,:,i)));
        elseif strcmp(classifierName,'CNX') 
            face227 = imresize(faceArray(:,:,:,i), [227 227]); %resize to 227x227 Alex net input
            personPredict(i) = cellstr(classify(convnet227,face227));
        elseif strcmp(classifierName,'MLP') 
            if strcmp(featureType, 'HOG')
                personPredict(i) = cellstr(classify(mlpnet,testFeatures(:,:,:,i)));
            elseif strcmp(featureType, 'BAG')
                personPredict(i) = cellstr(classify(mlpnet,testFeatures(:,:,:,i)));
            elseif strcmp(featureType, 'RAW')
                personPredict(i) = cellstr(classify(mlpnet,testFeatures(:,:,:,i)));
            end
        elseif strcmp(classifierName,'SVM')
            if strcmp(featureType, 'HOG')
                personPredict(i) = cellstr(predict(faceClassifier,testFeatures(i,:,:,:)));
            elseif strcmp(featureType, 'BAG')
                label=faceClassifierBag.Labels(predict(faceClassifierBag,faceArray(:,:,:,i))); 
                personPredict(i) = cellstr(label); %note: SVM BAG takes image not feature input
            end
        end
    end
    
    %Optional diagnostics
    if verbose>0
        personPredict
    end
end

%% Function 5. ClassifyEmotion - classify face emotion against pre-trained emotion classifier

function [emotionPredict] = ClassifyEmotion(faceArray, verbose)
    
    %load appropriate model
    load('Models/emotionCNN100j', 'convnetE')
    
    %Loop through images and classify faces
    nBoxes = size(faceArray,4);
    for i=1:nBoxes
        img = rgb2gray(faceArray(:,:,:,i));
        emotionlabel = char(classify(convnetE,img)); %eg '0Happy'
        emotionPredict(i,1) = cellstr(emotionlabel(1)); %eg '0' only
    end
    
    %Optional diagnostics
    if verbose>0
        emotionlabel, emotionPredict
    end
end

%% Function 6. DataCleansing - final data preparation

function [P, faceArrayClean] = DataCleansing(personPredict, xy, emotionPredict, faceArrayGray, verbose);

    %Combine face predictions with xy coordinates converted from strings to
    %characters and first character of emotion response as char
    pxye = cat(2,str2double(personPredict),xy, str2double(emotionPredict));

    %Find false positives (category = 999) from result arrays R&E
    idx = ~strcmp(personPredict,'999');

    %Remove false positives from results
    P = pxye(idx,:);
    faceArrayClean = faceArrayGray(:,:,:,idx);
    
    %Optional diagnostics
    if verbose>0
        P=P
    end
end

%% Function 8. Feature Extraction - HOG

function [extractedFeatures] = extractFeaturesHOG(faceArray, nBins, cellSize, verbose);
    [hog, vis] = extractHOGFeatures(zeros(100,100,3),'CellSize',cellSize, 'NumBins', nBins);
    featureSize = length(hog);

    %Optional diagnostics
    if verbose>0
        display('HOG features for test data');
    end

    %Feature extraction
    extractedFeatures = zeros(size(faceArray,4), featureSize, 'single');
    for i=1:size(faceArray,4)
        img = faceArray(:,:,:,i);
        extractedFeatures(i,:) = extractHOGFeatures(img, 'CellSize', cellSize, 'NumBins', nBins);
    end
    
end

%% Function 8. Feature Extraction - Bag of Features

function [extractedFeatures] = extractFeaturesBAG(faceArray, bag, verbose);
   %Optional diagnostics
    if verbose>0
        display('Bag of features for test data');
    end
    %Feature extraction
    featureSize = bag.VocabularySize;
    extractedFeatures = zeros(size(faceArray,4), featureSize, 'single');
    for i=1:size(faceArray,4)
        img = faceArray(:,:,:,i);
        extractedFeatures(i,:) = encode(bag, img);
    end   
end