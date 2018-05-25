%% RecogniseFace function test harness for still image library

%This script tests the RecogniseFace script with all JPG images in the selected
%image datastore.  
%True label (taken from image folder) and function results are saved in "Parray", 
%representing the P response for all images tested.  In addition a confusion matrix is displayed.
%
%Sample images folders are provided (and may be changed in line 13 below)
%Classifier parameters being tested can be changed in line 37 and 38.

%% Load Images
% Load Image Set From a Folder of Images
imageFolder = 'testimagesIV'; %testimagesGp testimagesNoface testimagesIV
imdsIn = imageDatastore(fullfile(imageFolder),'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
% Count each label and number of folders
tblIn = countEachLabel(imdsIn);
imgCountIn = sum(table2array(tblIn(:,2)));

%Load reference image data store for checking
imageFolder2 = 'facesref';
imdsTrue = imageDatastore(fullfile(imageFolder2),'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
tbl = countEachLabel(imdsTrue);
studentCount = size(tbl,1);
imgCount = sum(table2array(tbl(:,2)));

%Emotion list
emotionList = [')'; '('; 'o';'X'];
emotionCol = [{'blue'}; {'black'}; {'cyan'}; {'red'}];
%Close all Matlab image windows
close all;
    
%Parameters for selecting images
startnum = 1; %7 for 10x10 visualisation used in report 
nimages = 9; %imgCountIn for all test images, 100 for 10x10 visualisation, 9 for quick test
%nimages = 1; % for 1 image only
step = 5; %1 for all test images, 5 to cover all students with 10x10 visualisation
featureType = 'HOG'; %BAG, HOG, RAW(MLP only)
classifierName = 'MLP'; %CNN, CNX, MLP, SVM
verbose = false;
groupImage = strcmp('testimagesGp', imageFolder);

Parray=[];
imgArray = uint8(zeros(300,300,3,1));
counter=0;
for n = startnum:step:startnum+step*(nimages-1)
    counter=counter+1
    img = readimage(imdsIn,n);
    if verbose==true
        figure, imshow(imresize(img,0.15));
    end
    P = RecogniseFace(img,featureType, classifierName, verbose);
    if isempty(P), P = [0,0,0,0], end; %simpler to retain zero array for multiple results than '[]' 
    PLabelTrue = cellstr(imdsIn.Labels(n));
    Ptemp=ones(size(P,1),1)*n; Ptemp(:,2)=str2double(PLabelTrue); Ptemp(:,3:6)=P;  
    Parray = cat(1,Parray, Ptemp);
    
    if P(1,1)>0
        booleanIndex = str2double(cellstr(imdsTrue.Labels))==P(1,1);
        integerIndex = find(booleanIndex);
        trueImage = readimage(imdsTrue, integerIndex(1));
    else
        %if no image found
        trueImage = ones(100,100,3)*100;
    end
    
    %colour code red/green
    if P(1,1) == str2double(PLabelTrue)
        trueImage(:,:,2)=trueImage(:,:,2)*2;
    else
        trueImage(:,:,1)=trueImage(:,:,1)*2;
    end
    
    scaley = 300/size(img,1);    scalex = 300/size(img,2);
    imageOut = insertMarker(imresize(img,[300 300]),[P(1,2)*scalex,P(1,3)*scaley],'*', 'color', 'red', 'size', 50);
    leftPoint = ceil(size(imageOut,2)/2)-50;
    imageOut(end-99:end,leftPoint:leftPoint+99,:)=trueImage(:,:,:);
    imageOut=insertText(imageOut, [10,10], emotionList(P(1,4)+1), 'FontSize', 50, 'BoxColor', emotionCol(P(1,4)+1));
    imgArray(:,:,:,counter) = imageOut;
    
    if groupImage
        gpImageOut = img;
        for loop=1:size(P,1)
            if P(loop,1)>0
                booleanIndex = str2double(cellstr(imdsTrue.Labels))==P(loop,1);
                integerIndex = find(booleanIndex);
                trueImage = readimage(imdsTrue, integerIndex(1));
            else
                %if no image found
                trueImage = ones(100,100,3)*100;
            end
            trueImage=imresize(trueImage,[150 150]);

            x=max(221,ceil(P(loop,2)));y=max(100,ceil(P(loop,3)));
            gpImageOut(y-99:y+50,x-220:x-71,1)=trueImage(:,:,1);
            gpImageOut(y-99:y+50,x-220:x-71,2)=trueImage(:,:,1);
            gpImageOut=insertText(gpImageOut, [x-220,y-99], emotionList(P(loop,4)+1), 'FontSize', 40, 'BoxColor', emotionCol(P(loop,4)+1));

        end
        imshow(gpImageOut)
    end
    if mod(counter,10)==0
        close all
    end
end

Parray
if ~groupImage, montage(imgArray), end

%% Calculate and plot confusion matrix (note: multiple faces detected in one image counts as multiple items)
display('Confusion Matrix....')
C = confusionmat(Parray(:,2),Parray(:,3))
