%% Detect function test harness for still image library

%This script tests the detectnum script with all JPG images in the selected
%image datastore, assuming these are filed with folder name = true label.  
%True label and detectnum results are saved in "ocrArray".  In addition the 
%first such result per image is stored as a separate column in ocrArray

%% Extract OCR labels from input images
global cleanBoxes;
global ocrResults;
global bboxes;
global trainImage;
global mserRegions;
global ocrWords;
    
%Create an Image Set From a Folder of Images
imageFolder = 'testimagesIV'; %select from 'testimagesIV' (raw jpgs + extracts from videos)
    %'testimagesGp' (group pictures) or 'testimagesMultiOCR' (2 or 3 OCR per image)
imds = imageDatastore(fullfile(imageFolder),'IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
% Count each label and number of folders
tbl = countEachLabel(imds);
studentCount = size(tbl,1);
imgCount = sum(table2array(tbl(:,2)));

%Close all Matlab image windows
close all;

%Define which pictures to analyse
startindex=1;
endindex=5; %change to '=imgCount' for all images in folder
%endindex=startindex;  %for limiting test run to one image

%Run through images and store results in ocrResults array
ocrArray = [];
for imnum= startindex:endindex;
    imnum
    image=readimage(imds,imnum);
    ocrArray(imnum).true = char(imds.Labels(imnum));
    ocrArray(imnum).true2 = ocrArray(imnum).true(end-1:end);
    ocrArray(imnum).allresults = detectnum(image,3); %, (image, 1) to show boxes, (image,2) for MSER 
    ocrArray(imnum).firstresult = ocrArray(imnum).allresults(1);

end

vertcat(ocrArray.firstresult)
display('To view results, open ocrArray structure array in Workspace')
