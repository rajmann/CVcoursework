%% Detect function test harness for video library

%This script tests the detectnum script with all .mov videofiles in the selected
%image datastore, assuming these are filed with folder name = true label.  
%True label and detectnum results are saved in "ocrArray".
%To simplify review of results, the majority vote and structure array responses 
%are stored in ocrArray separately.

%% Extract OCR labels from input images
global textBoxes;
global cleanBoxes;
global bboxes;
global trainImage;
global ocrResults;
global mserRegions;
global ocrWords;

    
%Create an Image Set From a Folder of Images
imageFolder = 'testimagesVFull';
imds = imageDatastore(fullfile(imageFolder),'IncludeSubfolders',true,'FileExtensions','.mov','LabelSource','foldernames');
% Count each label and number of folders
tbl = countEachLabel(imds);
studentCount = size(tbl,1);
imgCount = sum(table2array(tbl(:,2)));

%Close all Matlab image windows
close all;

%Define which pictures to analyse
startindex=1;
endindex=imgCount;
endindex=2;  %for limiting test run

%Run through images and store results in ocrResults array
ocrArray = [];
line = 0;
for imnum= startindex:endindex;
    imnum
    close all
    filepath = char(imds.Files(imnum));
    v=VideoReader(filepath);
    output = detectnum(v, 0);
    ocrArray(imnum).true = char(imds.Labels(imnum));
    ocrArray(imnum).true2 = ocrArray(imnum).true(end-1:end);
    ocrArray(imnum).ocrMajority = output(1);
    ocrArray(imnum).ocrFrameByFrame = output{2};
end

display('To view results, open ocrArray structure array in Workspace')