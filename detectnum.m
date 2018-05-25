%% Detect function - intro

%Detectnum returns text of OCR found in an image or video input as cell array.
%If multiple numbers are detected, these are returned as rows of the cell array 
%An optional second input ('verbose') is accepted for visualisation of the steps within the function: verbose is 0=off, 1=box or 2=box and MSER
%or 3 for intermediate stages as well
%returns 'Z' if no mserRegions found
%returns 'X' if mserRegions found but no text in white backgrounds 

%% Detect function - file structure

%This file consists of 2 functions as follows.  

%1. Detectnum 
%This function deals with the difference between images and video inputs.
%Videos are broken into a number of frames and then processed as individual
%images. Images (whether from video or direct) are then passed to the detectnumI
%function.

%2. DetectnumI
%This function identifies an area of interest and peforms OCR processing on
%an RGB (non-video) image.
%Regions of interest are identified with MSER, merging neighbouring boxes
%and then checking that a box has a predominantly white perimiter before
%applying OCR.

%% Function 1:  Detectnum (manage still image / video distinction)

function [words] = detectnum(inputImage, verbose)

%default for 'verbose' options is zero (no visualisation of progress)
    if ~exist('verbose','var')
        verbose=0;
    end
    
   %script for processing video images
   if size(inputImage) == 1  
       line=0;
       for time = 0.1:0.5:inputImage.duration;
            line = line +1
            inputImage.CurrentTime = time;
            image = readFrame(inputImage);

            %Call function again with single frame as input image
            ocrArray(line).allresults = detectnum(image, verbose); %store all results per frame
            ocrArray(line).firstresult = ocrArray(line).allresults(1); %store first result per frame

       end
       
       %concatenate frame by frame results
       frames = vertcat(ocrArray.firstresult);
       
       %OCR on video frames can be unreliable due to blurred images.  Some
       %error handling and 'majority vote' handling is therefore
       %required....
       
       %create clean version of results array with only non-blank 2-character responses
       cleanFrames = frames(strlength(strtrim(frames))==2);
       %return most frequent frame-by-frame result as 'majority vote'
       [unqrows,unqID,ID ] = unique(cleanFrames);
       words = {cleanFrames{unqID(mode(ID))}, frames};
   
   % script for processing single image
   else 
       words = detectnumI(inputImage, verbose);
   end
end

%% Function 2:  DetectnumI (find text in still image)

function [ocrWords] = detectnumI(inputImage, verbose)
        
    confThreshold = 0.75;
    global textBoxes;
    global cleanBoxes;
    global ocrResults
    global bboxes;
    global trainImage;
    global mserRegions;
    global ocrWords;
    
    %default for 'verbose' options is zero (no visualisation of progress)
    if ~exist('verbose','var')
        verbose=0;
    end
    
    % Rescale image to 1008 high (25% of standard input images)
    scale = 1008/size(inputImage,1);
    trainImage =imresize(inputImage,scale);
    %[y,x,~]=size(trainImage);
    %TrainImage=TrainImage(y/3:2*y/3,x/4:3*x/4,:); %crop image
    I = rgb2gray(trainImage);

    % Detect MSER regions within image
    [mserRegions, mserConnComp] = detectMSERFeatures(I, ...
        'RegionAreaRange',[round(20) round(500)],'ThresholdDelta',4);
    
    if size(mserRegions,1)==0 %if no mserRegions found at all, give up and return 'Z'
        ocrWords='Z';

    else
        if verbose==3
            % Show regions
            figure
            imshow(I)
            hold on
            plot(mserRegions, 'showPixelList', true,'showEllipses',false)
            title('MSER regions')
            hold off
        end

        % Use regionprops to measure MSER properties
        mserStats = regionprops(mserConnComp, 'BoundingBox');

        %increase bounding boxes by a margin on all sides (but constrain to within image I)
        margin = round(15);
        [yextent, xextent] = size(I);

        bboxes = vertcat(mserStats.BoundingBox);
        bboxes(:,1) = max(0,bboxes(:,1)-margin);
        bboxes(:,2) = max(0,bboxes(:,2)-margin);
        width = bboxes(:,3)+2*margin;
        width = min(width, xextent - bboxes(:,1));
        bboxes(:,3) = width;
        height = bboxes(:,4)+2*margin;
        height = min(height, yextent - bboxes(:,2));
        bboxes(:,4) = height;


        if verbose == 3
            % Show remaining regions
            figure
            IExpandedBBoxes = insertShape(I,'Rectangle',bboxes,'LineWidth',3);
            imshow(IExpandedBBoxes); title('All boxes + margin');
            hold on
            plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        end

        %Merge overlapping boxes

        % Compute the overlap ratio
        overlapRatio = bboxOverlapRatio(bboxes, bboxes);

        % Set the overlap ratio between a bounding box and itself to zero to
        % simplify the graph representation.
        n = size(overlapRatio,1);
        overlapRatio(1:n+1:n^2) = 0;

        % Create the graph
        g = graph(overlapRatio);

        % Find the connected text regions within the graph
        componentIndices = conncomp(g);

        % Merge the boxes based on the minimum and maximum dimensions.
        textBoxes = accumarray(componentIndices', bboxes(:,1), [], @min);
        textBoxes(:,2) = accumarray(componentIndices', bboxes(:,2), [], @min);
        textBoxes(:,3) = accumarray(componentIndices', bboxes(:,3)+bboxes(:,1), [], @max)-textBoxes(:,1);
        textBoxes(:,4) = accumarray(componentIndices', bboxes(:,4)+bboxes(:,2), [], @max)-textBoxes(:,2);

        if verbose==3
            % Show merged regions
            figure
            ITextBoxes = insertShape(I,'Rectangle',textBoxes,'LineWidth',3);
            imshow(ITextBoxes); title('Connected Boxes');
            hold on
            plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        end

        %Remove boxes without white edges
        perimColour=[];
        for i=1:size(textBoxes,1)
            left=round(textBoxes(i,1)+1);
            top=round(textBoxes(i,2)+1);
            right=round(textBoxes(i,3)+left-2);
            bottom=round(textBoxes(i,4)+top-2);
            perimTop=mean(mean(I(top:top+3,left:right)));
            perimBottom=mean(mean(I(bottom-3:bottom,left:right)));
            perimLeft=mean(mean(I(top:bottom,left:left+3)));
            perimRight=mean(mean(I(top:bottom,right-3:right)));
            perimColour(i)=(perimTop+perimBottom+perimLeft+perimRight)/4;
        end

        %set white colour background threshold to lower of 210 or brightest perim
        colourThreshold=min(210,max(perimColour));

        %keep only white background boxes
        cleanBoxes = textBoxes;
        if size(textBoxes,1)>1
            cleanBoxes(perimColour<colourThreshold,:)=[];
        end

        % Show MSER, clean boxes with white background

        if verbose==3
            % Show merged regions
            figure
            ITextBoxes = insertShape(I,'Rectangle',cleanBoxes,'LineWidth',3);
            imshow(ITextBoxes); title('White Background Boxes');
            hold on
            plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        end

        % Perform OCR on regions and get words from results

        cleanBoxes = cleanBoxes + [1 1 -2 -2]; %ensure not outside image
        ocrResults = ocr(imsharpen(I), cleanBoxes, 'TextLayout', 'Word', 'CharacterSet','0':'9');
        ocrWords = vertcat(ocrResults.Words); %concatenate into string array
        ocrWords = strrep(ocrWords,' ',''); %remove any spaces within text
        confidence = vertcat(ocrResults.WordConfidences);

        % If more than one result, remove less confident results and remove 
        % blank ocr results (hopefully these aren't text!)
        if size(cleanBoxes,1) > 1
            idx = ocrWords==" " | ocrWords=="" | size(ocrWords,1)==0;
            % Also remove ocr results with confidence < 75% (hopefully these aren't
            % text!) but always keep at least one result
            idx = idx | confidence<min(confThreshold, max(confidence));
            cleanBoxes(idx,:)=[];
            ocrWords(idx)=[];
        end
        
        % If no boxes found, return 'X' (and dummy box for display)
        if size(cleanBoxes,1) == 0
            cleanBoxes = [100 100 100 100];
            ocrWords = 'X';
        end

        % If only one box but no text, this is returned as 0x1 cell, return
        % 'Y'
        if size(ocrWords,1)==0
            ocrWords = 'Y';
        end
        
        % If box found but no words, return 'Y'
        if strcmp(ocrWords,'') 
            ocrWords = 'Y';
        end

        
        % Show MSER, clean boxes and detected text
        if verbose>0
            figure
            Iocr = insertObjectAnnotation(I,'Rectangle',cleanBoxes,ocrWords);
            imshow(Iocr), title('Detected text');
            if verbose>2
                hold on
                plot(mserRegions, 'showPixelList', true,'showEllipses',false)
            end
        end
    end
end

