% Define video file and output folder

% path = "../../../experimental_studies/gaips/matlab_calibrations/stereo_calibration/videos&frames/sync/";
path = "../../../experimental_studies/gaips/matlab_calibrations/colmap/";
videoFile = path + "G3.MP4";
outputFolder = path + "G3_frames";
mkdir(outputFolder); % Create folder if it doesn't exist

% Read video
v = VideoReader(videoFile);
frameRate = v.FrameRate;

frameIdx = 1;
skipFrames = 1; % Number of frame  s to skip

while hasFrame(v)
    img = readFrame(v); % Read the frame

    % Save the current frame
    imwrite(img, fullfile(outputFolder, sprintf('frame_%04d.jpg', frameIdx)));
    frameIdx = frameIdx + 1;

    % Skip the next X frames
    for i = 1:skipFrames
        if hasFrame(v)
            readFrame(v); % Read and discard the frame
        else
            break;
        end
    end
end

disp('Frames extracted successfully.');