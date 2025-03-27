% Define video file and output folder

path = "../../../experimental_studies/matlab_calibrations/stereo_calibration/";
videoFile = path + "CL2/G2_sync.MP4";
outputFolder = path + "CL2/G2_frames";
mkdir(outputFolder); % Create folder if it doesn't exist

% Read video
v = VideoReader(videoFile);
frameRate = v.FrameRate;

frameIdx = 1;
skipFrames = 5; % Number of frames to skip

while hasFrame(v)
    img = readFrame(v); % Read the frame

    % Save the current frame
    imwrite(img, fullfile(outputFolder, sprintf('frame_%04d.jpg', frameIdx)));
    frameIdx = frameIdx + 1;

    % Skip the next 100 frames
    for i = 1:skipFrames
        if hasFrame(v)
            readFrame(v); % Read and discard the frame
        else
            break;
        end
    end
end

disp('Frames extracted successfully.');