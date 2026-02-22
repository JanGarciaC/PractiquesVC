%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% VC i PSIV                                                      %%%
%%% Lab 0 (basat en les pr�ctiques de Gemma Rotger)                %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
% Hello! Welcome to the computer vision LAB. This is a section, and 
% you can execute it using the run section button on the top panel. If 
% you prefer, you can run all the code using the run button. Run this 
% section when you need to clear your data, figures and console 
% messages.
clearvars,
close all,
clc,

% With addpath you are adding the image path to your main path
% addpath('img')


%% PROBLEM 1 (+0.5) --------------------------------------------------
% TODO. READ THE CAMERAMAN IMAGE.
disp('Reading the image...');
imatge = imread('img/cameraman.jpg');
if size(imatge, 3) == 3
    imatge = rgb2gray(imatge);
end 



%% PROBLEM 2 (+0.5) --------------------------------------------------
% TODO: SHOW THE CAMERAMAN IMAGE
figure(1);
imshow(imatge);
title('Cameraman image');



%% PROBELM 3 (+2.0) --------------------------------------------------
% TODO. Negative efect using a double for instruction

disp('Computing the negative effect using a double for instruction...');
tic;

[rows, cols] = size(imatge);
imatge_neg = zeros(rows, cols, 'uint8');

for i = 1:rows
    for j = 1:cols
        imatge_neg(i, j) = 255 - imatge(i, j);
    end
end

toc;

figure(1);
imshow(imatge_neg);
title('Negative effect using double for loop');

% TODO. Negative efect using a vectorial instruction

disp('Computing the negative effect using a vectorial instruction...');
tic;

imatge_neg_vector = 255 - imatge;

toc;

figure(2);
imshow(imatge_neg_vector);
title('Negative effect using vectorial operation');

% You sould see that results in figures 1 and 2 are the same but times
% are much different.

%% PROBLEM 4 (+2.0) --------------------------------------------------
% Give some color (red, green or blue)
% Una imatge en color està composta per 3 canals: vermell, verd i blau.
% Muntarem una imatge amb tres imatges grises:
% - Canal vermell: imatge original
% - Canal verd: imatge negativa
% - Canal blau: imatge original

disp('Creating colored image - Method 1: Creating empty image and filling channels...');

% Method 1: Creating an empty image and filling each channel
r = imatge;
g = imatge_neg_vector;
b = imatge;

im_col_method1 = zeros(size(imatge, 1), size(imatge, 2), 3, 'uint8');
im_col_method1(:,:,1) = r;  % Red channel
im_col_method1(:,:,2) = g;  % Green channel
im_col_method1(:,:,3) = b;  % Blue channel

figure(1);
imshow(im_col_method1);
title('Method 1: Empty image + channel assignment');

disp('Creating colored image - Method 2: Using cat...');

% Method 2: Using cat (concatenation function in Matlab)
im_col_method2 = cat(3, r, g, b);

figure(2);
imshow(im_col_method2);
title('Method 2: Using cat function');

disp('Both methods produce the same result!');


%% PROBLEM 5 (+1.0) --------------------------------------------------

disp('Writing images to disk...');
% Write the original image
imwrite(imatge, 'img/cameraman_original.jpg');
% Write the negative image
imwrite(imatge_neg_vector, 'img/cameraman_negative.jpg');
% Write the colored image
imwrite(im_col_method1, 'img/cameraman_colored.jpg');
disp('Images written successfully');

%% PROBLEM 6 (+1.0) --------------------------------------------------

disp('Extracting lines from the image...');
% Extract line 128 from the grayscale image
lin128 = imatge(128, :);
figure(1);
plot(lin128);
hold on;
% Add a horizontal line with the mean value
mean_val = mean(lin128);
yline(mean_val, '--r', 'LineWidth', 2);
legend(['Original line', sprintf('Mean: %.2f', mean_val)]);
title('Line 128 from grayscale image');
hold off;

% Extract line 128 from the colored image
lin128rgb = im_col_method1(128, :, :);
figure(2);
% Extract each color channel from the line
red_line = squeeze(lin128rgb(1, :, 1));
green_line = squeeze(lin128rgb(1, :, 2));
blue_line = squeeze(lin128rgb(1, :, 3));

plot(red_line, 'r', 'LineWidth', 2); hold on;
plot(green_line, 'g', 'LineWidth', 2);
plot(blue_line, 'b', 'LineWidth', 2);

% Add horizontal lines with the mean values for each channel
mean_red = mean(red_line);
mean_green = mean(green_line);
mean_blue = mean(blue_line);
yline(mean_red, '--r', 'LineWidth', 1.5, 'Alpha', 0.7);
yline(mean_green, '--g', 'LineWidth', 1.5, 'Alpha', 0.7);
yline(mean_blue, '--b', 'LineWidth', 1.5, 'Alpha', 0.7);

legend(sprintf('Red (mean=%.2f)', mean_red), sprintf('Green (mean=%.2f)', mean_green), sprintf('Blue (mean=%.2f)', mean_blue));
title('Line 128 from colored image');
xlabel('Pixel position');
ylabel('Intensity');
hold off;


%% PROBLEM 7 (+2) ----------------------------------------------------

disp('Computing histogram using imhist...');
tic;
hist_data = imhist(imatge);
toc;
figure(1);
plot(hist_data);
title('Histogram using imhist');
xlabel('Gray level');
ylabel('Frequency');

disp('Computing histogram using manual loop...');
tic;
h = zeros(256, 1);
for i = 1:size(imatge, 1)
    for j = 1:size(imatge, 2)
        h(double(imatge(i, j)) + 1) = h(double(imatge(i, j)) + 1) + 1;
    end
end
toc;
figure(2);
plot(h);
title('Histogram using manual loop');
xlabel('Gray level');
ylabel('Frequency');

%% PROBLEM 8 Binarize the image text.png (+1) ------------------------

disp('Reading Alice text image...');
% Read the image
imtext = imread('img/alice.jpg');
if size(imtext, 3) == 3
    imtext = rgb2gray(imtext);
end

% Show the original image
figure(1);
imshow(imtext);
title('Original Alice image');

% Calculate and show the histogram
disp('Computing histogram of the text image...');
hist_text = imhist(imtext);

figure(2);
plot(hist_text);
xlabel('Gray level');
ylabel('Frequency');
title('Histogram of text image');
hold on;
% Mark potential threshold values
xline(120, '--r', 'Alpha', 0.5);
xline(175, '--g', 'Alpha', 0.5);
xline(230, '--b', 'Alpha', 0.5);
legend('Histogram', 'Low (100)', 'Optimal (130)', 'High (160)');
hold off;

% Define 3 different thresholds:
% - th1: Below the optimal value (underestimates the text)
% - th2: The optimal value (good separation)
% - th3: Above the optimal value (overestimates the text)
th1 = 120;   % Low threshold
th2 = 175;   % Optimal threshold
th3 = 230;   % High threshold

disp(sprintf('Applying thresholds: %d (low), %d (optimal), %d (high)', th1, th2, th3));

% Apply the thresholds as binary images (0/1 or logical)
% Below threshold = 0 (black), Above threshold = 1 (white)
threshimtext1 = imtext > th1;
threshimtext2 = imtext > th2;
threshimtext3 = imtext > th3;

% Show the original image and the segmentations in a subplot
figure(3);
subplot(2, 3, 1);
imshow(imtext);
title('Original image');
axis off;

subplot(2, 3, 2);
text(0.5, 0.5, '');
axis off;

subplot(2, 3, 3);
text(0.5, 0.5, '');
axis off;

subplot(2, 3, 4);
imshow(threshimtext1);
title(sprintf('Threshold = %d (Low)', th1));
axis off;

subplot(2, 3, 5);
imshow(threshimtext2);
title(sprintf('Threshold = %d (Optimal)', th2));
axis off;

subplot(2, 3, 6);
imshow(threshimtext3);
title(sprintf('Threshold = %d (High)', th3));
axis off;

disp('Binary segmentation complete!');


%% THE END -----------------------------------------------------------
% Well done, you finished this lab! Now, remember to deliver it 
% properly on Caronte.

% File name:
% lab0_NIU.zip 
% (put matlab file lab0.m and python file lab0.py in the same zip file)
% Example lab0_1234567.zip

















