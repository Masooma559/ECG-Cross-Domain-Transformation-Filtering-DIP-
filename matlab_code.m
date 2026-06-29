% 1. Load and Pre-process
img = imread('dataset/normal/heartbeat_0.png');
if size(img, 3) == 3, I = rgb2gray(img); else, I = img; end

% --- STEP 1: DENOISING & ENHANCEMENT ---
denoised = medfilt2(I, [3 3]);
enhanced = adapthisteq(denoised, 'ClipLimit', 0.005, 'Distribution', 'uniform');
final_img = imgaussfilt(enhanced, 0.4);

% --- STEP 2: MORPHOLOGICAL CLEANING ---
se = strel('disk', 1);
morphed = imopen(final_img, se);

% --- STEP 3: SEGMENTATION (R2015 Version) ---
% Find optimal threshold using Otsu's Method
level = graythresh(morphed);
% Use im2bw for MATLAB versions older than 2016a
binary_mask = im2bw(morphed, level);

% --- STEP 4: METRICS ---
mse_val = immse(morphed, I);
psnr_val = psnr(morphed, I);

% --- STEP 5: VISUALIZATION ---
figure('Name', 'DIP Project Pipeline R2015', 'NumberTitle', 'off');

subplot(2,2,1); imshow(I); 
title('1. Original GAF');

subplot(2,2,2); imshow(enhanced); 
title(['2. Enhanced (PSNR: ', num2str(psnr_val, '%.2f'), ' dB)']);

subplot(2,2,3); imshow(morphed); 
title('3. Morphologically Cleaned');

subplot(2,2,4); imshow(binary_mask); 
title('4. Final Binary Segmentation');

fprintf('\n--- R2015 DIP Pipeline Results ---\n');
fprintf('MSE: %0.4f | PSNR: %0.2f dB\n', mse_val, psnr_val);