% Lab 1 - Detecció de vehicles en seqüències de tràfic

% CONFIGURACIÓ
input_path = "fotos/input";
gt_path = "fotos/groundtruth";
output_path = "fotos/output";
alpha_val    = 0.5;
beta_val     = 60;
thr          = 80;
video_output = "resultats.avi";

% Crear carpetes output si no existeixen
if ~exist(fullfile(output_path, 'simple'), 'dir'), mkdir(fullfile(output_path, 'simple')); end
if ~exist(fullfile(output_path, 'gauss'),  'dir'), mkdir(fullfile(output_path, 'gauss'));  end
if ~exist(fullfile(output_path, 'morph'),  'dir'), mkdir(fullfile(output_path, 'morph'));  end

% TASCA 1 - CARREGAR DATASET
files = dir(fullfile(input_path, '*.jpg'));
files = sort({files.name});
files = files(1:300);

train_files = files(1:150);
test_files  = files(151:300);

first_img = imread(fullfile(input_path, train_files{1}));
if size(first_img, 3) == 3
    first_img = rgb2gray(first_img);
end
[H, W] = size(first_img);

train_images = zeros(H, W, 150, 'double');
for i = 1:150
    img = imread(fullfile(input_path, train_files{i}));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    train_images(:,:,i) = double(img);
end
fprintf('Training images loaded: %d x %d x %d\n', H, W, 150);

% TASCA 2 - MITJANA I STD
mu    = mean(train_images, 3);
sigma = std(train_images, 0, 3);
fprintf('Background model computed\n');

imwrite(uint8(mu),    fullfile(output_path, 'mean.png'));
imwrite(uint8(sigma), fullfile(output_path, 'std.png'));

figure; imshow(uint8(mu));    title('Mitjana (background model)');
figure; imshow(uint8(sigma)); title('Desviacic estàndard');

% TASCA 5 - Inicialitzar vídeo
v = VideoWriter(video_output, 'Motion JPEG AVI');
v.FrameRate = 20;
open(v);

% TASCA 6 - ACCURACY + bucle test
acc_simple = zeros(1, 150);
acc_gauss  = zeros(1, 150);
acc_morph  = zeros(1, 150);

for i = 1:150
    img = imread(fullfile(input_path, test_files{i}));
    if size(img, 3) == 3, img = rgb2gray(img); end

    gt_name = strrep(test_files{i}, 'in', 'gt');
    gt_name = strrep(gt_name, '.jpg', '.png');
    gt = imread(fullfile(gt_path, gt_name));
    if size(gt, 3) == 3, gt = rgb2gray(gt); end

    % TASCA 3: model simple
    seg_simple = simple_subtraction(img, mu, thr);

    % TASCA 4: model gaussià
    seg_gauss = gaussian_model(img, mu, sigma, alpha_val, beta_val);

    % Morfologia sense Toolbox
    seg_morph = apply_morphology(seg_gauss);

    % Eliminar components petits sense Toolbox
    seg_morph = remove_small_components(seg_morph, 250);

    out_simple    = uint8(seg_simple * 255);
    out_gauss     = uint8(seg_gauss  * 255);
    out_morph_255 = uint8(seg_morph  * 255);

    imwrite(out_simple,    fullfile(output_path, 'simple', test_files{i}));
    imwrite(out_gauss,     fullfile(output_path, 'gauss',  test_files{i}));
    imwrite(out_morph_255, fullfile(output_path, 'morph',  test_files{i}));

    writeVideo(v, out_morph_255);

    gt_fg = uint8(gt == 255);
    acc_simple(i) = sum(sum(out_simple    == gt_fg * 255)) / numel(out_simple);
    acc_gauss(i)  = sum(sum(out_gauss     == gt_fg * 255)) / numel(out_gauss);
    acc_morph(i)  = sum(sum(out_morph_255 == gt_fg * 255)) / numel(out_morph_255);
end

close(v);

fprintf('\n--- RESULTATS MITJANS ---\n');
fprintf('Accuracy model simple:   %.4f\n', mean(acc_simple));
fprintf('Accuracy model gaussia:  %.4f\n', mean(acc_gauss));
fprintf('Accuracy amb morfologia: %.4f\n', mean(acc_morph));
fprintf('Video guardat a: %s\n', video_output);


% FUNCIONS LOCALS

function fg = simple_subtraction(img, mu, thr)
    diff = abs(double(img) - mu);
    fg   = uint8(diff > thr);
end

function fg = gaussian_model(img, mu, sigma, alpha_val, beta_val)
    diff = abs(double(img) - mu);
    fg   = uint8(diff > (alpha_val .* sigma + beta_val));
end

function mask = apply_morphology(mask)
    % Dilate 2x2
    k1   = ones(2,2);
    mask = double(conv2(double(mask), k1, 'same') > 0);
    % Close = dilate 7x7 + erode 7x7
    k2 = ones(7,7);
    n  = sum(k2(:));
    mask = double(conv2(mask, k2, 'same') > 0);
    mask = uint8(conv2(mask, k2, 'same') >= n);
end

function mask = remove_small_components(mask, min_area)
    [H, W] = size(mask);
    labeled = zeros(H, W, 'int32');
    current_label = 0;

    for r = 1:H
        for c = 1:W
            if mask(r,c) == 1 && labeled(r,c) == 0
                current_label = current_label + 1;
                % BFS
                queue = [r, c];
                labeled(r,c) = current_label;
                head = 1;
                while head <= size(queue,1)
                    cr = queue(head,1);
                    cc = queue(head,2);
                    head = head + 1;
                    % 4-connexitat: amunt, avall, esquerra, dreta
                    neighbors = [cr-1,cc; cr+1,cc; cr,cc-1; cr,cc+1];
                    for k = 1:4
                        nr = neighbors(k,1);
                        nc = neighbors(k,2);
                        if nr>=1 && nr<=H && nc>=1 && nc<=W
                            if mask(nr,nc)==1 && labeled(nr,nc)==0
                                labeled(nr,nc) = current_label;
                                queue(end+1,:) = [nr, nc]; %#ok<AGROW>
                            end
                        end
                    end
                end
            end
        end
    end

    % Eliminar components amb àrea < min_area
    for lbl = 1:current_label
        area = sum(labeled(:) == lbl);
        if area < min_area
            mask(labeled == lbl) = 0;
        end
    end
end
