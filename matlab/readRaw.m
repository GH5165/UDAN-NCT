clc;clear;
maindir = 'D:\gh\CodeMATLAB\data\gluedbox\ob\';
dir_folder = dir(fullfile(maindir, '*.raw'));
names_file = {dir_folder.name};

size_name = size(names_file);
save_dir = 'D:\gh\CodeMATLAB\data\gluedbox\ob\proj\';
q = 1;
M = 512;
N = 512;

for num_file = 1:size_name(2)
    dir_file = fullfile(maindir, names_file{num_file});
    f1 = fopen(dir_file, 'r');
    data = fread(f1, 'uint16');
    fclose(f1);
    len = length(data);
    k = len/(M*N);
    image = reshape(data, M, N, k);
    image = mat2gray(image);
    image = imrotate(image,-90);
    
    b = num2str(q,'%03d'); 
    imwrite(image,[save_dir, b,'.tif']);
    
    q = q+1;
end

disp("over");