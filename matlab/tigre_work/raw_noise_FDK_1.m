
clc;clear;

folder_path = "D:\gh\CodeMATLAB\result\FDK";
files = dir(fullfile(folder_path+"\FDK_raw3_select",'*.png'));

for i = 1:length(files)
        z = files(i).name;
        in_path = folder_path + "\FDK_raw3_select\" + z;
        out_file_path = folder_path + "\FDK_raw3_noise\" + z;
        in_img = imread(in_path);
        if ~exist(in_path,'file')==0
            out_img = addNoise(in_img,0.0049);
        end
        out_img = mat2gray(out_img);
        imwrite(out_img,out_file_path);
end
disp("over");