%% Initialize
clear;clc;
close all;

%% Define Geometry
% geo=defaultGeometry('nVoxel',[128;128;128],'nDetector',[1024;1024]);        
% Distances         %DSD必须比DSO大，实际上DSD越接近１，感兴区被放的越大，其中他俩大小好像没啥影响，但是数字小好像跑的慢
geo.DSD = 600.3;                             % Distance Source Detector      (mm)   
geo.DSO = 600.2;                             % Distance Source Origin        (mm)
% Detector parameters
geo.nDetector=[512; 512];					% number of pixels              (px)
geo.dDetector=[0.1; 0.1]; 					% size of each pixel            (mm)
geo.sDetector=geo.nDetector.*geo.dDetector; % total size of the detector    (mm)
% Image parameters
geo.nVoxel=[512;512;32];                   % number of voxels              (vx) 这个决定了输出的大小
geo.sVoxel=[51;51;32];                   % total size of the image       (mm)
geo.dVoxel=geo.sVoxel./geo.nVoxel;          % size of each voxel            (mm)
% Offsets
geo.offOrigin =[0;0;0];                     % Offset of image from origin   (mm)              
geo.offDetector=[0; 0];                     % Offset of Detector            (mm)
                                            % These two can be also defined
                                            % per angle

% Auxiliary 
geo.accuracy=0.5;                           % Variable to define accuracy of
                                            % 'interpolated' projection
                                            % It defines the amoutn of
                                            % samples per voxel.
                                            % Recommended <=0.5             (vx/sample)

% Optional Parameters
% There is no need to define these unless you actually need them in your
% reconstruction
                                            
                                            
geo.COR=0;                                  % y direction displacement for 
                                            % centre of rotation
                                            % correction                   (mm)
                                            % This can also be defined per
                                            % angle
                                            
geo.rotDetector=[0;0;0];                    % Rotation of the detector, by 
                                            % X,Y and Z axis respectively. (rad)
                                            % This can also be defined per
                                            % angle        
                                            
geo.mode='parallel';                            % Or 'parallel'. Geometry type.  

%% Load data and generate projections 
views_num=200;
angles=linspace(0,1*pi,views_num);

% prealocate   
proj = single(zeros(geo.nDetector(1),geo.nDetector(2),length(angles)));
% load files
in_folder_path = "D:\gh\CodeMATLAB\data\proj_tif\";
out_folder_path = "D:\gh\CodeMATLAB\result\SART\";
files = dir(fullfile(in_folder_path,'*.tif'));
iii=0;
for ii=1:200/views_num:200
    iii=iii+1;
    name = files(round(ii)).name;
    imageName = strcat(in_folder_path,name);
    I =imread(imageName);
    proj(:,:,iii)=single(I);
end
% 加噪声
proj_noise = single(addCTnoise(proj,'Poisson',1e3)); 
proj_noise = proj; 
proj_noise = -log(proj_noise/(max(proj_noise(:))+1)); % Beer-Lanbert law

lambda=1;
lambdared=0.999;
initmode='none';
verbose=true;
qualmeas={'RMSE'};

%% SIRT and SART both have no extra input parameters.
% =========================================================================
tic;
epochs=100;
dst_path = strcat('D:\gh\CodeMATLAB\result\SART\SART_rawtif_',num2str(views_num),'views1e3_',num2str(epochs));
mkdir(dst_path);
[imgSART,errL2SART,qualitySART]=SART(proj_noise,geo,angles,epochs,...
                            'lambda',lambda,'lambda_red',lambdared,'verbose',verbose,'QualMeas',qualmeas);
% imgOSSART=OS_SART(proj,geo,angles,40,...
%                             'lambda',lambda,'lambda_red',lambdared,'verbose',verbose,'QualMeas',qualmeas);
elapsed_time = toc;
disp(['Elapsed time = ', num2str(elapsed_time)]);
%% 将截面放入文件夹存储起来
for i=1:geo.nVoxel(3)
    img = imgSART(:,:,i);
    imwrite(img,strcat(dst_path,'\',num2str(i),'.png'));
end
% volumeViewer(img);
%% plot the results
% plotImg([imgSART;],'Dim','Z','Savegif','D:\gh\CodeMATLAB\result\SART\sart_proj_3.gif');


