%% Initialize
clear;clc;
close all;

%% Define Geometry
% geo=defaultGeometry('nVoxel',[128;128;128],'nDetector',[1024;1024]);        
%DSD必须比DSO大，实际上DSD越接近１，感兴区被放的越大，其中他俩大小好像没啥影响，但是数字小好像跑的慢
% Distances         
geo.DSD = 600.3;                             % Distance Source Detector      (mm)   
geo.DSO = 600.2;                             % Distance Source Origin        (mm)
% Detector parameters
geo.nDetector=[512; 512];					% number of pixels              (px)
geo.dDetector=[0.1; 0.1]; 					% size of each pixel            (mm)
geo.sDetector=geo.nDetector.*geo.dDetector; % total size of the detector    (mm)
% Image parameters
geo.nVoxel=[512;512;128];                   % number of voxels              (vx) 这个决定了输出的大小
geo.sVoxel=[51;51;128];                   % total size of the image       (mm)
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
% angles=linspace(pi/180,pi,201);
angles=linspace(0,pi,200);

% prealocate   
proj = single(zeros(geo.nDetector(1),geo.nDetector(2),length(angles)));
% load files
in_folder_path = "D:\gh\CodeMATLAB\data\proj\";
out_folder_path = "D:\gh\CodeMATLAB\result\SART\";
files = dir(fullfile(in_folder_path,'*.png'));
for i = 1:length(angles)
    name = files(i).name;
    imageName = strcat(in_folder_path,name);
    I =imread(imageName);
    proj(:,:,i)=single(I);
end

proj=-log(proj/(max(proj(:))+1)); % Beer-Lanbert law

lambda=1;
lambdared=0.999;
initmode='none';
verbose=true;
qualmeas={'RMSE'};

% SIRT and SART both have no extra input parameters.
% =========================================================================
imgOSSART=OS_SART(proj,geo,angles,80,...
                            'lambda',lambda,'lambda_red',lambdared,'verbose',verbose,'QualMeas',qualmeas);
%% 将截面放入文件夹存储起来
for i=1:geo.nVoxel(3)
    img = imgOSSART(:,:,i);
    imwrite(img,strcat('D:\gh\CodeMATLAB\result\OS_SART\OSSART_raw\',num2str(i),'.png'));
end
% volumeViewer(img);
%% plot the results
plotImg([imgOSSART;],'Dim','Z','Savegif','D:\gh\CodeMATLAB\result\OS_SART\ossart_proj_3.gif');


