%% Initialize
clear;clc;
close all;

%% Define Geometry
% geo=defaultGeometry('nVoxel',[128;128;128],'nDetector',[1024;1024]);    
% DSD必须比DSO大，实际上DSD越接近１，感兴区被放的越大，其中他俩大小好像没啥影响，但是数字小好像跑的慢
% Distances         
geo.DSD = 100.3;             % Distance Source Detector  (mm)   
geo.DSO = 100.2;             % Distance Source Origin    (mm)
% Detector parameters
geo.nDetector=[512; 512];    % number of pixels      (px)
geo.dDetector=[0.1; 0.1];    % size of each pixel    (mm)
geo.sDetector=geo.nDetector.*geo.dDetector; % total size of the detector    (mm)
% Image parameters
geo.nVoxel=[512;512;128];      % number of voxels              (vx) 
geo.sVoxel=[51.2;51.2;128];    % total size of the image       (mm)
geo.dVoxel=geo.sVoxel./geo.nVoxel;   % size of each voxel      (mm)
% Offsets
geo.offOrigin =[0;0;0];              % Offset of image from origin   (mm)              
geo.offDetector=[0; 0];              % Offset of Detector            (mm)

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
                                            
geo.mode='parallel';                            % 'cone' Or 'parallel'. Geometry type.  

%% Load data and generate projections 

% see previous demo for explanation
angles=linspace(0,1*pi,200);
% angles=linspace(pi/180,1*pi,201);
% angles=linspace(pi/180,1*pi,200);


proj = single(zeros(geo.nDetector(1),geo.nDetector(2),length(angles)));
% load files
for ii=1:length(angles)
    imageName=strcat('D:\gh\CodeMATLAB\data\gluedbox\raw\proj\',num2str(ii,"%03d"),'.tif');
    I =imread(imageName);
%     I1=I(:,:,1:3);%删除第四维 A
%     I2=rgb2gray(I1);%转灰度图
    proj(:,:,ii)=single(I);
end

% Step 3: validate

% plotProj(proj,angles)
% you need to make sure that:
%     1) white=metal/bone/high density and black=air.
%         If this is not the case
proj=-log(proj/(max(proj(:))+1)); % Beer-Lanbert law
%     2) rotation happens left-right instead of top-bottom.
%        If its top bottom:
% proj=permute(proj,[2,1,3]);

% Step 4: test
% out_folder_path = "D:\gh\CodeMATLAB\result\SART\";
imgFDK=FDK(proj,geo,angles);
for i=1:geo.nVoxel(3)-1
    img = imgFDK(:,:,i);
    imwrite(img,strcat('D:\gh\CodeMATLAB\result\FDK\FDK_raw3_test0\',num2str(i),'.png'));
end
 
% volumeViewer(imgFDK);
% plotImg(imgFDK,'dim','z','Savegif','D:\gh\CodeMATLAB\result\FDK\FDKraw3_test0.gif');
% If this does not look good, possible things to check:
% - Are the angles in the right direction? maybe they need to be inverted. 
% - Is your geometry properly defined? mistake on geometry will be
%   detrimental to image quality
% - if microCT: halos around the features? COR correction needed. 


