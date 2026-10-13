function [out_img]=addNoise(in_img,factor)
%        Image=im2double(in_img);
       Image=in_img;
       theta=0:179;
       Noised_Radon_Image=[];       
       for t=1:180
           Radon_Image_line=radon(Image,theta(t));
           for s=1:729
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           %The parameter 0.0025 was the core of this code, this parameter control the 
           %intensity of noise which will be added to CT.
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                Radon_Image_line(s)=(1+normrnd(0,factor))*Radon_Image_line(s);
           end    
           Noised_Radon_Image=[Noised_Radon_Image,Radon_Image_line];
       end
       Radon_Image=radon(Image,theta);
       Original_Image=Image;
       Antiradon_Image=uint8(iradon(Radon_Image,theta));
       Radon_noise=Image-(Antiradon_Image(2:513,2:513));
       Antinoisedimage=uint8(iradon(Noised_Radon_Image,theta));
       NoisedImage=Antinoisedimage(2:513,2:513)-Radon_noise;
       NoisedImage(find(NoisedImage>2000))=2000;
       out_img = NoisedImage;

end
