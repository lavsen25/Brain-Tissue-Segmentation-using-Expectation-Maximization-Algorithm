%This is the main script which gives the possibility to run the algorithm
%in 2d or 3d mode.
%The 2d mode performs the expectation maximization slice by slice. 
%The 3d mode takes the whole 3d volume and gives the final segmentation
%map.
%Written by Lavsen Dahal 
%Submitted for: Masters in Medical Imaging and Applications(MAIA)
%Semester 3: University of Girona - Medical Image Segmentation Lab 2

clc;
clear all;
close all;
alg_type= '3d'; %Possible values: 2d or 3d (2d for slice implementation and 3d for volume

switch alg_type
    case '2d'
        disp('Implementing EM algorithm for 2D slice');  
        em_2d;
    case '3d'
        disp('Implementing EM algorithm for 3D volume');
        em_3d
         otherwise
        disp('wrong choice');
end
