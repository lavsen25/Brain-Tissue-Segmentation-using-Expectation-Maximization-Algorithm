% Author: Lavsen Dahal
%This script performs the expectation maximization for 3d volume and
%displays the average dice of all the slices.
%The scripts also provides the option to visualize the groundtruth and the 
%corresponding segmentation results for some or all slides.
%By default it is set to visualize slides no 25-30 in z dimension of NIFTY.

clc;
clear all;
close all;
%Load the NIFTY Volume for Input Images and GroundTruth
addpath ( 'nifti' , '-end' )
imagePathT1= 'Data/1/T1.nii';
imagePathFlair = 'Data/1/T2_FLAIR.nii';
imagePathLabel= 'Data/1/LabelsForTesting.nii';

NiT1= load_untouch_nii(imagePathT1);
NiFlair= load_untouch_nii(imagePathFlair);
GT= load_untouch_nii(imagePathLabel);   

%Removing all the background 0 from the slices
full_vol_T1= reshape(NiT1.img,numel(NiT1.img),1);
temp_T1=full_vol_T1;

full_vol_flair= reshape(NiFlair.img,numel(NiFlair.img),1);
temp_flair=full_vol_flair;

full_vol_gt= reshape(GT.img,numel(GT.img),1);
length_gt=length(full_vol_gt);

vec_gt_with0_ind= find(~full_vol_gt);
vec_gt_without0_ind= find(full_vol_gt);
temp_T1(vec_gt_with0_ind)=[];
temp_flair(vec_gt_with0_ind)=[];

%-------------------------------------------------------------------------%
%Initialization by using K-means Algorithm
data= [temp_T1 temp_flair] ;
data=double(data);
[r,c]=size(data);
[idx,centroids] = kmeans(data,3);

% For fixing the random nature of k-means in assigning labels
% By checking the centroids of T1 image returned from k-means, 
% it is identified CSF has lowest mean, followed by GM and WM.
% CSF- Forced Label 1, GM- Label 2, WM- Label 3 for consistency with
% groundtruth

[~,index] =sort(centroids(:,1));
temp_idx=idx;
for i=1:3
    compare=(index(i)==idx);
    temp_idx(compare)=i;        
end
centroids=sortrows(centroids);
idx=temp_idx;

unq_idx=unique(idx);
no_cluster=length(unq_idx);

% Initialization of cluster Parameters  weight(wt), mean (mu), covariance matrix
% (cov_m) for each hard-coded cluster assignments
alpha=zeros(1,no_cluster);
mu=zeros(no_cluster,2);
cov_m= zeros(2,2,no_cluster);
data_all=cell(1,no_cluster);
nksoft=zeros(1,no_cluster);

for i = 1:no_cluster
    uniq_indx= (idx==unique(i));
    nksoft(i)=sum(uniq_indx);
    alpha(i)=nksoft(i)/length(data);
    mu(i,:)=centroids(i,:);
    data_temp_T1= temp_T1(uniq_indx);
    data_temp_flair=temp_flair(uniq_indx);
    data_all{i}= double([data_temp_T1 data_temp_flair]);
    %Compute covariance matrix for each cluster
    temp_cov = bsxfun (@ minus, data_all{i},mu(i,:));
    cov_m(:,:,i) = (1/nksoft(i))* (temp_cov' * temp_cov) ;
end

%E-step : Estimating cluster responsibilities from given cluster parameter
%initializations/estimates
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
count=0;
max_iter = 100;
member_wt=zeros(length(data),3);
data=double(data);
mu=double(mu);
cov_m=double(cov_m);

while (1)
    
for j = 1:no_cluster
member_wt(:,j)=double(alpha(j)*mvnpdf(data, mu(j,:), cov_m(:,:,j))); 
end

member_wt_sum=sum(member_wt,2)+eps;
ll=sum(log(member_wt_sum));
member_wt=(member_wt ./ (member_wt_sum )) ;


%-------------------------------------------------------------------------%
%M-step: Maximize likelihood over parameters given current responsibilities
%Estimate cluster parameters from soft assignments
%Update means, weights and covariances
 
for i = 1:no_cluster
Nksoft=sum(member_wt(:,i));
nksoft(i)=Nksoft;
alpha(i)= nksoft(i)/length(data);
mu(i,:)= (1/nksoft(i)) * sum(member_wt(:,i).*data);
temp_1 = bsxfun (@ minus, data,mu(i,:));
temp=member_wt(:,i).*temp_1;
cov_m(:,:,i) = (1/nksoft(i))* (temp_1' * temp) ;
end

ll_old=ll;
ll=sum(log(sum(member_wt,2)));  %Compute log_likelihood

eps_new = abs(ll-ll_old);
count=count+1;
fprintf('Number of iterations is %d \n ',count); 
fprintf('Log likelihood is %d \n ',ll); 
if ( eps_new < eps || count > max_iter)
            break;
end

end

% %Post-Processing after reaching convergence
% %Adding zeros back to the background positions

%Change the cluster responsibility to the class label. 
%class=zeros(1,length_image);
 member_wt=member_wt';
 [~,class]=max(member_wt);
% %Adding background zeros
 data_last=zeros(1,length_gt);

 data_last(vec_gt_with0_ind)=0;
 data_last(vec_gt_without0_ind)=class;
 size_gt=size(GT.img);
 seg_result=reshape(data_last,  size_gt(1),size_gt(2),size_gt(3));
 
 %------------------------------------------------------------------------%
 %Compute Dice Score
  dice_all=zeros(3,1);
 for label = 1:3
 dice_all(label)=dice_average(seg_result,GT,label,size_gt(3));
 end
 
 fprintf('The Dice Coefficient average for CSF, GM And WM are:\n');
 disp(dice_all');
 %------------------------------------------------------------------------%

 %Loop To visualize the results in 2d slices
 start_slice=25;
 end_slice=30;
 for i = start_slice:end_slice
     dice_score=compute_dice(double(seg_result(:,:,i)),double(GT.img(:,:,i)));
     RGB = label2rgb(seg_result(:,:,i), 'hsv' ,'k');
    figure,subplot(121),imshow(GT.img(:,:,i),[]), title('A 2D Groundtruth');
    subplot(122), imshow(RGB), title('Segmentation Result'); 
    dim=[0.4 0.05 0.3 0.2];
    if isnan(dice_score(2)) || isnan(dice_score(3))
        str = {'Dice Coefficient',strcat('CSF : ',num2str(dice_score(1))) };  
    else
        str = {'Dice Coefficient',strcat('CSF : ',num2str(dice_score(1))), strcat('Gray Matter :',num2str(dice_score(2))),strcat('White Matter : ', num2str(dice_score(3))) };
    end
    annotation('textbox',dim,'String',str,'FitBoxToText','on' ,'Color','blue', 'FontSize' , 16);
   % pause(1);
 end



