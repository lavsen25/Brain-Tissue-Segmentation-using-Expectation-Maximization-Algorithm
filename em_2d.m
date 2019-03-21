% Image Segmentation Lab 2 
% Expectation Maximization 
% Author: Lavsen Dahal

clc;
clear all;
close all;
%Load the NIFTY Volume for Input Image and GroundTruth
addpath ( 'nifti' , '-end' )
imagePathT1= 'Data/1/T1.nii';
imagePathFlair = 'Data/1/T2_FLAIR.nii';
imagePathLabel= 'Data/1/LabelsForTesting.nii';

%View of Nifti Images ----------------
%Visualization of one of the 2D slice
NiT1= load_untouch_nii(imagePathT1);
NiFlair= load_untouch_nii(imagePathFlair);

GT= load_untouch_nii(imagePathLabel);

size_vol=size(NiT1.img);
seg_all_results=zeros(size_vol(1),size_vol(2),size_vol(3));

slice_to_process=40;
for slice_i = 1:slice_to_process
    
fprintf('----------------------------------');
fprintf('Currently Processing slice %d\n', slice_i);
fprintf('----------------------------------');
slice_no=slice_i;

 
 %Removing all the background 0 from the slices
 slice_T1= NiT1.img(:,:,slice_no);
 [r_image,col_image]= size(slice_T1);
 
 slice_T1= reshape(slice_T1,numel(slice_T1),1);
 length_image=size(slice_T1,1);
 temp_T1=slice_T1;
 
 slice_flair= NiFlair.img(:,:,slice_no);
 
 slice_flair= reshape(slice_flair,numel(slice_flair),1);
 temp_flair=slice_flair;
 
 slice_gt= GT.img(:,:,slice_no);
 slice_gt_untouch=slice_gt;
 slice_gt= reshape(slice_gt,numel(slice_gt),1);
 
 vec_gt_with0_ind= find(~slice_gt);
 vec_gt_without0_ind= find(slice_gt);
 temp_T1(vec_gt_with0_ind)=[];
 temp_flair(vec_gt_with0_ind)=[];
 
 if (isempty(temp_T1) || isempty(temp_flair))
     disp('No data in groudtruth');
     temp_flair=slice_flair;
     temp_T1=slice_T1;
     
 end
 

%-------------------------------------------------------------------------%
%Initialization by using K-means Algorithm
data= [temp_T1 temp_flair] ;
data=double(data);
[r,c]=size(data);
[idx,centroids] = kmeans(data,3);

%For fixing the random nature of k-means in assigning labels
%By checking the centroids of T1 image returned from k-means, 
%it is identified CSF has lowest mean, followed by GM and WM.
%CSF- Forced Label 1, GM- Label 2, WM- Label 3 for consistency with
%groundtruth

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
% % 
% [~,p]= cholcov(cov_m(:,:,j), 0);
% if(p~=0)
%    %cov_m(:,:,j)= cov_m(:,:,j) + .00001 * eye(2);
%    cov_m(:,:,j)=[1 0 ; 0 1];
% end
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
ll=sum(log(sum(member_wt,2)));  %Caculate log likelihood
eps_new = abs(ll-ll_old);
count=count+1;
% fprintf('Number of iterations is %d \n ',count); 
% fprintf('Log likelihood is %d \n ',ll); 
if ( eps_new < eps || count > max_iter)
            break;
end

end

% %Post-Processing after reaching convergence
% %Adding zeros back to the background positions
%

%Change the cluster responsibility to the class label. 
%class=zeros(1,length_image);
member_wt=member_wt';
 [~,class]=max(member_wt);
 data_last=zeros(1,length_image);     % %Adding background zeros
 data_last(vec_gt_with0_ind)=0;
 data_last(vec_gt_without0_ind)=class;
 seg_result=reshape(data_last,  r_image, col_image);
 seg_all_results(:,:,slice_i)=seg_result;   %Save all the segmentation results in 3d array
 
 RGB = label2rgb(seg_result, 'hsv' ,'k');
% figure,
% subplot(141), imshow(NiT1.img(:,:,slice_no),[] ), title('A 2d T1 Slice');
% subplot(142),imshow(NiFlair.img(:,:,slice_no),[]), title('A 2D T2 Flair');
% subplot(143),imshow(GT.img(:,:,slice_no),[]), title('A 2D Groundtruth');
% subplot(144), imshow(RGB), title('Segmentation Result');

 dice_score = compute_dice(seg_result,slice_gt_untouch);
 fprintf('The Dice Coefficient for CSF, GM And WM are:\n');
 disp(dice_score');
 
figure,subplot(121),imshow(GT.img(:,:,slice_no),[]), title('A 2D Groundtruth');
subplot(122), imshow(RGB), title('Segmentation Result'); 
dim=[0.4 0.05 0.3 0.2];
 if isnan(dice_score(2)) || isnan(dice_score(3))
        str = {'Dice Coefficient',strcat('CSF : ',num2str(dice_score(1))) };  
    else
        str = {'Dice Coefficient',strcat('CSF : ',num2str(dice_score(1))), strcat('Gray Matter :',num2str(dice_score(2))),strcat('White Matter : ', num2str(dice_score(3))) };
 end
 
annotation('textbox',dim,'String',str,'FitBoxToText','on' ,'Color','blue', 'FontSize' , 16);
 pause(1);
end %end of for loop for all slices


%Compute dice average after getting the results 
dice_all=zeros(3,1);
 for label = 1:3
 dice_all(label)=dice_average(seg_all_results,GT,label,slice_to_process);
 end
 
 fprintf('The Dice Coefficient average for CSF, GM And WM are:\n');
 disp(dice_all');
%  

% 
% 
