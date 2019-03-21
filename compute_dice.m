function dice= compute_dice(image,gt)

%This functions computes the dice score for CSF, GM, AND WM. It assumes the
%labels for CSF is 1, GM is 2 and WM is 3.
image=reshape(image,numel(image),1);
gt=reshape(gt,numel(gt),1);


dice_gt=zeros(length(image),3);
label_total=zeros(3,1);
dice_im=zeros(length(image),3);
dice=zeros(3,1);
inter_vec_dif=zeros(length(image),3);
for i = 1:3
dice_gt(:,i)=gt==i;
label_total(i)=sum(dice_gt(:,i));
dice_im(:,i)=image==i;
inter_vec_dif(:,i)=dice_gt(:,i)-dice_im(:,i);
error=sum(inter_vec_dif(:,i)>0)/label_total(i);

dice(i)=1-error;


end
dice=round(dice,2);

end





