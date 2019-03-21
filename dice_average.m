function dice= dice_average(seg_result,GT,label,dim_in_z) 
%This function returns the dice average for a particular label for all the
%slides

count=0;
dice=0;
 for i = 1:dim_in_z
 dice_score = compute_dice(double(seg_result(:,:,i)),double(GT.img(:,:,i)));
 if ~isnan(dice_score(label)) 
     dice=dice+dice_score(label);
     count=count+1;
 end
 end
  dice=dice/count;
end

  