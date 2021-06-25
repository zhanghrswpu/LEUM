function [data_ave, processed_data, rating_ind, tempCnt] = preprocess_dataset(original_data, user_num, item_num)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
data_2_vec = zeros(1, user_num*item_num);
processed_data = zeros(user_num, item_num);
[m, ~] = size(original_data);
rating_ind = zeros(user_num, item_num);
total_rating = 0;
tempCnt = 1;
for i = 1:m
    tempUserId = original_data(i, 1);
    tempItemId = original_data(i, 2);
    tempRating = original_data(i, 3);
    if and(tempUserId<=user_num, tempItemId<=item_num)
        processed_data(tempUserId, tempItemId) = tempRating;
        rating_ind(tempUserId, tempItemId) = 1;
        data_2_vec(1, tempCnt) = tempRating;
        tempCnt = tempCnt + 1;
    end
    total_rating = total_rating + tempRating;
end
data_ave = total_rating / m;
%processed_data = sparse(processed_data);
processed_data = processed_data - data_ave; 
end

