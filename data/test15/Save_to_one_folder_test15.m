%한 폴더로 모으기.
clear all
clc

path_name = 'D:\ptorch\projects\Temporal_Video_SR\data\test15\test_15fps';
output_path = 'D:\ptorch\projects\Temporal_Video_SR\data\test15\one_folder';
list = dir(path_name);
n = length(list);

for i=3:n
    path_name2 = strcat(path_name,'\',list(i).name);
    list2 = dir(path_name2);
    m = length(list2);
    
    if i-3 < 10
        folder_name = strcat('00', int2str(i-3))
    elseif i-3 < 100
        folder_name = strcat('0', int2str(i-3))
    end
    
    for j=3:m
        path_name3 = strcat(path_name2,'\',list2(j).name);
        out_name3 = strcat(output_path, '\',strcat(folder_name, '_', list2(j).name));
        data3 = imread(path_name3);
     
        imwrite(data3, out_name3);
    
        
    end
    
end