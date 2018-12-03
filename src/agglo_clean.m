% Cleaner code for agglomerative descriptor 
% Need training data examples and spindescriptors
% Which should be under the current dir
spinfiles=dir('*/*_spinCNN_8.dat');
% Change this to the actual path of your 'matlab_pcl'
addpath('~/Documents/matlab_pcl/')

spinfiles2=dir('*/*_spinCNNvectors_8.dat');

% to stare individual data
descriptors=cell(size(spinfiles));
descriptors_sizes=zeros(size(spinfiles,1),1);
bounds=zeros(size(spinfiles,1),1);
vector_fields=cell(size(spinfiles2));
field_minMax=zeros(size(spinfiles2,1),2);
vector_fields_mags=cell(size(spinfiles));
vector_fields_probs=cell(size(spinfiles));
vector_field_mags_c=cell(size(spinfiles));
object_files=cell(size(spinfiles,1),3);
full_field_minMax=zeros(size(spinfiles,1),2);
% read and save all possible training iT under current directory
for i=1:size(spinfiles,1)
    file_path=strcat(spinfiles(i).folder,'/',spinfiles(i).name);
    fileID=fopen(file_path);
    sizes=fread(fileID,[1,2],'uint32');
    rows=sizes(1);
    cols=sizes(2);
    A=fread(fileID,'float');
    fclose(fileID);
    descriptors{i}=transpose(reshape(A,[cols,rows]));
    descriptors_sizes(i)=size(descriptors{i},1);
    file_path=strcat(spinfiles2(i).folder,'/',spinfiles2(i).name);
    fileID=fopen(file_path);
    sizes=fread(fileID,[1,2],'uint32');
    rows=sizes(1);
    cols=sizes(2);
    A=fread(fileID,'float');
    fclose(fileID);
    tokens=strsplit(spinfiles(i).name,'_');
    obj=tokens{2};
    af=tokens{1};
    fileID=strcat(spinfiles(i).folder,'/',af,'_',obj,'_field_clean.pcd');
    tmp=loadpcd(fileID);
    mags=sqrt(sum(abs(tmp(4:6,:)).^2,1));
    full_field_minMax(i,:)=[min(mags) max(mags)];
    tokens=strsplit(spinfiles(i).folder,'/');
    object_files{i,1}=tokens{end};
    object_files{i,2}=af;
    object_files{i,3}=obj;
    vector_fields{i}=transpose(reshape(A,[cols,rows]));
    vector_fields_mags{i}=sqrt(sum(abs(vector_fields{i}).^2,2));
    field_minMax(i,:)=[min(vector_fields_mags{i}) max(vector_fields_mags{i})];
    vector_fields_probs{i}=(vector_fields_mags{i}-field_minMax(i,1)).*((1-0)/(field_minMax(i,2)-field_minMax(i,1)))+0;
    vector_field_mags_c{i}=(vector_fields_mags{i}-full_field_minMax(i,1)).*((1-0)/(full_field_minMax(i,2)-full_field_minMax(i,1)))+0;
end
object_names=string(object_files);
%T=table(object_names(:,1),object_names(:,2),object_names(:,3),round(full_field_minMax(:,1),4),round(full_field_minMax(:,2),4),'VariableNames',{'Directory','Affordance','Object','Min','Max'});
T=table(object_names(:,1),object_names(:,2),object_names(:,3),'VariableNames',{'Directory','Affordance','Object'});

bigDescriptor=[];
extents=zeros(size(spinfiles,1),6);
box_diag=zeros(size(spinfiles));
all_affordances_files=1:size(spinfiles,1);
real_tensor_ids=[];
% agglomerate descriptors in big array
for i=1:size(spinfiles,1)
    bounds(i)=size(bigDescriptor,1);
    extents(i,:)=[min(descriptors{i}) max(descriptors{i})];
    box_diag(i)=sqrt(sum((extents(i,1:3)-extents(i,4:6)).^2,2));
    cnt=contains(spinfiles(i).name,'Contain');
    % do not consider 'contain' affordances or tensors smaller than 10cm
    % (if any)
    if cnt || box_diag(i)<0.1 
        continue
    end
    bigDescriptor=[bigDescriptor;descriptors{i}];
    real_tensor_ids=[real_tensor_ids;i];
end
%% regular grid

p_min=min(bigDescriptor,[],1);
p_max=max(bigDescriptor,[],1);
distance=sqrt((p_min(1)-p_max(1))^2+(p_min(2)-p_max(2))^2+(p_min(3)-p_max(3))^2);
%evenly distribute distance in nxn grid
cell_size=0.01;
xgv=p_min(1):cell_size:p_max(1);
ygv=p_min(2):cell_size:p_max(2);
zgv=p_min(3):cell_size:p_max(3);
[X,Y,Z]=meshgrid(xgv,ygv,zgv);
x_v=reshape(X,[1,size(X,1)*size(X,2)*size(X,3)]);
y_v=reshape(Y,[1,size(Y,1)*size(Y,2)*size(Y,3)]);
z_v=reshape(Z,[1,size(Z,1)*size(Z,2)*size(Z,3)]);
actual_points=[x_v',y_v',z_v'];
clear x_v y_v z_v X Y Z xvg ygv zgv
% get NN for every cell centroid 
Idx=knnsearch(actual_points,bigDescriptor);
bigIdx=1:size(actual_points,1);
C=intersect(bigIdx,Idx);
AffMapPoints=cell(length(C),1);
AffMapIdx=cell(length(C),1);
AffMapAffId=cell(length(C),1);
AffMapAffDiff=cell(length(C),1);
AffMapAffKP=cell(length(C),1);
AffMapAffOr=cell(length(C),1);
AllIds=[];
AffMapAffLarge=cell(length(C),1);
centroid_members=zeros(length(C),1);
real_centres=actual_points(C,:);
better_approx=actual_points(C,:);
AllOrientations=[];
real_ids512=[];
starts_all=zeros(length(C),1);
AllAfordances=[];
stats=zeros(length(C),2);
progress=0;
% clustering, for every non-empty cell centroid, save provenance vectors,
% keypoints , ids ,etc.
for i=1:length(C)
    ids=find(Idx==C(i)); %[1,size(C)]
    ids=ids-1;  %[0,size(C)-1)
    affordances=zeros(length(ids),1);
    orientations=zeros(length(ids),1);
    real_ids=zeros(length(ids),1);
    bigger_ids=zeros(length(ids),1);
    for j=1:length(ids)
        affordance=find(bounds<=ids(j),1,'last')-1; %from 0 to descriptors_size
        %floor(ids(j)/size(descriptors{1},1)); %
        big_id=ids(j)-bounds(find(bounds<=ids(j),1,'last')); %id inside 8-orientation descriptor
        %big_id=ids(j)-affordance*size(descriptors{1},1);
        bigger_ids(j)=big_id;
        orientation=floor(big_id/size(vector_fields{real_tensor_ids(affordance+1)},1));
        real_id=big_id-orientation*size(vector_fields{real_tensor_ids(affordance+1)},1);
        affordances(j)=affordance;
        orientations(j)=orientation;
        real_ids(j)=real_id;
    end
    ids=ids+1;
    AffMapPoints{i}=bigDescriptor(ids,:);
    %If there is only one point in the cluster, make that point the cluster
    %centre .
    if length(ids)==1
        real_centres(i,:)=bigDescriptor(ids,:);
        better_approx(i,:)=bigDescriptor(ids,:);
        %If there are more points in the cluster, make that average the cluster
        %centre.
    else
        better_approx(i,:)=sum(bigDescriptor(ids,:))/length(ids);
    end
    miniData=sqrt(sum((better_approx(i,:)-bigDescriptor(ids,:)).^2,2));
    v=std(miniData)^2;
    m=mean(miniData);
    stats(i,:)=[m,v];
    ids=ids-1;
    AffMapIdx{i}=ids;
    AffMapAffId{i}=affordances;
    AffMapAffDiff{i}=length(unique(affordances));
    AffMapAffKP{i}=real_ids;
    AffMapAffOr{i}=orientations;
    AllOrientations=[AllOrientations; orientations];
    AffMapAffLarge{i}=bigger_ids;
    starts_all(i)=length(real_ids512);
    AllIds=[AllIds;bigger_ids];
    AllAfordances=[AllAfordances;affordances];
    real_ids512=[real_ids512;real_ids];
    centroid_members(i)=length(ids);
    progress_u=int8(100*i/length(C));
    if progress_u-progress>=1
        if mod(progress_u,10)==0
            fprintf(' %d\n',progress_u);
        else
            fprintf(' %d',progress_u);
        end
        progress=progress_u;
    end
end

% At some point we considered sampling from the agglomeration
% with different strategies
% however, best results were obtained when
% all non-empy centroids are considered in the agglo representation

good_samples=1:length(C);

affordance_indices=1:length(real_tensor_ids);% 1:size(spinfiles,1);
affordance_orientations=0:7;
data2=zeros(length(affordance_indices),length(affordance_orientations));
bare_points=[];
points_ids=[];
vector_ids=[];
vector_ids_agglomerative=[];
%%
medoids=good_samples;
x1=good_samples;
medoid_members=zeros(length(medoids),1);
start_all=zeros(length(medoids),1);
medoid_members_counter=0;
%myAff_points=[];
%myVectors=[];
%sample_anAffordance=[];
%myOri=1;
%myAff=4;
tic
for i=1:length(medoids)
    % ids from 1-n_affordances
    diff_ids=intersect(affordance_indices,AffMapAffId{medoids(i)}+1);
    % ids from 0-7 (8 orientations)
    % get 1 affordance 1 orientation (i.e intersection)
    % alternatively we can get all affordances and all orientations
    % as in paper
    diff_ori=intersect(affordance_orientations,AffMapAffOr{medoids(i)});
    start_all(i)=length(vector_ids);
    %fprintf('Orientations %d Affordances %d \n',length(diff_ori),length(diff_ids));
    for j=1:length(diff_ids)
        %get local indices of point for this affordance
        % ids from 1-length(AffMapAffId{medoids(i)})
        idx_local_affordance=find(AffMapAffId{medoids(i)}+1==diff_ids(j));
        for k=1:length(diff_ori)
            %get local indices of point for this orientation
            % ids from 1-length(AffMapAffOr(medoids(i)))
            idx_local_orientation=find(AffMapAffOr{medoids(i)}==diff_ori(k));
            %get indices in common
            idx_local_common=intersect(idx_local_affordance,idx_local_orientation);
            if ~isempty(idx_local_common)
                %get global indices of the intersection -> 0-spinDescriptor
                idx_global=AffMapAffLarge{medoids(i)}(idx_local_common);
                % ids of the points in the bigDesrcriptor
                real_global=idx_global+bounds(diff_ids(j))+1;
                ids=knnsearch(bigDescriptor(real_global,:),better_approx(medoids(i),:));
                %[~,id]=min(Dx(real_global));
                bare_points=[bare_points;bigDescriptor(real_global(ids),:)];
                % ids from 0-affordanceDescriptor
                pv_id=idx_global(ids)-floor(idx_global(ids)/(descriptors_sizes(diff_ids(j))/8))*size(vector_fields{diff_ids(j)},1);
                points_ids=[points_ids; diff_ids(j),diff_ori(k), pv_id];
                %Here I need again the real tensor ids from spinfiles
                vector_ids_agglomerative=[vector_ids_agglomerative;vector_fields{real_tensor_ids(diff_ids(j))}(pv_id+1,:)];
                vector_ids=[vector_ids;pv_id];
                medoid_members_counter=medoid_members_counter+1;
            end
        end
    end
    medoid_members(i)=medoid_members_counter;
    medoid_members_counter=0;
end
% start_i=(myOri*512)+1;
% end_i=start_i+511;
toc
%get per affordance per orientation keypoint counts
data_individual=zeros(length(affordance_indices),length(affordance_orientations));
index=0;
for i=1:size(points_ids,1)
    row=points_ids(i,1);
    col=points_ids(i,2)+1;
    data_individual(row,col)=data_individual(row,col)+1;
end
% build the pointclouds for c++ code
aux_cloud=[medoid_members start_all zeros(length(medoids),1)];
useful_cloud=points_ids;
vector_data=[points_ids vector_ids_agglomerative];
aff_ids=single(points_ids(:,1));
agglo_weights=zeros(size(aff_ids));
for i=1:length(aff_ids)
    %agglo_weights(i)=1-vector_field_mags_c{real_tensor_ids(aff_ids(i))}(points_ids(i,3)+1);
    agglo_weights(i)=1-vector_fields_probs{real_tensor_ids(aff_ids(i))}(points_ids(i,3)+1);
end
actual_mags=sqrt(sum(abs(vector_ids_agglomerative).^2,2));
vectors_data_cloud=[actual_mags agglo_weights zeros(size(aff_ids))];

% save data
% d_number can be modified to something else
% for me it was easier to assign an ID and keep track of different
% IDS/versions
d_number='999';

savepcd(strcat('New',d_number,'_Approx_descriptor_8_members.pcd'),aux_cloud');
savepcd(strcat('New',d_number,'_Approx_descriptor_8_extra.pcd'),useful_cloud');
savepcd(strcat('New',d_number,'_Approx_descriptor_8.pcd'),better_approx(x1,:)');
savepcd(strcat('New',d_number,'_Approx_descriptor_8_points.pcd'),bare_points');
savepcd(strcat('New',d_number,'_Approx_descriptor_8_vectors.pcd'),vector_ids_agglomerative');
savepcd(strcat('New',d_number,'_Approx_descriptor_8_vdata.pcd'),vectors_data_cloud');

a=size(data_individual);
fileID = fopen(strcat('point_count',d_number,'.dat'),'w');
fwrite(fileID,fliplr(a),'uint32');
fwrite(fileID,data_individual,'float');
fclose(fileID);

T=table(object_names(real_tensor_ids,1),object_names(real_tensor_ids,2),object_names(real_tensor_ids,3),'VariableNames',{'Directory','Affordance','Object'});
csv_name=strcat('tmp',d_number,'.csv');
writetable(T,csv_name)