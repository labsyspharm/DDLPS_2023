%% Analysis DDLP data
%  2022/11/29 Jerry Lin

tic;
for i = 1:length(slideName)
    disp(strcat('Processing:',slideName{i}));
    data1 = eval(strcat('data',slideName{i}));
    data1 =data1(:,label1);
    data1 = CycIF_removezero(data1);
    data1{:,2:end}=uint16(data1{:,2:end});
    eval(strcat('data',slideName{i},'=data1;'));
    toc;
    clear data1;
end

%%

for i = 1:length(slideName)
    disp(strcat('Processing:',slideName{i}));
    data1 = eval(strcat('data',slideName{i}));
    data1.Properties.VariableNames = label1;
    eval(strcat('data',slideName{i},'=data1;'));
    toc;
    clear data1;
end

%%
name1 = 'CD20';
marker1 = strcat('mean_',name1);
figure,bar(sumAll{:,marker1}*100);
ytickformat('percentage');
set(gca,'xticklabels',slideINFO.Case_number);
set(gca,'xticklabelrotation',45);
title(strcat(name1,'+'),'FontSize',16);
set(gcf,'color','w');

%% Imoprt data
tic;
for i =1:length(slideName)
    disp(strcat('Processing:',slideName{i}))
    data1 = CycIF_importMcMicro(filelist{i},0.65);
    data1{:,2:end} = uint16(data1{:,2:end});
    data1 = CycIF_removezero(data1);
    eval(strcat('data',slideName{i},'=data1;'));
    toc;
end


%% Counts tables for all slides (only markers)

tic;
maxL = 200;
allcounts = [];

for s=1:length(slideName)
    disp(strcat('Processing:',slideName{s}));
    data1 = eval(strcat('data',slideName{s}));

    data1.col = round(data1.Xt ./ maxL)+1;
    data1.row = round(data1.Yt ./ maxL)+1;
    data1.frame = round(data1.col+ max(data1.col)*(data1.row-1));

    count1 = varfun(@sum,data1,'GroupingVariables','frame','Inputvariables',labelp2);
    if isempty(allcounts)
        allcounts = count1{:,3:end};
    else
        allcounts = vertcat(allcounts,count1{:,3:end});
    end
    eval(strcat('data',slideName{s},'=data1;'));
    toc;
end   

%% --Calculate LDA & plots (markers only, All slides)-----

maxT = 16;
tic;
lda1 = fitlda(allcounts,maxT);
toc;
figure,imagesc(lda1.TopicWordProbabilities);colormap(jet);
title ('Topic Word probabilities');
set(gca,'ytick',1:length(labelp3));
set(gca,'yticklabels',labelp3);
xlabel('Topics');
colorbar;
caxis([0 0.25]);

figure
for topicIdx = 1:maxT
    subplot(4,4,topicIdx)
    temp1 = table;
    temp1.Word = labelp3;
    
    temp1.Count = lda1.TopicWordProbabilities(:,topicIdx);
    wordcloud(temp1,'Word','Count');
    title("Topic: " + topicIdx)
end
toc;

%% Predict topics for all slides

tic;
maxL = 200;

for s=1:length(slideName)
    disp(strcat('Processing:',slideName{s}));
    data1 = eval(strcat('data',slideName{s}));
    
    count1 = varfun(@sum,data1,'GroupingVariables','frame','Inputvariables',labelp2);
    count1.topics = predict(ldaAll,count1{:,3:end});
    temp2 = count1(:,{'frame','topics'});
    data1 = join(data1,temp2,'keys','frame');
    %data1 = removevars(data1, {'col','row'});

    toc;
    eval(strcat('data',slideName{s},'=data1;'));
end

%% Bar plots (by samples)

colors = [1,0,0;1,0,0;1,1,0;0,1,1;0,1,0;0,1,0;0,0,1;0,0,1];
name1 = 'Pan_CK';
marker1 = strcat('mean_',name1,'p');
figure,b1=bar(sumSample{:,marker1}*100);
ytickformat('percentage');
b1.FaceColor = 'flat';

for i = 1:size(sumSample,1)
    b1.CData(i,:)=colors(i,:);
end
title(strcat(name1,'+'),'FontSize',16,'Interpreter','none');
set(gca,'xticklabels',sumSample.slideID);
set(gca,'XTickLabelRotation',45)

%% fix issue

for s=1:length(slideName)
    disp(strcat('Processing:',slideName{s}));
    data1 = eval(strcat('data',slideName{s}));

    data1 = removevars(data1, 'topics_data1');
    data1.Properties.VariableNames{106} = 'topics';

    eval(strcat('data',slideName{s},'=data1;'));
end

%% Find PD-L1 neighbor cells
k = 10;
marker1 = 'PD_1p';
marker2 = 'PD1nn';

tic;
for s=1:length(slideName)
    disp(strcat('Processing:',slideName{s}));
    data1 = eval(strcat('data',slideName{s}));

    data2 = data1(data1{:,marker1},:);
    list1 = knnsearch([data1.Xt,data1.Yt],[data2.Xt,data2.Yt],'K',k);
    list1 = list1(:);
    list1 = sortrows(list1);
    list1 = unique(list1);
    data1{:,marker2} = false(size(data1,1),1);
    data1{list1,marker2} = true;
    data1{data1{:,marker1},marker2} = false;

    eval(strcat('data',slideName{s},'=data1;'));
    toc;
end

%% Generate data

allPDL1nn = alldata(alldata.PD_L1nn,:);
allPD1nn = alldata(alldata.PD_1nn,:);
allMDM2nn = alldata(alldata.MDM2nn,:);

allPD1PDL1 = allPD1nn(allPD1nn.PD_L1p,:);
allPDL1PD1 = allPDL1nn(allPDL1nn.PD_1p,:);


%% TLS segmentation

tic;
for i =1:length(slideName)
    disp(strcat('Processing:',slideName{i}));
    data0 = eval(strcat('data',slideName{i}));
    
    data0.cellID = (1:size(data0,1))';
    data1 = data0(data0.CD20KNN>0,:);

    pc1 = pointCloud(double([data1.Xt,data1.Yt,ones(size(data1,1),1)]));
    label1=pcsegdist(pc1,100);
    data1.label1 = label1;

    table1 = tabulate(data1.label1);
    table1 = table1(table1(:,2)>200,:);
    table1(:,1) = (1:size(table1,1))';
    data1 = data1(ismember(data1.label1,table1(:,1)),:);
    

    data0.TLS = zeros(size(data0,1),1);
    data0.TLS(data1.cellID) = data1.label1;
    eval(strcat('data',slideName{i},'=data0;'));
    tabulate(data0.TLS);
    toc;
    %clear data0 data1;
end

%% PD1+ neighboring cells, enrichment

sum1 = sumPD1nn;
sum2 = sumAlldata;
sum3 = sum1(ismember(sum1.slideName,{'LSP13593','LSP13598'}),:);
sum4 = sum2(ismember(sum2.slideName,{'LSP13593','LSP13598'}),:);


figure,bar(sum3{:,strcat('mean_',labelp4)}'./sum4{:,strcat('mean_',labelp4)}');
%labelp5 = regexprep(labelp4,'p$','+');
set(gca,'xtick',1:length(labelp4));
set(gca,'xticklabels',labelp5);
ylabel('fold enrichment');
xlims = xlim;
line([xlims(1),xlims(2)],[1 1],'Color','k','LineStyle','--');
legend('Case 39','Case 53');
title('Marker enrichment in PD1+ neighboring cells');

%% Single marker+ dot plot (with %)

data1 = dataLSP13593;
samplename = 'Case 39';
name1 = 'NGFR';
marker1 = strcat(name1,'p');


figure,CycIF_tumorview(data1,marker1,2,100000);
daspect([1 1 1]);
set(gcf,'color','w');

m1 = mean(data1{:,marker1});
title(strcat(name1,{'+  '},num2str(m1*100,'%0.2f'),'%(',samplename,')'),'FontSize',20);


%% Plot PD1-PDL1 interaction

data1 = dataLSP13598;
samplename = 'Case 53';

figure,CycIF_tumorview(data1,'PD_L1p',2,100000);
hold on;
data2 = datasample(data1,100000);
%data2  = data1;
data2 = data2(data2.PD_1p,:);
scatter(data2.Xt,data2.Yt,3,'green','filled');

data3 = data1((data1.PD_1p & data1.PDL1nn) | (data1.PD_L1p & data1.PD1nn),:);
pr1 = sum((data1.PD_1p & data1.PDL1nn) | (data1.PD_L1p & data1.PD1nn))/sum(data1.PD_1p)*100;
dscatter(data3.Xt,data3.Yt,'PLOTTYPE','CONTOUR');
daspect([1 1 1]);
title(strcat(samplename,{' ('},num2str(pr1,'%0.2f'),'% PD1+)'),'FontSize',16);
set(gcf,'color','w');

legend('Other','PDL1+','PD1+','Interaction');


%% Plot PDL1 & CD20 colocalization

data1 = dataLSP13598;
samplename = 'Case 53';

figure,CycIF_tumorview(data1,'PD_L1p',2,200000);
hold on;
data2 = datasample(data1,200000);
data2 = data2(data2.CD20_2p,:);
scatter(data2.Xt,data2.Yt,3,'green','filled','MarkerFaceAlpha',0.25,'MarkerEdgeAlpha',0.25);

title(strcat('PDL1+ CD20+ colocalization (',samplename,')'),'FontSize',20);
set(gcf,'color','w');
legend off;
legend({'Other','PDL1+','CD20+'},'FontSize',16);
daspect([1 1 1]);

%% Bar graph (single marker)
marker1 = 'mean_PD_1p';
name1 = 'PD1';
sumTemp = sumAlldata;

figure;
b1=bar(sumTemp{:,marker1}*100);
b1.FaceColor = 'flat';
b1.CData(8,:)=[1,0,1];
b1.CData(9,:)=[1,0,1];

ytickformat('percentage');
set(gca,'xticklabels',sumTemp.Case_number);
title(strcat(name1,'+ cells'),'FontSize',16)

%% heatmap (alldata);

sumTemp = sumAlldata;

%clustergram(zscore(sumTemp{:,strcat('mean_',labelp4)}),'cluster','all','colormap',redbluecmap);
figure,imagesc(zscore(sumTemp{:,strcat('mean_',labelp4)}));
colormap(redbluecmap);
cb1=colorbar;
title(cb1,'Z score')
set(gca,'xtick',1:length(labelp4));
%labelp5 = regexprep(labelp4,'p$','+');
set(gca,'xticklabels',labelp5);
set(gca,'yticklabels',sumTemp.Case_number);

%% Clustergram

sumTemp = sumAlldata;
%labelp5 = regexprep(labelp4,'p$','+');
clustergram(zscore(sumTemp{:,strcat('mean_',labelp4)}),'cluster','all','colormap',redbluecmap,'Rowlabels',sumTemp.Case_number,'ColumnLabels',labelp5,'DisplayRatio',0.1);


%% Bar graph (single marker)

%marker1 = 'mean_PD_1p';
%name1 = 'PD1';
sumTemp = sumAlldata;

figure;
b1=bar(sumTemp.mean_CD8ap./sumTemp.mean_FOXP3p);
b1.FaceColor = 'flat';
b1.CData(8,:)=[1,0,1];
b1.CData(9,:)=[1,0,1];

%ytickformat('percentage');
set(gca,'xticklabels',sumTemp.Case_number);
title('CD8/FOXP3+ ratio','FontSize',16)

%% Bar graph (PD1+PDL1)

sumTemp = sumAlldata;
list1 = sumTemp.mean_PD_1p;
list2 = sumTemp.mean_PD_L1p;

figure;
b1=bar(horzcat(list1,list2)*100);
legend('PD1+','PDL1+');
legend('box','off')


ytickformat('percentage');
set(gca,'xticklabels',sumTemp.Case_number);
title('PD1+ & PDL1+ cells','FontSize',16)

