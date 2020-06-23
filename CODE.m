%%                 =|>- MIAMI POWER ELECTRONICS LABORATORY               %%
%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  %
%  MATLAB CODE FOR HIDDEN MARKOV MODEL-ADDED MACHINE LEARNING
%  FOR DC-LINK CAPACITOR AGE CLASSIFICATION 
%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  %
%  AUTHOR                                                            DATE %
%  Viktoriia Sysoeva                                           05/21/2020 %
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - %

%% CAPACITOR CLASSES     INPUT DATASETS
% - 1 - 6 New 0 old           X1
% - 2 - 5 New 1 Old           X2
% - 3 - 3 New 3 Old           X3
% - 4 - 1 New 5 Old           X4
% - 5 - 0 New 6 Old           X5
clear; close all; clc;

%% STAGE 1. PREPROCESSING

%% 1.1. Input Data & Processing Specification

nC = 5;               % Number of Classes
tSample = 1e-8;       % Sampling period, s

%Parameters for data that requires splitting
nAXS = 2;             % Number of acquired signals for each class
split = 100;          % Number of signals to be extracted from 1 acquired signal
AXSlength = 10000000; % Length of acquired signal, samples
tXS = 100e-03;        % Signal record length, s

%Parameters for data that does not require splitting
nAXT = 200;           % Number of acquired signals for each class
XTlength = 100000;    % Length of acquired signal, samples
tX = 1e-03;           % Signal record length, s

%% 1.2. Parameters Computation 

%Parameters for data that requires splitting
XSlength = AXSlength/split;  % Length of extracted signal 
sSplit = nAXS*split;         % Number of extracted signals for each class
nXS = nC*sSplit;             % Total Number of extracted signals for all classes

%Parameters for data that does not require splitting
nXT = nC*nAXT;               % Total Number of acquired signals for all classes

%Parameters for dataset X
s = sSplit+nAXT;             % Number of signals for each class
s_data = nXS+nXT;            % Total number of signals
time = 0:tSample:tX-tSample; % Time vector for 1 signal
fs = 1./diff(time);          
Fs = fs(1);                  % Sampling Rate

%% 1.3. Datasets XS of split data

XS1=Dsplit('X1/',XSlength,sSplit,split);
XS2=Dsplit('X2/',XSlength,sSplit,split);
XS3=Dsplit('X3/',XSlength,sSplit,split);
XS4=Dsplit('X4/',XSlength,sSplit,split);
XS5=Dsplit('X5/',XSlength,sSplit,split);

%% 1.4. Datasets XT of not split data

XT1=Dread('X1/',XTlength,nAXT);
XT2=Dread('X2/',XTlength,nAXT);
XT3=Dread('X3/',XTlength,nAXT);
XT4=Dread('X4/',XTlength,nAXT);
XT5=Dread('X5/',XTlength,nAXT);

%% 1.5. Dataset of type X - time-domain data

X1(:,1:sSplit)=XS1;
X2(:,1:sSplit)=XS2;
X3(:,1:sSplit)=XS3;
X4(:,1:sSplit)=XS4;
X5(:,1:sSplit)=XS5;

X1(:,sSplit+1:s)=XT1;
X2(:,sSplit+1:s)=XT2;
X3(:,sSplit+1:s)=XT3;
X4(:,sSplit+1:s)=XT4;
X5(:,sSplit+1:s)=XT5;

% Plot Data X
figure
plot(time,X1(:,1),'k');
hold on
plot(time,X2(:,1),'r');
hold on
plot(time,X3(:,1),'g');
hold on
plot(time,X4(:,1),'m');
hold on
plot(time,X5(:,1),'c');
title('Data X, Experimental Results');
xlabel('Time (s)')
ylabel('Amplitude (A)')
legend({'N^{6}A^{0}','N^{5}A^{1}','N^{3}A^{3}','N^{1}A^{5}',...
'N^{0}A^{6}'},'Location','northeast')
hold off

%% 1.6. Datasets of type F, MAR, MAH, LR, SG

% F - single-sided FFT Spectrum
% MAR - Spectrum smoothed with moving average, rectangular window
% MAH - Spectrum smoothed with moving average, Hanning window
% LR  - Spectrum smoothed with locally weighted linear regression
% SG - Spectrum smoothed with Savitzky-Golay filter

w = 1000;   %window length for smoothing filters
SL = 50000; %signal length

% Class 1 
F_1=zeros(SL,s);
MAR_1=zeros(SL,s);
MAH_1=zeros(SL,s);
LR_1=zeros(SL,s);
SG_1=zeros(SL,s);

for i=1:s
    Y=X1(:,i);
    [f,F]=Spec(Y,Fs);  
    [MAR,MAH,LR,SG]=SpecSm(F,w,SL);
    F_1(:,i)=F;
    MAR_1(:,i)=MAR;
    MAH_1(:,i)=MAH;
    LR_1(:,i)=LR;    
    SG_1(:,i)=SG;
end

AvF_1=mean(F_1,2);
AvMAR_1=mean(MAR_1,2);
AvMAH_1=mean(MAH_1,2);
AvLR_1=mean(LR_1,2);
AvSG_1=mean(SG_1,2);

% Class 2
F_2=zeros(SL,s);
MAR_2=zeros(SL,s);
MAH_2=zeros(SL,s);
LR_2=zeros(SL,s);
SG_2=zeros(SL,s);

for i=1:s
    Y=X2(:,i);
    [f,F]=Spec(Y,Fs);  
    [MAR,MAH,LR,SG]=SpecSm(F,w,SL);
    F_2(:,i)=F;
    MAR_2(:,i)=MAR;
    MAH_2(:,i)=MAH;
    LR_2(:,i)=LR;    
    SG_2(:,i)=SG;
end

AvF_2=mean(F_2,2);
AvMAR_2=mean(MAR_2,2);
AvMAH_2=mean(MAH_2,2);
AvLR_2=mean(LR_2,2);
AvSG_2=mean(SG_2,2);

% Class 3
F_3=zeros(SL,s);
MAR_3=zeros(SL,s);
MAH_3=zeros(SL,s);
LR_3=zeros(SL,s);
SG_3=zeros(SL,s);

for i=1:s
    Y=X3(:,i);
    [f,F]=Spec(Y,Fs);  
    [MAR,MAH,LR,SG]=SpecSm(F,w,SL);
    F_3(:,i)=F;
    MAR_3(:,i)=MAR;
    MAH_3(:,i)=MAH;
    LR_3(:,i)=LR;    
    SG_3(:,i)=SG;
end

AvF_3=mean(F_3,2);
AvMAR_3=mean(MAR_3,2);
AvMAH_3=mean(MAH_3,2);
AvLR_3=mean(LR_3,2);
AvSG_3=mean(SG_3,2);

% Class 4
F_4=zeros(SL,s);
MAR_4=zeros(SL,s);
MAH_4=zeros(SL,s);
LR_4=zeros(SL,s);
SG_4=zeros(SL,s);

for i=1:s
    Y=X4(:,i);
    [f,F]=Spec(Y,Fs);  
    [MAR,MAH,LR,SG]=SpecSm(F,w,SL);
    F_4(:,i)=F;
    MAR_4(:,i)=MAR;
    MAH_4(:,i)=MAH;
    LR_4(:,i)=LR;    
    SG_4(:,i)=SG;
end

AvF_4=mean(F_4,2);
AvMAR_4=mean(MAR_4,2);
AvMAH_4=mean(MAH_4,2);
AvLR_4=mean(LR_4,2);
AvSG_4=mean(SG_4,2);

% Class 5
F_5=zeros(SL,s);
MAR_5=zeros(SL,s);
MAH_5=zeros(SL,s);
LR_5=zeros(SL,s);
SG_5=zeros(SL,s);

for i=1:s
    Y=X5(:,i);
    [f,F]=Spec(Y,Fs);  
    [MAR,MAH,LR,SG]=SpecSm(F,w,SL);
    F_5(:,i)=F;
    MAR_5(:,i)=MAR;
    MAH_5(:,i)=MAH;
    LR_5(:,i)=LR;    
    SG_5(:,i)=SG;
end

AvF_5=mean(F_5,2);
AvMAR_5=mean(MAR_5,2);
AvMAH_5=mean(MAH_5,2);
AvLR_5=mean(LR_5,2);
AvSG_5=mean(SG_5,2);

% Plot Data F
figure
plot(f,AvF_1,'k');
hold on
plot(f,AvF_2,'r');
hold on
plot(f,AvF_3,'g');
hold on
plot(f,AvF_4,'m');
hold on
plot(f,AvF_5,'c');
title('Data F, Experimental Results');
xlabel('Frequency (Hz)')
ylabel('Amplitude (dBuA)')
legend({'N^{6}A^{0}','N^{5}A^{1}','N^{3}A^{3}','N^{1}A^{5}',...
'N^{0}A^{6}'},'Location','northeast')
hold off

% Plot Data MAR
figure
plot(f,AvMAR_1,'k');
hold on
plot(f,AvMAR_2,'r');
hold on
plot(f,AvMAR_3,'g');
hold on
plot(f,AvMAR_4,'m');
hold on
plot(f,AvMAR_5,'c');
title('Data MAR, Experimental Results');
xlabel('Frequency (Hz)')
ylabel('Amplitude (dBuA)')
legend({'N^{6}A^{0}','N^{5}A^{1}','N^{3}A^{3}','N^{1}A^{5}',...
'N^{0}A^{6}'},'Location','northeast')
hold off

% Plot Data MAH
figure
plot(f,AvMAH_1,'k');
hold on
plot(f,AvMAH_2,'r');
hold on
plot(f,AvMAH_3,'g');
hold on
plot(f,AvMAH_4,'m');
hold on
plot(f,AvMAH_5,'c');
title('Data MAH, Experimental Results');
xlabel('Frequency (Hz)')
ylabel('Amplitude (dBuA)')
legend({'N^{6}A^{0}','N^{5}A^{1}','N^{3}A^{3}','N^{1}A^{5}',...
'N^{0}A^{6}'},'Location','northeast')
hold off

% Plot Data LR
figure
plot(f,AvLR_1,'k');
hold on
plot(f,AvLR_2,'r');
hold on
plot(f,AvLR_3,'g');
hold on
plot(f,AvLR_4,'m');
hold on
plot(f,AvLR_5,'c');
title('Data LR, Experimental Results');
xlabel('Frequency (Hz)')
ylabel('Amplitude (dBuA)')
legend({'N^{6}A^{0}','N^{5}A^{1}','N^{3}A^{3}','N^{1}A^{5}',...
'N^{0}A^{6}'},'Location','northeast')
hold off

% Plot Data SG
figure
plot(f,AvSG_1,'k');
hold on
plot(f,AvSG_2,'r');
hold on
plot(f,AvSG_3,'g');
hold on
plot(f,AvSG_4,'m');
hold on
plot(f,AvSG_5,'c');
title('Data SG, Experimental Results');
xlabel('Frequency (Hz)')
ylabel('Amplitude (dBuA)')
legend({'N^{6}A^{0}','N^{5}A^{1}','N^{3}A^{3}','N^{1}A^{5}',...
'N^{0}A^{6}'},'Location','northeast')
hold off

%% 1.7. Datasets for ANN

% Matrices of features
ANN_X=zeros(XTlength,s_data);
ANN_F=zeros(SL,s_data);
ANN_MAR=zeros(SL,s_data);
ANN_MAH=zeros(SL,s_data);
ANN_LR=zeros(SL,s_data);
ANN_SG=zeros(SL,s_data);

ANN_X=ANNfeatures(X1,X2,X3,X4,X5,s);
ANN_F=ANNfeatures(F_1,F_2,F_3,F_4,F_5,s);
ANN_MAR=ANNfeatures(MAR_1,MAR_2,MAR_3,MAR_4,MAR_5,s);
ANN_MAH=ANNfeatures(MAH_1,MAH_2,MAH_3,MAH_4,MAH_5,s);
ANN_LR=ANNfeatures(LR_1,LR_2,LR_3,LR_4,LR_5,s);
ANN_SG=ANNfeatures(SG_1,SG_2,SG_3,SG_4,SG_5,s);

% Matrix of targets
ANN_Targets = zeros(nC,s_data);
for i=1:nC
ANN_Targets(i,((i-1)*s+1):(i*s)) = 1;
end

%% 1.8. Datasets for SVM

% Matrices of features
SVM_X=ANN_X.';
SVM_F=ANN_F.';
SVM_MAR=ANN_MAR.';
SVM_MAH=ANN_MAH.';
SVM_LR=ANN_LR.';
SVM_SG=ANN_SG.';

% Matrix of targets
SVM_Targets = zeros(s_data,1);
for i=1:nC
SVM_Targets((s*(i-1))+1:s*i)=i;
end

%% STAGE 2. MACHINE LEARNING CLASSIFICATION

%% 2.1. SVM
% SVM Parameters
total_trial=100; % total number of SVM trials
model=templateSVM('KernelFunction','linear','KernelScale','auto',...
'Standardize',true,'BoxConstraint',25);

% Training and testing division
N_train=[51:200, 351:400, 451:600, 751:800, 851:1000, 1151:1200,...
         1251:1400, 1551:1600, 1651:1800, 1951:2000];
N_test=[201:350, 1:50, 601:750, 401:450, 1001:1150, 801:850,...
        1401:1550, 1201:1250, 1801:1950, 1601:1650];

%% SVM for data type X
[SVM_OUT_X, SVM_Targets_test,...
SVM_accuracy_X, avg_SVM_accuracy_X, max_SVM_accuracy_X]=...
SVMclassification(SVM_X, SVM_Targets, total_trial, N_train, N_test, model);

% Print SVM accuracy
fprintf ('\n Average SVM accuracy, data type X: %f \n', avg_SVM_accuracy_X)
fprintf ('\n Maximum SVM accuracy, data type X: %f \n', max_SVM_accuracy_X)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_Targets_test,SVM_OUT_X(num_trial,:));
Testconfusion.Title = 'X Confusion Matrix';
end

%% SVM for data type F
[SVM_OUT_F, SVM_Targets_test,...
SVM_accuracy_F, avg_SVM_accuracy_F, max_SVM_accuracy_F]=...
SVMclassification(SVM_F, SVM_Targets, total_trial, N_train, N_test, model);

% Print SVM accuracy
fprintf ('\n Average SVM accuracy, data type F: %f \n', avg_SVM_accuracy_F)
fprintf ('\n Maximum SVM accuracy, data type F: %f \n', max_SVM_accuracy_F)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_Targets_test,SVM_OUT_F(num_trial,:));
Testconfusion.Title = 'F Confusion Matrix';
end

%% SVM for data type MAR
[SVM_OUT_MAR, SVM_Targets_test,...
SVM_accuracy_MAR ,avg_SVM_accuracy_MAR, max_SVM_accuracy_MAR]=...
SVMclassification(SVM_MAR, SVM_Targets, total_trial, N_train, N_test, model);

% Print SVM accuracy
fprintf ('\n Average SVM accuracy, data type MAR: %f \n', avg_SVM_accuracy_MAR)
fprintf ('\n Maximum SVM accuracy, data type MAR: %f \n', max_SVM_accuracy_MAR)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_Targets_test,SVM_OUT_MAR(num_trial,:));
Testconfusion.Title = 'MAR Confusion Matrix';
end

%% SVM for data type MAH
[SVM_OUT_MAH, SVM_Targets_test,...
SVM_accuracy_MAH, avg_SVM_accuracy_MAH, max_SVM_accuracy_MAH]=...
SVMclassification(SVM_MAH, SVM_Targets, total_trial, N_train, N_test, model);

% Print SVM accuracy
fprintf ('\n Average SVM accuracy, data type MAH: %f \n', avg_SVM_accuracy_MAH)
fprintf ('\n Maximum SVM accuracy, data type MAH: %f \n', max_SVM_accuracy_MAH)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_Targets_test,SVM_OUT_MAH(num_trial,:));
Testconfusion.Title = 'MAH Confusion Matrix';
end

%% SVM for data type LR
[SVM_OUT_LR, SVM_Targets_test,...
SVM_accuracy_LR ,avg_SVM_accuracy_LR, max_SVM_accuracy_LR]=...
SVMclassification(SVM_LR, SVM_Targets, total_trial, N_train, N_test, model);

% Print SVM accuracy
fprintf ('\n Average SVM accuracy, data type LR: %f \n', avg_SVM_accuracy_LR)
fprintf ('\n Maximum SVM accuracy, data type LR: %f \n', max_SVM_accuracy_LR)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_Targets_test,SVM_OUT_LR(num_trial,:));
Testconfusion.Title = 'LR Confusion Matrix';
end

%% SVM for data type SG
[SVM_OUT_SG, SVM_Targets_test,...
SVM_accuracy_SG ,avg_SVM_accuracy_SG, max_SVM_accuracy_SG]=...
SVMclassification(SVM_SG, SVM_Targets, total_trial, N_train, N_test, model);

% Print SVM accuracy
fprintf ('\n Average SVM accuracy, data type SG: %f \n', avg_SVM_accuracy_SG)
fprintf ('\n Maximum SVM accuracy, data type SG: %f \n', max_SVM_accuracy_SG)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_Targets_test,SVM_OUT_SG(num_trial,:));
Testconfusion.Title = 'SG Confusion Matrix';
end

%% 2.2. ANN
% ANN parameters
total_trial=2;      % total number of ANN trials
hiddenLayerSize=40;   % number of hidden neurons
trainFcn='trainscg';  % training function
net=patternnet(hiddenLayerSize, trainFcn); % ANN model
% input/output processing functions
net.input.processFcns={'removeconstantrows','mapminmax'}; 

% Training, validation, and testing division
net.divideFcn='divideind';  % divide data by index
net.divideMode='sample';    % divide samples
N_train=[51:140, 351:400, 451:540, 751:800, 851:940,...
    1151:1200, 1251:1340, 1551:1600, 1651:1740, 1951:2000];
N_valid=[141:200, 541:600, 941:1000, 1341:1400, 1741:1800];
N_test=[201:350, 1:50, 601:750, 401:450, 1001:1150,...
    801:850, 1401:1550, 1201:1250, 1801:1950, 1601:1650];

%% ANN for data type X
[ANN_OUT_X,ANN_Targets_test,...
ANN_accuracy_X,avg_ANN_accuracy_X, max_ANN_accuracy_X]=...
ANNclassification(ANN_X,ANN_Targets,total_trial,N_train, N_valid, N_test, net);

% Print ANN accuracy
fprintf ('\n Average ANN accuracy, data type X: %f \n', avg_ANN_accuracy_X)
fprintf ('\n Maximum ANN accuracy, data type X: %f \n', max_ANN_accuracy_X)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_Targets_test,ANN_OUT_X(num_trial,:));
Testconfusion.Title = 'X Confusion Matrix';
end

%% ANN for data type F
[ANN_OUT_F,ANN_Targets_test,...
ANN_accuracy_F,avg_ANN_accuracy_F, max_ANN_accuracy_F]=...
ANNclassification(ANN_F,ANN_Targets,total_trial,N_train, N_valid, N_test, net);

% Print ANN accuracy
fprintf ('\n Average ANN accuracy, data type F: %f \n', avg_ANN_accuracy_F)
fprintf ('\n Maximum ANN accuracy, data type F: %f \n', max_ANN_accuracy_F)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_Targets_test,ANN_OUT_F(num_trial,:));
Testconfusion.Title = 'F Confusion Matrix';
end

%% ANN for data type MAR
[ANN_OUT_MAR,ANN_Targets_test,...
ANN_accuracy_MAR,avg_ANN_accuracy_MAR, max_ANN_accuracy_MAR]=...
ANNclassification(ANN_MAR,ANN_Targets,total_trial,N_train, N_valid, N_test, net);

% Print ANN accuracy
fprintf ('\n Average ANN accuracy, data type MAR: %f \n', avg_ANN_accuracy_MAR)
fprintf ('\n Maximum ANN accuracy, data type MAR: %f \n', max_ANN_accuracy_MAR)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_Targets_test,ANN_OUT_MAR(num_trial,:));
Testconfusion.Title = 'MAR Confusion Matrix';
end

%% ANN for data type MAH
[ANN_OUT_MAH,ANN_Targets_test,...
ANN_accuracy_MAH,avg_ANN_accuracy_MAH, max_ANN_accuracy_MAH]=...
ANNclassification(ANN_MAH,ANN_Targets,total_trial,N_train, N_valid, N_test, net);

% Print ANN accuracy
fprintf ('\n Average ANN accuracy, data type MAH: %f \n', avg_ANN_accuracy_MAH)
fprintf ('\n Maximum ANN accuracy, data type MAH: %f \n', max_ANN_accuracy_MAH)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_Targets_test,ANN_OUT_MAH(num_trial,:));
Testconfusion.Title = 'MAH Confusion Matrix';
end

%% ANN for data type LR
[ANN_OUT_LR,ANN_Targets_test,...
ANN_accuracy_LR,avg_ANN_accuracy_LR, max_ANN_accuracy_LR]=...
ANNclassification(ANN_LR,ANN_Targets,total_trial,N_train, N_valid, N_test, net);

% Print ANN accuracy
fprintf ('\n Average ANN accuracy, data type LR: %f \n', avg_ANN_accuracy_LR)
fprintf ('\n Maximum ANN accuracy, data type LR: %f \n', max_ANN_accuracy_LR)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_Targets_test,ANN_OUT_LR(num_trial,:));
Testconfusion.Title = 'LR Confusion Matrix';
end

%% ANN for data type SG
[ANN_OUT_SG,ANN_Targets_test,...
ANN_accuracy_SG,avg_ANN_accuracy_SG, max_ANN_accuracy_SG]=...
ANNclassification(ANN_SG,ANN_Targets,total_trial,N_train, N_valid, N_test, net);

% Print ANN accuracy
fprintf ('\n Average ANN accuracy, data type SG: %f \n', avg_ANN_accuracy_SG)
fprintf ('\n Maximum ANN accuracy, data type SG: %f \n', max_ANN_accuracy_SG)

% Plot confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_Targets_test,ANN_OUT_SG(num_trial,:));
Testconfusion.Title = 'SG Confusion Matrix';
end

%% STAGE 3. HIDDEN MARKOV MODEL-BASED OUTPUT CORRECTION
%% 3.1. SVM ouput correction
total_trial=100;
%State sequence
SVM_state_seq=flip(SVM_Targets_test.');

%% SVM output correction for data type MAR
% Observation sequence
SVM_obs_seq=flip(SVM_OUT_MAR,2);

[SVM_OUT_MAR_corrected,SVM_accuracy_MAR_corrected,avg_SVM_accuracy_MAR_corrected,...
max_SVM_accuracy_MAR_corrected]=HHM_SVM_corr(SVM_obs_seq,SVM_state_seq,total_trial);

% Print corrected accuracy
fprintf ('\n Average corrected SVM accuracy, data type MAR: %f \n', avg_SVM_accuracy_MAR_corrected);
fprintf ('\n Maximum corrected SVM accuracy, data type MAR: %f \n', max_SVM_accuracy_MAR_corrected);

% Plot original and corrected accuracy
figure
plot(1:total_trial,SVM_accuracy_MAR,'k',1:total_trial,SVM_accuracy_MAR_corrected,'m')
ylim([0.8 1])
title('MAR Average Accuracy');
xlabel('Trial');
ylabel('Accuracy')
legend({'Original','Corrected'},'Location','southeast')

% Plot corrected confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_state_seq,SVM_OUT_MAR_corrected(num_trial,:));
Testconfusion.Title = 'Corrected MAR Confusion Matrix';
end

%% SVM output correction for data type MAH
% Observation sequence
SVM_obs_seq=flip(SVM_OUT_MAH,2);

[SVM_OUT_MAH_corrected,SVM_accuracy_MAH_corrected,avg_SVM_accuracy_MAH_corrected,...
max_SVM_accuracy_MAH_corrected]=HHM_SVM_corr(SVM_obs_seq,SVM_state_seq,total_trial);

% Print corrected accuracy
fprintf ('\n Average corrected SVM accuracy, data type MAH: %f \n', avg_SVM_accuracy_MAH_corrected);
fprintf ('\n Maximum corrected SVM accuracy, data type MAH: %f \n', max_SVM_accuracy_MAH_corrected);

% Plot original and corrected accuracy
figure
plot(1:total_trial,SVM_accuracy_MAH,'k',1:total_trial,SVM_accuracy_MAH_corrected,'m')
ylim([0.8 1])
title('MAH Average Accuracy');
xlabel('Trial');
ylabel('Accuracy')
legend({'Original','Corrected'},'Location','southeast')

% Plot corrected confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_state_seq,SVM_OUT_MAH_corrected(num_trial,:));
Testconfusion.Title = 'Corrected MAH Confusion Matrix';
end

%% SVM output correction for data type LR
% Observation sequence
SVM_obs_seq=flip(SVM_OUT_LR,2);

[SVM_OUT_LR_corrected,SVM_accuracy_LR_corrected,avg_SVM_accuracy_LR_corrected,...
max_SVM_accuracy_LR_corrected]=HHM_SVM_corr(SVM_obs_seq,SVM_state_seq,total_trial);

% Print corrected accuracy
fprintf ('\n Average corrected SVM accuracy, data type LR: %f \n', avg_SVM_accuracy_LR_corrected);
fprintf ('\n Maximum corrected SVM accuracy, data type LR: %f \n', max_SVM_accuracy_LR_corrected);

% Plot original and corrected accuracy
figure
plot(1:total_trial,SVM_accuracy_LR,'k',1:total_trial,SVM_accuracy_LR_corrected,'m')
ylim([0.8 1])
title('LR Average Accuracy');
xlabel('Trial');
ylabel('Accuracy')
legend({'Original','Corrected'},'Location','southeast')

% Plot corrected confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_state_seq,SVM_OUT_LR_corrected(num_trial,:));
Testconfusion.Title = 'Corrected LR Confusion Matrix';
end

%% SVM output correction for data type SG
% Observation sequence
SVM_obs_seq=flip(SVM_OUT_SG,2);

[SVM_OUT_SG_corrected,SVM_accuracy_SG_corrected,avg_SVM_accuracy_SG_corrected,...
max_SVM_accuracy_SG_corrected]=HHM_SVM_corr(SVM_obs_seq,SVM_state_seq,total_trial);

% Print corrected accuracy
fprintf ('\n Average corrected SVM accuracy, data type SG: %f \n', avg_SVM_accuracy_SG_corrected);
fprintf ('\n Maximum corrected SVM accuracy, data type SG: %f \n', max_SVM_accuracy_SG_corrected);

% Plot original and corrected accuracy
figure
plot(1:total_trial,SVM_accuracy_SG,'k',1:total_trial,SVM_accuracy_SG_corrected,'m')
ylim([0.8 1])
title('SG Average Accuracy');
xlabel('Trial');
ylabel('Accuracy')
legend({'Original','Corrected'},'Location','southeast')

% Plot corrected confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(SVM_state_seq,SVM_OUT_SG_corrected(num_trial,:));
Testconfusion.Title = 'Corrected SG Confusion Matrix';
end

%% 3.2. ANN ouput correction
total_trial=100;
%State sequence
ANN_state_seq=flip(ANN_Targets_test);

%% ANN output correction for data type MAR
% Observation sequence
ANN_obs_seq=flip(ANN_OUT_MAR,2);

[ANN_OUT_MAR_corrected,ANN_accuracy_MAR_corrected,avg_ANN_accuracy_MAR_corrected,...
max_ANN_accuracy_MAR_corrected]=HHM_ANN_corr(ANN_obs_seq,ANN_state_seq,total_trial);

% Print corrected accuracy
fprintf ('\n Average corrected ANN accuracy, data type MAR: %f \n', avg_ANN_accuracy_MAR_corrected);
fprintf ('\n Maximum corrected ANN accuracy, data type MAR: %f \n', max_ANN_accuracy_MAR_corrected);

% Plot original and corrected accuracy
figure
plot(1:total_trial,ANN_accuracy_MAR,'k',1:total_trial,ANN_accuracy_MAR_corrected,'m')
ylim([0.8 1])
title('MAR Average Accuracy');
xlabel('Trial');
ylabel('Accuracy')
legend({'Original','Corrected'},'Location','southeast')

% Plot corrected confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_state_seq,ANN_OUT_MAR_corrected(num_trial,:));
Testconfusion.Title = 'Corrected MAR Confusion Matrix';
end

%% ANN output correction for data type MAH
% Observation sequence
ANN_obs_seq=flip(ANN_OUT_MAH,2);

[ANN_OUT_MAH_corrected,ANN_accuracy_MAH_corrected,avg_ANN_accuracy_MAH_corrected,...
max_ANN_accuracy_MAH_corrected]=HHM_ANN_corr(ANN_obs_seq,ANN_state_seq,total_trial);

% Print corrected accuracy
fprintf ('\n Average corrected ANN accuracy, data type MAH: %f \n', avg_ANN_accuracy_MAH_corrected);
fprintf ('\n Maximum corrected ANN accuracy, data type MAH: %f \n', max_ANN_accuracy_MAH_corrected);

% Plot original and corrected accuracy
figure
plot(1:total_trial,ANN_accuracy_MAH,'k',1:total_trial,ANN_accuracy_MAH_corrected,'m')
ylim([0.8 1])
title('MAH Average Accuracy');
xlabel('Trial');
ylabel('Accuracy')
legend({'Original','Corrected'},'Location','southeast')

% Plot corrected confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_state_seq,ANN_OUT_MAH_corrected(num_trial,:));
Testconfusion.Title = 'Corrected MAH Confusion Matrix';
end

%% ANN output correction for data type LR
% Observation sequence
ANN_obs_seq=flip(ANN_OUT_LR,2);

[ANN_OUT_LR_corrected,ANN_accuracy_LR_corrected,avg_ANN_accuracy_LR_corrected,...
max_ANN_accuracy_LR_corrected]=HHM_ANN_corr(ANN_obs_seq,ANN_state_seq,total_trial);

% Print corrected accuracy
fprintf ('\n Average corrected ANN accuracy, data type LR: %f \n', avg_ANN_accuracy_LR_corrected);
fprintf ('\n Maximum corrected ANN accuracy, data type LR: %f \n', max_ANN_accuracy_LR_corrected);

% Plot original and corrected accuracy
figure
plot(1:total_trial,ANN_accuracy_LR,'k',1:total_trial,ANN_accuracy_LR_corrected,'m')
ylim([0.8 1])
title('LR Average Accuracy');
xlabel('Trial');
ylabel('Accuracy')
legend({'Original','Corrected'},'Location','southeast')

% Plot corrected confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_state_seq,ANN_OUT_LR_corrected(num_trial,:));
Testconfusion.Title = 'Corrected LR Confusion Matrix';
end

%% ANN output correction for data type SG
% Observation sequence
ANN_obs_seq=flip(ANN_OUT_SG,2);

[ANN_OUT_SG_corrected,ANN_accuracy_SG_corrected,avg_ANN_accuracy_SG_corrected,...
max_ANN_accuracy_SG_corrected]=HHM_ANN_corr(ANN_obs_seq,ANN_state_seq,total_trial);

% Print corrected accuracy
fprintf ('\n Average corrected ANN accuracy, data type SG: %f \n', avg_ANN_accuracy_SG_corrected);
fprintf ('\n Maximum corrected ANN accuracy, data type SG: %f \n', max_ANN_accuracy_SG_corrected);

% Plot original and corrected accuracy
figure
plot(1:total_trial,ANN_accuracy_SG,'k',1:total_trial,ANN_accuracy_SG_corrected,'m')
ylim([0.8 1])
title('SG Average Accuracy');
xlabel('Trial');
ylabel('Accuracy')
legend({'Original','Corrected'},'Location','southeast')

% Plot corrected confusion matrix
for num_trial=1:total_trial
figure
Testconfusion = confusionchart(ANN_state_seq,ANN_OUT_SG_corrected(num_trial,:));
Testconfusion.Title = 'Corrected SG Confusion Matrix';
end

%% >>>>>>>>>>>>>>> FUNCTIONS  <<<<<<<<<<<<<<<< %%

%% Read CT data
function [M] = readCS(numberFile,localDirPath)
if numberFile(1)>=0 &&  numberFile(1) < 10 
    folder = ([localDirPath ]);
    baseFileName =([ 'tek000' num2str( numberFile,'%d') 'CH1.csv']);
elseif numberFile(1)>=10 && numberFile(1) < 100
    folder = ([localDirPath ]);
    baseFileName =([ 'tek00' int2str(numberFile) 'CH1.csv']);
elseif numberFile(1)>=100 && numberFile(1) < 1000
    folder = ([localDirPath ]);
    baseFileName =([ 'tek0' int2str(numberFile) 'CH1.csv']);
else
folder = ([localDirPath ]);
    baseFileName =([ 'tek' int2str(numberFile) 'CH1.csv']);
end 
fullFileName = fullfile(folder, baseFileName);    
if exist(fullFileName,'file')
  M = csvread(fullFileName,21,0);
else
  warningMessage = sprintf('%s does not exist', fullFileName);
  uiwait(warndlg(warningMessage));
end
end

%% Splitting - Dataset XS
function Dout=Dsplit(Din,xs,sSplit,split)
Dout=zeros(xs,sSplit); 
for numberFile=1:2
NF=numberFile-1;
    [M] = readCS(NF,Din);
    X1 = M(:,2);
    for i=1:split
    Dout(:,i+NF*split) = X1((i-1)*xs+1:i*xs);
    end
end
end

%% Dataset XT
function Dout=Dread(Din,Xlength,s)
Dout=zeros(Xlength,s); 
for numberFile = 2:201   
    [M] = readCS(numberFile,Din);
    X = M(:,2);
    Dout(:,numberFile-1) = X;
end
end

%% Perform FFT
function [f1,OUT]=Spec(IN,FS)
f1=round(0:FS/length(IN):FS/2);
xdft1=20*log10(abs(fft(IN)/length(IN))/1E-6)+120;
OUT=xdft1(1:length(IN)/2+1);
OUT=OUT(1:50000,1);
f1=f1(1,1:50000);
end

%% Spectral smoothing
function [MAR,MAH,LR,SG]=SpecSm(Y,w,SL)
MAR = smoothdata(Y,'movmean',w);
LR = smoothdata(Y,'lowess',w);
SG = smoothdata(Y,'sgolay',w);
wn = hann(w);
A=zeros(SL+w,1);
A(w/2+1:SL+w/2,1)=Y;
A(1:w/2)=Y(1);
A(SL+w/2+1:SL+w)=Y(SL);
MAH = conv(A, wn, 'valid')./5e2;
MAH=MAH(1:SL);
end

%% ANN matrix of features
function Dout=ANNfeatures(D1,D2,D3,D4,D5,s)
Dout(:,s*0+1:s*1)=D1;
Dout(:,s*1+1:s*2)=D2;
Dout(:,s*2+1:s*3)=D3;
Dout(:,s*3+1:s*4)=D4;
Dout(:,s*4+1:s*5)=D5;
end

%% SVM classification
function [SVM_OUT,SVM_Targets_test,SVM_accuracy,avg_SVM_accuracy, max_SVM_accuracy]=SVMclassification(SVM_Features,SVM_Targets,total_trial,N_train, N_test,model)
SVM_Features_train=SVM_Features(N_train,:);
SVM_Features_test=SVM_Features(N_test,:);
SVM_Targets_train=SVM_Targets(N_train,:);
SVM_Targets_test=SVM_Targets(N_test,:);
for num_trial=1:total_trial
% SVM Training 
SVMmodel=fitcecoc(SVM_Features_train,SVM_Targets_train,'Learners',model);
% SVM Validation
CVmodel=crossval(SVMmodel,'Holdout',0.3);
% SVM Testing
TrainedModel=CVmodel.Trained{1}; 
[Labels_predict,Score_predict]=predict(TrainedModel,SVM_Features_test);
% SVM output
SVM_OUT(num_trial,:)=Labels_predict.';
% SVM accuracy
SVM_accuracy(num_trial)=sum(Labels_predict==SVM_Targets_test)/length(Labels_predict);
end
% Average SVM accuracy
avg_SVM_accuracy=mean(SVM_accuracy);
% Maximum SVM accuracy
max_SVM_accuracy=max(SVM_accuracy);
end

%% ANN classification
function [ANN_OUT,ANN_Targets_test,ANN_accuracy,avg_ANN_accuracy, max_ANN_accuracy]=ANNclassification(ANN_Features,ANN_Targets,total_trial,N_train,N_valid, N_test, net)
net.divideParam.trainInd=N_train;
net.divideParam.valInd=N_valid;
net.divideParam.testInd=N_test;
ANN_Targets_test=vec2ind(ANN_Targets(:,N_test));
for num_trial=1:total_trial
% ANN training
[ANNmodel,tr] = train(net,ANN_Features,ANN_Targets);
% ANN testing
y = ANNmodel(ANN_Features);
e = gsubtract(ANN_Targets,y);
yind = vec2ind(y);
OUT=yind(:,net.divideParam.testInd);
% ANN output
ANN_OUT(num_trial,:)=OUT;
% ANN accuracy
ANN_accuracy(num_trial)=sum(OUT==ANN_Targets_test)/length(ANN_OUT);
end
% Average ANN accuracy
avg_ANN_accuracy=mean(ANN_accuracy);
% Maximum ANN accuracy
max_ANN_accuracy=max(ANN_accuracy);
end

%% HMM-based SVM output correction
function [SVM_OUT_corr,SVM_accuracy_corr,avg_SVM_accuracy_corr,max_SVM_accuracy_corr]=HHM_SVM_corr(SVM_obs_seq,SVM_state_seq,total_trial)
for num_trial=1:total_trial

observed=SVM_obs_seq(num_trial,:);
corrected(1)=observed(1);
    tbd=[];
    for k=2:length(observed)
        temp=observed(k);
        if isempty(tbd)
            if temp>observed(k-1)
                corrected(k)=observed(k-1);
            elseif temp==observed(k-1)
                corrected(k-1)=temp;
                corrected(k)=temp;
            else
                if k<length(observed)
                tbd=temp;
                else
                    corrected(k)=corrected(k-1);
                end
            end   
        else
            if temp>tbd
                corrected(k-1:k)=temp;
                tbd=[];
            elseif temp==tbd
                corrected(k-1:k)=tbd;
                tbd=[];
            else
                if k<length(observed)
                    corrected(k-1:k)=tbd;
                    tbd=temp;
                else
                    corrected(k-1:k)=tbd;
                    tbd=[];
                end
            end
        end        
    end
% Corrected output
SVM_OUT_corr(num_trial,:)=corrected;
% Corrected SVM accuracy
SVM_accuracy_corr(num_trial)=sum(corrected==SVM_state_seq(1:length(SVM_OUT_corr)))/length(SVM_OUT_corr);
end
% Average corrected SVM accuracy
avg_SVM_accuracy_corr=mean(SVM_accuracy_corr);
% Maximum corrected SVM accuracy
max_SVM_accuracy_corr=max(SVM_accuracy_corr);
end

%% HMM-based ANN output correction
function [ANN_OUT_corr,ANN_accuracy_corr,avg_ANN_accuracy_corr,max_ANN_accuracy_corr]=HHM_ANN_corr(ANN_obs_seq,ANN_state_seq,total_trial)
for num_trial=1:total_trial

observed=ANN_obs_seq(num_trial,:);
corrected(1)=observed(1);
    tbd=[];
    for k=2:length(observed)
        temp=observed(k);
        if isempty(tbd)
            if temp>observed(k-1)
                corrected(k)=observed(k-1);
            elseif temp==observed(k-1)
                corrected(k-1)=temp;
                corrected(k)=temp;
            else
                if k<length(observed)
                tbd=temp;
                else
                    corrected(k)=corrected(k-1);
                end
            end   
        else
            if temp>tbd
                if temp-tbd==1
               corrected(k-1:k)=temp;
                else
                    corrected(k-1:k)=temp-1;
                end
                tbd=[];
            elseif temp==tbd
                corrected(k-1:k)=tbd;
                tbd=[];
            else
                if k<length(observed)
                    corrected(k-1:k)=tbd;
                    tbd=temp;
                else
                    corrected(k-1:k)=tbd;
                    tbd=[];
                end
            end
        end        
    end
% Corrected output
ANN_OUT_corr(num_trial,:)=corrected;
% Corrected ANN accuracy
ANN_accuracy_corr(num_trial)=sum(corrected==ANN_state_seq(1:length(ANN_OUT_corr)))/length(ANN_OUT_corr);
end
% Average corrected ANN accuracy
avg_ANN_accuracy_corr=mean(ANN_accuracy_corr);
% Maximum corrected SVM accuracy
max_ANN_accuracy_corr=max(ANN_accuracy_corr);
end