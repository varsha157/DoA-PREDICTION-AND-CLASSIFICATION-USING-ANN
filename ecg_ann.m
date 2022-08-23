clc
clear all
close all

%IMPORTING DATA SET AND SETTING FS
load case1.mat;
s=EEG;
fs=128;
s1=bis;
figure(1);
subplot(2,1,1);
plot(s, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Samples', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('Raw EEG Signal', 'fontsize', 15);
grid on
xlim([0 1000])

%EEG PRE-PROCESSING STAGE
%1. REMOVING OUTLIERS
windowSize = 6;
numMedians = 2;
[filteredData,outliers]=hampel(s,windowSize,numMedians);

subplot(2,1,2);
plot(filteredData, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Samples', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('Removal of Outliers (Filtered Signal)', 'fontsize', 15);
grid on
xlim([0 600])

subplot(3,1,3);
plot(s, 'linewidth', 1.2);
xlim([0 600])
hold on
plot(filteredData, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Samples', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('Comparison of Original and Filtered EEG Signal', 'fontsize', 15);
grid on
xlim([0 600]);
legend('Orginal Data', 'Filtered Data');

%2. FIR FILTERATION
N=10;
%LOW PASS FILTER 0-47 Hz
[n,d]=fir1(N,0.3274,blackman(N+1)); %fir1(N,Wn,WINDOW) {0<WN<1} 
[h w] = freqz (n,d,128); %freqz(B(N),A(D),N)
h=freqz(n,d,128); %{N-point complex frequency response}

figure(2);
subplot(2,1,1);
plot(w*fs/(2*pi),abs(h), 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Frequency (in Hz)', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Gain in dB', 'fontsize', 12, 'fontweight', 'bold');
title('Response of LPF', 'fontsize', 15);
grid on

sf = filter(n,d,filteredData); %filter(B,A,X) {Filters X with a and b}
subplot(2,1,2)
plot(sf, 'linewidth', 1.2)
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Samples', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('Filtered EEG Signal Output', 'fontsize', 15);
grid on
xlim([0 600]);



%3. DENOISING USING WAVELET TRANSFORM TO REMOVE EOG ARTIFACTS.
wname1='db7';
wname2='db9';
d1 = mean(dbwavf(wname1));
d2 = mean(dbwavf(wname2));
d=d1+d2;
st=median(abs(d))/0.6745;
n=length(sf)
h=sqrt(2*log(n))
thr=st*h

eogremecg=wthresh(sf,'s',thr);
figure(3);
subplot(2,1,1);
plot(sf, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Samples', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('EEG SIGNAL WITH ARTIFACTS', 'fontsize', 15);
grid on
xlim([0 8000]);




%4. DENOISING USING MEDIAN FILTER TO REMOVE EMG SIGNALS (Spikes).
eeg = medfilt1(eogremecg,12);
subplot(2,1,2);

plot(eeg, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Samples', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('EEG SIGNAL AFTER REMOVAL OF ARTIFACTS', 'fontsize', 15);
grid on
xlim([0 8000]);

%FEATURE EXTRACTION
%1. Beta Ratio
N=2048;
waveletFunction = 'db8';
[C,L] = wavedec(eeg,8,waveletFunction);
cD1 = detcoef(C,L,1);
cD2 = detcoef(C,L,2);
cD3 = detcoef(C,L,3);
cD4 = detcoef(C,L,4);
cD5 = detcoef(C,L,5); %GAMA
cD6 = detcoef(C,L,6); %BETA
cD7 = detcoef(C,L,7); %ALPHA
cD8 = detcoef(C,L,8); %THETA
cA8 = appcoef(C,L,waveletFunction,8); %DELTA

D1 = wrcoef('d',C,L,waveletFunction,1);
D2 = wrcoef('d',C,L,waveletFunction,2);
D3 = wrcoef('d',C,L,waveletFunction,3);
D4 = wrcoef('d',C,L,waveletFunction,4);
D5 = wrcoef('d',C,L,waveletFunction,5); %GAMMA
D6 = wrcoef('d',C,L,waveletFunction,6); %BETA
D7 = wrcoef('d',C,L,waveletFunction,7); %ALPHA
D8 = wrcoef('d',C,L,waveletFunction,8); %THETA
A8 = wrcoef('a',C,L,waveletFunction,8); %DELTA

figure(5);
Gamma = D5;
subplot(5,1,1); 
plot(1:1:length(Gamma),Gamma, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Time', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('GAMMA', 'fontsize', 15);
grid on
xlim([0 10000])

Beta = D6;
subplot(5,1,2); 
plot(1:1:length(Beta), Beta, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Time', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('BETA', 'fontsize', 15);
grid on
xlim([0 10000])

Alpha = D7;
subplot(5,1,3); 
plot(1:1:length(Alpha),Alpha, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Time', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('ALPHA', 'fontsize', 15);
grid on
xlim([0 10000])

Theta = D8;
subplot(5,1,4);
plot(1:1:length(Theta),Theta, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Time', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('THETA', 'fontsize', 15);
grid on
xlim([0 10000])

Delta = A8;
subplot(5,1,5);
plot(1:1:length(Delta),Delta, 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Samples', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Amplitude', 'fontsize', 12, 'fontweight', 'bold');
title('DELTA', 'fontsize', 15);
grid on
xlim([0 10000])

%PSD CALCULATIONS
figure(6);
D5 = detrend(D5,0);
xdft = fft(D5);
freq = 0:N/length(D5):N/2;
xdft = xdft(1:length(D5)/2+1);
psdx = (1/(fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);

subplot(5,1,1);
plot(freq,(psdx), 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Frequency (Hz)', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Power', 'fontsize', 12, 'fontweight', 'bold');
title('GAMMA', 'fontsize', 15);
grid on
xlim([0 100]);

D6 = detrend(D6,0);
xdft2 = fft(D6);
freq2 = 0:N/length(D6):N/2;
xdft2 = xdft2(1:length(D6)/2+1);
psdx2 = (1/(fs*N)) * abs(xdft2).^2;
psdx2(2:end-1) = 2*psdx2(2:end-1);

subplot(5,1,2);
plot(freq2,abs(psdx2), 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Frequency (Hz)', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Power','fontsize', 12, 'fontweight', 'bold');
title('BETA', 'fontsize', 15);
grid on
xlim([0 100]);

D7 = detrend(D7,0);
xdft3 = fft(D7);
freq3 = 0:N/length(D7):N/2;
xdft3 = xdft3(1:length(D7)/2+1);
psdx3 = (1/(fs*N)) * abs(xdft3).^2;
psdx3(2:end-1) = 2*psdx3(2:end-1);

subplot(5,1,3);
plot(freq3,abs(psdx3), 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Frequency (Hz)', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Power', 'fontsize', 12, 'fontweight', 'bold');
title('ALPHA', 'fontsize', 15);
grid on
xlim([0 100]);

xdft4 = fft(D8);
freq4 = 0:N/length(D8):N/2;
xdft4 = xdft4(1:length(D8)/2+1);
psdx4 = (1/(fs*N)) * abs(xdft4).^2;
psdx4(2:end-1) = 2*psdx4(2:end-1);

subplot(5,1,4);
plot(freq4,abs(psdx4), 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Frequency (Hz)', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Power', 'fontsize', 12, 'fontweight', 'bold');
title('THETA', 'fontsize', 15);
grid on
xlim([0 100]);

A8 = detrend(A8,0);
xdft5 = fft(A8);
freq5 = 0:N/length(A8):N/2;
xdft5 = xdft5(1:length(A8)/2+1);
psdx5 = (1/(fs*N)) * abs(xdft5).^2;
psdx5(2:end-1) = 2*psdx5(2:end-1);

subplot(5,1,5);
plot(freq3,abs(psdx5), 'linewidth', 1.2);
set(gca, 'fontsize', 13, 'fontweight', 'bold');
xlabel('Frequency (Hz)', 'fontsize', 12, 'fontweight', 'bold');
ylabel('Power', 'fontsize', 12, 'fontweight', 'bold');
title('DELTA', 'fontsize', 15);
grid on
xlim([0 100]);


%BETA RATIO   
beta1=psdx2;
beta2=psdx3;
beta_num=log(beta2);
beta_den=log(beta1);
beta=10*(abs(beta_den-beta_num));
[filteredData1,outliers1]=hampel(beta,6,1);

betaratio=sum(filteredData1)/length(filteredData1)

figure(7);
plot(filteredData1);
xlim([2000 6000]);
xlabel('SAMPLES');
ylabel('AMPLITUDE');
title('BETA RATIO');
BIS=sum(s1)/767
beta=beta(1:767,1);

%1. ANN
 
    Training_Set=beta(1:end,1);%specific training set
    Target_Set=s1(1:end,1); %specific target set
    Input=Training_Set'; %Convert to row
    Target=Target_Set'; %Convert to row
    X = con2seq(Input); %Convert to cell
    T = con2seq(Target); %Convert to cell
    %%2. Data preparation
    N = 365;% Multi-step ahead prediction
    % Input and target series are divided in two groups of data:
    % 1st group: used to train the network
    inputSeries  = X(1:end-N);
    targetSeries = T(1:end-N);
    inputSeriesVal  = X(end-N+1:end);
    targetSeriesVal = T(end-N+1:end); 
    % Create a Nonlinear Autoregressive Network with External Input
    %time delay 
    delay = 2;
    inputDelays = 1:2;
    feedbackDelays = 1:2;
    hiddenLayerSize = 10;
      % DoA PREDICTION
    net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize);
    % Prepare the Data for Training and Simulation
    % The function PREPARETS prepares timeseries data for a particular network,
    % shifting time by the minimum amount to fill input states and layer states.
    % open loop or closed loop feedback modes.
    [inputs,inputStates,layerStates,targets] = preparets(net,inputSeries,{},targetSeries);
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    % Train the Network
    [net,tr] = train(net,inputs,targets,inputStates,layerStates);
    % Test the Network
    outputs = net(inputs,inputStates,layerStates);
    errors = gsubtract(targets,outputs);
    % View the Network
    view(net)
    % Plot
    netc = closeloop(net);
    netc.name = [net.name ' - Closed Loop'];
    view(netc)
    [xc,xic,aic,tc] = preparets(netc,inputSeries,{},targetSeries);
    yc = netc(xc,xic,aic);
    
    
if (betaratio>=65) && (betaratio<=80)
    disp('SEDATION')
elseif (betaratio>=45) && (betaratio<65)
    disp('GENERAL ANAESTHESIA')
elseif betaratio<40
    disp('BURST SUPPRESSION')
end


    

