# DoA-PREDICTION-AND-CLASSIFICATION-USING-ANN
Development of algorithm to extract frequency domain feature, beta ratio of EEG signals and compare it with BIS values using a NARX Neural Network and obtain the Depth of Anesthesia. 


**PROJECT DESCRIPTION:**

The electroencephalogram (EEG) can reflect brain activity and contains abundant information of different anesthetic states of the brain. During surgery, general anesthesia is necessary and important to ensure the safety of patients. Overdose anesthesia may make the recovery time longer, while inadequate anesthesia may lead to intraoperative awareness and psychological effects on patients. However, there is not an identical definition of the anesthetic state among anesthesiologists. Objective, non-invasive and reliable monitoring depth of anesthesia (DoA) is still a clinical concern for anesthesiologists. The bi-spectral index (BIS) monitor collects raw EEG data through its sensors and uses an algorithm to analyze and interpret the data. The data displays as a number on the BIS view monitor. BIS values range from 0 to 100. A value of 0 represents the absence of brain activity, and 100 represents the awake state.

In this study, we propose a method that combines multiple EEG-based features with an artificial neural network (ANN) to assess the DoA. The parameter beta ratio is to be extracted from the EEG Signal and given as input to the ANN which uses Bi-spectral index (BIS) as the reference output to find out the dependence using regression analyses. This method would be promising and feasible for a monitoring system to assess the Depth of Anesthesia.

**PROPOSED SYSTEM**

The proposed solution involves pre-processing of the EEG signal to identify the right frequency band of the EEG signal and remove noise and other artifacts such as EMG  and EEG. The processed signal is then analysed to extract the required features such as beta ratio using wavelet decomposition technique. The extracted feature is fed to an Artificial Neural Network using Levenberg Marquardt regression algorithm along with bis values to train the data based on the mentioned model and test the data to find out the regression between BIS and beta ratio, calculate the error in regression and also obtain the time series plot of the test and train input and outputs respectively.

![image](https://user-images.githubusercontent.com/111851675/186106845-0ddb13a9-5a28-4f08-aa44-a2af0a41348c.png)


**PRE-PROCESSING OF EEG DATA**
Step:01. The EEG raw data are imported into MATLAB.
Step:02. Sampling of EEG data with sampling frequency 128 Hz.
Step:03. Removal of outliers using Hampel filter.
Step:04. Filter EEG data with a low pass FIR filter to allow only the band of signal between 0-47 Hz, as this is where the majority of EEG data is found.
Step:05. Denoising to remove EOG artifacts by calculating the coefficients of Daubechies wavelet coefficients as most of the EOG artifacts lie in this region. Using   the coefficients, the threshold is calculated to remove EOG artifacts based on soft adaptive thresholding.
Step:06. Denoising to remove EMG artifacts using median filter to remove the unwanted spikes in amplitude caused to muscular movements.
Step:07. Plot the processed EEG signal as a function of time.

**FEATURE EXTRACTION**
Step:01. Obtain and plot the wavelet decomposition (alpha, beta, gamma, theta and delta) of the processed EEG signal.
Step:02. Calculate the power spectral density of each band.
Step:03. Calculate beta ratio using the above-mentioned formula.
Step:04. Plot the beta ratio as a function of time.

**REGRESSION ANALYSIS USING ANN**
Step:01. Train the neural network using the beta ratio.
Step:02. Set the target data as the processed signal and reference data as BIS values.
Step:03. Train the algorithm by assigning a particular percentage of data for validation, testing and training.
Step:04. Obtain and plot the regression coefficient, error, time series response.

**RESULTS**

![image](https://user-images.githubusercontent.com/111851675/186107095-5ce7a595-7ba5-401b-aada-248ab0335a4c.png)

![image](https://user-images.githubusercontent.com/111851675/186107185-d4c8f2e4-2f1d-4b96-a9d8-966bb3e0f24b.png)

![image](https://user-images.githubusercontent.com/111851675/186107285-c1a8921c-4e1e-407f-b17a-60f2c456f310.png)

![image](https://user-images.githubusercontent.com/111851675/186107359-cddb7376-5dca-48b9-91e2-668762b52e9b.png)

![image](https://user-images.githubusercontent.com/111851675/186107434-2f976ba2-d91c-410e-8d5e-1a12a6c9bb75.png)

![image](https://user-images.githubusercontent.com/111851675/186107517-9333d854-19ac-4471-87f5-24c6c05019f5.png)

![image](https://user-images.githubusercontent.com/111851675/186107607-844e79f9-bf5b-41ad-8470-3d419d4bf815.png)

![image](https://user-images.githubusercontent.com/111851675/186107835-fe6c7b9e-1bb9-4107-b15d-bd42e8337d09.png)

![image](https://user-images.githubusercontent.com/111851675/186107750-7905bf1d-211b-4155-ad1e-991fbe7e75de.png)







