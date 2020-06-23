# HMM-supported ML for DC-link capacitor CM
Software implementation of Condition Monitoring for DC-link capacitors in MATLAB.

This code implements the Condition Monitoring approach introduced in the Master's Thesis "HIDDEN MARKOV MODEL-SUPPORTED MACHINE LEARNING FOR CONDITION MONITORING OF DC-LINK CAPACITORS".

The approach uses current measurements obtained with a current transducer and estimates a DC-link capacitor health through 3 stages.
1. Data preprocessing: FFT is applied to time-domain data to obtain signal spectrum; spectrum is smoothed using the moving average filter with a rectangular window, the moving average filter with a Hanning window, the locally weighted linear regression, and the Savitzky-Golay filter.
2. ML classification: SVM and the ANN learning algorithms classify the data into 5 classes corresponding to the capacitor age.
3. HMM-based output correction: HMM-supported output correction of the ML results outputs the estimated the DC-link capacitor class with an enhanced accuracy. 
