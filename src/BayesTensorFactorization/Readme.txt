Code of Bayesian CP Factorization for tensor completion. (Written by Qibin Zhao)


To run the code:
1. Add the whole folder and subfolder into the Matlab path search list. 
2. Run the demo files 


We provide two demo codes:
I. DemoBayesCP.m:    Demonstration on synthesic data. 

The experimental settings that one can test include that 
1) Tensor Size
2) True CP Rank
3) Data type (deterministic signals, random data)
4) Observation rate 
5) Noise SNR

The settings of algorithm does not need any changes, but if you like you can test
1) Initialization method (SVD, random)
2) Initial Rank
3) If the components will be pruned out or not.
4) Convergence condition ('tol', 'maxiters')
5) Visualization during model learning ('verbose')


After the model learning, you can visualize the results by
1. Performance RSE, RMSE, Estimated Rank, TimeCost, Noise estimation
2. Visualization of true latent factors and the estimated factors
3. visualization of observed tensor Y,  the estimated low-CP-rank tensor X by cubic style visualization


You may also run the classical CP factorization algorithm for comparisons. 


II. DemoBayesCP_Image.m   Demonstration for image completion. 
 
The experimental settings that one can test include that
1) image file name
2) observation rate (1-missing rate)


Two provided codes include FBCP, and FBCP-MP.  
The results of RSE, PSNR, SSIM, Time Cost can be evaluated and reported. The visual quality is visualized 
during model learning and after the algorightm is finished. 
