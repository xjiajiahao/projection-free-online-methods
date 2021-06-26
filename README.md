### Dependencies
1. Operating System: Linux
2. MATLAB R2019a or above

### How to run

#### Online Covariance Matrix Estimation
``` matlab
% conduct experiment in the stochastic setting and plot figures
test_CME_stoch;
plot_CME_stoch;

% conduct experiment in the adversarial setting and plot figures
test_CME_adv;
plot_CME_adv;
```


#### Online Collaborative Filtering

Step 1: compile the `./Utils/proj_l1.cpp` file from the terminal
``` bash
mex -largeArrayDims ./Utils/proj_l1.cpp -lm -ldl
```

Step 2: change the current directory to `./Utils`, and then execute the following script in MATLAB to generate the requried datasets
```matlab
build_CF_datasets;
```

Step 3: change the current directory back to `./`, and then execute the following scripts to conduct experiments
``` matlab
% conduct experiment on MovieLen 100K dataset and plot figures
test_CF_MovieLens100K;
plot_CF('MovieLens100K');

% conduct experiment on Jester 1 dataset and plot figures
test_CF_Jester1;
plot_CF('Jester1');

% conduct experiment on MovieLen 1M dataset and plot figures
test_CF_MovieLens1M;
plot_CF('MovieLens1M');

% conduct experiment on Jester 3 dataset and plot figures
test_CF_Jester3;
plot_CF('Jester3');
```


#### Online Binary Classification
Step 1: change the current directory to `./Utils`, and then execute the following script in MATLAB to generate the requried dataset
```matlab
build_eeg_dataset;
```

Step 2: change the current directory back to `./`, and then execute the following scripts to conduct experiments
``` matlab
% online binary classification with smooth cost functions
test_SVM_smooth;
plot_SVM_smooth;

% online binary classification with l1 regularized cost functions
test_SVM_l1;
plot_SVM_l1;
```
