## FRAPPE

This directory contains the data generation code and the code for both FRAPPE and Self-FRAPPE.

- `make_matlab_test.py` generates the synthetic dataset used to evaluate our methods and the baseline methods.
- `train_alexnet.py` trains the AlexNet model and evaluates its performance at different ranks. It also runs FRAPPE on it. Finally, it saves a copy of the weight tensor (to test the MATLAB baselines with).
- `eval_self_frappe.py` runs Self-FRAPPE on the synthetic dataset.
- `eval_self_frappe_real.py` runs Self-FRAPPE on the real generated dataset.
- `time_training.py` times the training of the FRAPPE model.
- `train_4D.py` trains the 4D FRAPPE model used for the AlexNet task.
- `alexnet_ss.py` runs Self-FRAPPE on the AlexNet weight tensor.

Other module code (cannot be run directly; used by other scripts):

- `extract_features.py` contains the 3D tensor feature extraction code.
- `extract_features4.py` contains the 4D tensor feature extraction code.
- `net.py` contains the AlexNet base code, as well as code for approximating layers with the CPD. Contains code from the PyTorch Lightning AlexNet demo.
- `predictor.py` contains code for loading trained versions of the 3D/4D FRAPPE.
- `self_frappe.py` contains the Self-FRAPPE implementation.
- `dset.py` includes code to load and generate datasets. This is where we generate our synthetic dataset.
