# QuGAP: Generating Universal Adversarial Perturbations for Quantum Classifiers

### Code Base
The entire code base consists of the following folders:

- `data/`     : contains datasets for XOR and Transverse Ising Model (TIM) Dataset (MNIST & FMNIST auto-downloaded - please ensure your device has an internet connection to allow download of MNIST/FMNIST datasets)
- `targets/`  : contains trained target models (we provide a number of such models)
- `uaps/`     : contains trained untargeted UAPs for testing (we provide UAPs with $\epsilon=0.15$ & $\epsilon=0.30$)
- `results/`  : stores all results from training and testing in pickled files
- `logs/`     : stores all logs from training and testing in log files
- `scripts/`  : has 6 bash scripts for running different tasks
            
### Supported Tasks
We support testing for the following classification tasks:
- MNIST/FMNIST - 2 class - $16\times16$
- MNIST/FMNIST - 4 class - $16\times16$
- MNIST/FMNIST - 2 class - $8\times8$
- TIM Dataset  - 2 class

Kindly note that MNIST/FMNIST $8\times8$ is a special case designed for use only for testing the quantum generator in `train_qgen.sh` and `train_qbim.sh`.

### Configuring the Task
The `config.py` contains $4$ different configurations each for a separate test setting. The default setting is Setting 4. **Please set the correct setting for the task you test in the config.py file.** Failure to do so will lead to erroneous results. The settings can be chosen by simply commenting out the unrequired setting and using the correct one as follows:

- Setting 1: Ising 2-class classification - 5 qubits, 1 ancillary qubit 
- Setting 2: MNIST/FMNIST $8\times8$ - 2-class classification - 7 qubits, 1 ancillary qubit
- Setting 3: MNIST/FMNIST $16\times16$ - 2-class classification - 9 qubits, 1 ancillary qubit
- Setting 4: MNIST/FMNIST $16\times16$ - 4-class classification - 10 qubits, 2 ancillary qubits

This is necessary because quantum devices have to be initialized with a fixed number of wires. The code is structured in such a way. Kindly note that **CUDA devices are required in order to read the pickled files**. Please test on machines which have **CUDA enabled devices and a RAM of atleast $20$ GB** in order to allow flawless execution.

### Python Scripts

**Please run all python scripts from within the `scripts/` folder**

General code for running any of the scripts is given as ```bash script_file_name.sh```

1. `train_target.sh`: Code to train target models (both classical and quantum)
2. `train_cgen.sh`: Code to train QuGAP-A for additive UAPs (control settings by changing)
3. `test_cgen.sh`: Code to test generated additive UAPs (**RUN DIRECTLY TO SEE RESULTS**)
4. `train_unitary.sh`: Code to train and test unitary simulation of QuGAP-U
5. `train_qbim.sh`: Code to train and test qBIM algorithm 
6. `train_qgen.sh`: Code to train and test QuGAP-U algorithm

Further comments can be found inside each `.sh` file to aid in modifying, testing and playing around with the settings of each task.