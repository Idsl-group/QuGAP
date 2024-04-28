# QuGAP: Generating Universal Adversarial Perturbations for Quantum Classifiers

Repository containing all the source code used for the paper *Generating Universal Adversarial Perturbations for Quantum Classifiers*. Work done by [Gautham Govind Anil](https://github.com/blaze010) and [Vishnu Vinod](https://github.com/vishnuvind) while at the University of British Columbia.

## Paper
*<u>Title</u>*: Generating Universal Adversarial Perturbations for Quantum Classifiers\
*<u>Authors</u>*: Gautham Govind Anil, Vishnu Vinod and Apurva Narayan\
*<u>Conference</u>*: [The 38th Annual AAAI Conference on Artificial Intelligence](https://aaai.org/aaai-conference/), February 20-27, 2024\
*Link to Paper (with supplementary material)*: [Here](https://arxiv.org/abs/2402.08648)
*AAAI Proceedings (Main Paper only)*: [Here](https://ojs.aaai.org/index.php/AAAI/article/view/28963)

## Code Base
Folder structure of the Project Directory is given below
```
    ├── data            [datasets for XOR and Transverse Ising Model (TIM) classification tasks]
    │   ├── ising
    │   │   ├── 2       [2 qubit data]
    |   |   │   └── ...     (2 files: train and test)
    │   │   └── 4       [4 qubit data]
    |   |       └── ... (2 files: train and test)
    │   ├── xor
    │   │   └── ...     (2 files: train and test)
    │   └── dataset.py  [Code to generate datasets]
    ├── logs            [Directory to store all logs]
    │   └── ...         (empty)
    ├── results         [Directory to store all results]
    │   └── ...         (empty)
    ├── scripts         [Bash scripts for running different tasks (detailed later)]
    │   ├── test_cgen.sh
    │   ├── train_cgen.sh
    │   ├── train_qbim.sh
    │   ├── train_qgen.sh
    │   ├── train_target.sh
    │   └── train_unitary.sh
    ├── targets         [Trained Target models (classical and PQC-based classifiers)]
    │   └── ...         (18 files)
    ├── uaps            [Trained untargeted additive UAPs for testing (we provide UAPs with strengths 0.15 & 0.30)]
    │   └── ...         (30 files)
    ├── LICENSE
    ├── README.md
    ├── config.py
    ├── models.py
    ├── qmodels.py
    ├── requirements.txt
    ├── test_cgen.sh
    ├── train_cgen.sh
    ├── train_qbim.sh
    ├── train_qgen.sh
    ├── train_target.sh
    ├── train_unitary.sh
    └── utils.py
```
            
## Supported Tasks
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

### Computational Requirements
This is necessary because quantum devices have to be initialized with a fixed number of wires. The code is structured in such a way. Kindly note that **CUDA devices are required in order to read the pickled files**. Please test on machines which have **CUDA enabled devices and a RAM of atleast $20$ GB** in order to allow flawless execution.

## Python Scripts

**Please run all python scripts from within the `scripts/` folder**

General code for running any of the scripts is given as ```bash script_file_name.sh```

1. `train_target.sh`: Code to train target models (both classical and quantum)
2. `train_cgen.sh`: Code to train QuGAP-A for additive UAPs (control settings by changing)
3. `test_cgen.sh`: Code to test generated additive UAPs (**RUN DIRECTLY TO SEE RESULTS**)
4. `train_unitary.sh`: Code to train and test unitary simulation of QuGAP-U
5. `train_qbim.sh`: Code to train and test qBIM algorithm 
6. `train_qgen.sh`: Code to train and test QuGAP-U algorithm

Further comments can be found inside each `.sh` file to aid in modifying, testing and playing around with the settings of each task.

## Citation

If you use the work presented in the paper or the code hosted in this repository, please consider citing our paper at:

```
@article{
    Anil_Vinod_Narayan_2024, 
    title={Generating Universal Adversarial Perturbations for Quantum Classifiers}, 
    author={Anil, Gautham and Vinod, Vishnu and Narayan, Apurva}, 
    volume={38}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/28963}, 
    DOI={10.1609/aaai.v38i10.28963}, 
    number={10}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    year={2024}, 
    month={Mar.}, 
    pages={10891-10899}
}
```
