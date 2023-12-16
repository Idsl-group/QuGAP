######################################################
#  RUN THIS CODE TO GENERATE DATASETS - XOR & ISING 
#  command: python3 dataset.py
#  ( or use the datasets we provide :) )
######################################################


# Library imports
import math
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pennylane as qml

def paulix(wire, num):
    pauli = np.array([[0, 1], [1, 0]])
    if wire==0: I = pauli
    else: I = np.eye(2)
        
    for i in range(1, num):
        if i==wire: I = np.kron(I, pauli)
        else: I = np.kron(I, np.eye(2))
    return I

def pauliz(wire, num):
    pauli = np.array([[1, 0], [0, -1]])
    if wire==0: I = pauli
    else: I = np.eye(2)
        
    for i in range(1, num):
        if i==wire: I = np.kron(I, pauli)
        else: I = np.kron(I, np.eye(2))
    return I

# Transverse Ising Model Dataset
qubit_sets = [2,4,8]
dsets = {'train': 5000, 'test':1000}
for qubits in qubit_sets:
    for dset in dsets.keys():
        Js, energies, states, phases = [], [], [], []
        for i in tqdm(range(dsets[dset])):
            J = 2*i/(dsets[dset]-1)
            if J<1: phase = 1 # ferromagnetic
            else: phase = 0 # paramagnetic
            hamil = 0
            for i in range(qubits): hamil += (-J)*paulix(i, qubits)
            for i in range(qubits-1): hamil += (-1)* pauliz(i, qubits)@pauliz(i+1, qubits)
            val, vec = np.linalg.eig(hamil)
            Js.append(J)
            energies.append(val[np.argmin(val)].astype(np.complex128))
            states.append(vec.T[np.argmin(val)].astype(np.complex128))
            phases.append(phase)
        items = {'Ground State': states, 'Transverse Field (J)': Js, 'Ground State Energy': energies, 'Phase': phases}
        df = pd.DataFrame(items)
        df.to_pickle('./ising/'+ str(qubits) + '/transverse_field_ising_model_' + dset + '.csv')


# XOR Dataset
dsets = {'train': 850, 'test': 150}
for dset in dsets.keys():
    classes, states = [], []
    for i in tqdm(range(dsets[dset])):
        theta = 2*np.pi * (i/dsets[dset])
        xx = complex(np.cos(theta),np.sin(theta))
        if 0<=theta<np.pi/2 or np.pi<=theta<3*np.pi/2: class_x = 0
        else: class_x = 1
        
        states.append(xx)
        classes.append(class_x)
    items = {'State': states, 'Label':classes}
    df = pd.DataFrame(items)
    df.to_pickle('./xor/' + dset + '.csv')
