# MachineLearning-Percolation
Machine learning code to study the phase transition in 2D percolation
**Modules/Libraries needed for data generation**
```
module load GCCcore/11.2.0 
module load Python/3.9.6
pip install --user imageio
module load GCC/11.2.0
module load OpenMPI/4.1.1 matplotlib/3.4.3
module load IPython
```
or install
```
sudo apt install python
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install operators

```

**Modules/Libraries needed for data ML training**
```
module load GCC/11.3.0 OpenMPI/4.1.4 PyTorch/1.12.1-CUDA-11.7.0
module load IPython
module load matplotlib
pip install --user torchvision
pip install --user seaborn
pip install --user tqdm
pip install --user torch-summary
pip install --user scikit-learn

```
or install (PyTorch 1.12.1 with CUDA-11.7.0 was used for the implementation)
```
pip3 install torch torchvision torchaudio
pip3 install --user torchvision
pip3 install --user seaborn
pip3 install --user tqdm
pip3 install --user torch-summary
pip3 install --user scikit-learn
```
**Creation of the dataset**
The codes to generate the percolation lattices can be found in the MakePerco/ directory.
To generate 1000 configuration with L=100 and p \in [0.1,0.9,0.1] use: 
```
MakePerco/perco_SLURM.sh ./TestData/ 100 1000 9000 1000 1000
```
**ML training and testing**
Five different directories in MLCode to launch ML training. To launch test for spanning from saved model, first change absolute path leading to MLtools.py in the Train-Pytorch-class_span.py file, then use the command 

```
./MLCode/perco_ML_training_sulis.sh ./trained_model/ 12345678 ./MLCode/class_span/Train-Pytorch-class_span.py  1
```

