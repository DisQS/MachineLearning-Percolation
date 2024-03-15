# MachineLearning-Percolation
Machine learning code to study the phase transition in 2D percolation
# Modules/Libraries needed for data generation
```
module load GCCcore/11.2.0 
module load Python/3.9.6
pip install --user imageio
module load GCC/11.2.0
module load OpenMPI/4.1.1 matplotlib/3.4.3
module load IPython
```
# Creation of the dataset
The codes to generate the percolation lattices can be found in the MakePerco/ directory.
To generate 1000 configuration with L=100 and p \in [0.1,0.9,0.1] use: 
```
MakePerco/perco_SLURM.sh ./TestData/ 100 1000 9000 1000 1000
```
# ML training and testing
five different directories in MLCode to launch ML training. To launch test for spanning from saved model, first change absolute path leading to MLtools.py in the Train-Pytorch-class_span.py file, then use the command 

```
./MLCode/perco_ML_training_sulis.sh ./trained_model/ 12345678 ./MLCode/class_span/Train-Pytorch-class_span.py  1
```

