# MachineLearning-Percolation
Machine learning code to study the phase transition in 2D percolation
** Modules/Libraries needed for data generation**
```
module load GCCcore/11.2.0 
module load Python/3.9.6
pip install --user imageio
module load GCC/11.2.0
module load OpenMPI/4.1.1 matplotlib/3.4.3
```
**Creation of the dataset**
The codes to generate the percolation lattices can be found in the MakePerco/ directory.
To generate 1000 configuration with L=100 and p \in [0.1,0.9,0.1] use: 
```
MakePerco/perco_SLURM.sh ./test/ 100 1000 9000 1000 1000
```

