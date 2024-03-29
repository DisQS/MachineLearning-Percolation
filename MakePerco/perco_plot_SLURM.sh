#!/bin/bash

dir=${1:-../data}
size=${2:-10}
option=${3:-0}
cores=${4:-1}
ext=${5:-.png}

  
codedir=`pwd`

echo "PERCO: dir=" $dir ", size=" $size ", option=" $option ", cores=" $cores ", ext=" $ext

cd $dir
cd "L"$size

for directory in p0*
do

cd $directory

jobfile="plot-"$size-$directory".sh"

#echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2012
#SBATCH --time=48:00:00


module restore Python_3_9_6
#module load GCC/11.2.0
#module load OpenMPI/4.1.1 matplotlib/3.4.3

module load parallel/20210722

#conda init --all; conda activate

pwd
echo "--- working in directory=$directory"


#python $codedir/perco_generate_plot.py $option $pklfile `basename $pklfile .pkl`.cor $ext
ls *.pkl| parallel -j$cores -a - python $codedir/perco_generate_plot.py $option {} `basename {} .pkl`.cor $ext

chmod -R g+w *$ext

#echo "--- finished in directory=  $directory"
EOD

#cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
#(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
(./${jobfile})

cd ..
done

cd $codedir
