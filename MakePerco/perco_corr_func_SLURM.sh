#!/bin/bash

dir=${1:-../data}
size=${2:-10}
option=${3:-0}
cores=${4:-1}

codedir=`pwd`

echo "PERCO: dir=" $dir ", size=" $size ", option=" $option ", cores=" $cores

cd $dir
cd "L"$size

for directory in p0*
do

cd $directory

jobfile="corr-"$size-$directory".sh"

#echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2012
#SBATCH --time=00:48:00

module load Anaconda3
#conda init --all; conda activate

pwd
echo "--- working in directory=$directory"

#python $codedir/perco_generate_corr.py $option $pklfile `basename $pklfile .pkl`.cor

if [ $option = 1 ]; then
  ls *.pkl| parallel -j$cores -a - python $codedir/perco_generate_corr.py $option {} {}
else
  ls *.pkl| parallel -j1 -a - python $codedir/perco_generate_corr.py $option {} {}
fi

chmod -R g+w *

echo "--- finished in directory=$directory"
EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
#(./${jobfile})

cd ..
done

cd $codedir

