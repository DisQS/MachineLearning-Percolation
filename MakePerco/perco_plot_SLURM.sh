#!/bin/bash

dir=${1:-../data}
#size=${2:-10}
option=${2:-0}
cores=${3:-1}
py=${4:-perco_generate_plot.py}
  
codedir=`pwd`

echo "PERCO: dir=" $dir ", size=" $size ", option=" $option ", cores=" $cores",py=" $py

cd $dir
#cd "L"$size

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

module load Anaconda3/2019.03
module load GCCcore/8.3.0
module load parallel/20190922

#conda init --all; conda activate

pwd
echo "--- working in directory=$directory"

#python $codedir/perco_generate_plot.py $option $pklfile `basename $pklfile .pkl`.cor
ls *.pkl| parallel -j$cores -a - python $codedir/$py 0 {} {}

chmod -R g+w *.pdf

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
