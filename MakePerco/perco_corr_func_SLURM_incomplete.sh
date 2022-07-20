#!/bin/bash

dir=${1:-../data}
size=${2:-10}
option=${3:-0}
cores=${4:-1}

codedir=`pwd`

echo "PERCO: dir=" $dir ", size=" $size ", option=" $option ", cores=" $cores

cd $dir
cd "L"$size
pwd

for directory in p0*
do

echo $directory
cd $directory

jobfile=$size-$directory".sh"

echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3700

module restore TorchGPU_1_7_1
#conda init --all; conda activate

pwd
echo "--- working in directory=$directory"

#nbpkl=\`ls -lR ./*.pkl | wc -l\`
#nbcor=\`ls -lR ./*.cor | wc -l\`
#echo \$nbpkl \$nbcor

rm -f missing_cor.lst
for corfile in \*.cor
do 
    #echo \$corfile
    if [ \$(cat \$corfile | wc -l) -lt 1032 ]; then
        #echo "in the loop" 
        echo \$corfile>> missing_cor.lst
        echo "--- MISSING:" \$corfile
    fi
done

#if [ $option = 2 ]
    #sort -n missing_cor.lst | parallel -j1 -a - python $codedir/perco_generate_corr.py $option {} {}
#fi
#echo $nbpkl 
#echo "fin"

echo "--- finished in directory=$directory"

EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
#(sbatch -q taskfarm ${jobfile})
(sbatch ${jobfile})
#(./${jobfile})

cd ..
done

cd $codedir

