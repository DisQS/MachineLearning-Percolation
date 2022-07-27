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

module restore new_TorchGPU_1_7_1

#conda init --all; conda activate

pwd
echo "--- working in directory=$directory ---"

rm -f missing_cor.lst

corcount=0
miscount=0
for pklfile in *.pkl
do 
    count=\$((\$count + 1))
    echo -ne \$count \$miscount'\r'

    corfile=\`basename \$pklfile .pkl\`.cor
    echo \$corfile
    if [ \$(cat \$corfile | wc -l) -lt 1032 ]; then
        echo \$corfile
	miscount=\$((\$miscount + 1)) 
        echo \$corfile>> missing_cor.lst
    fi
done
filename='missing_cor.lst'
echo "read missing_cor file"


file=\$(cat missing_cor.lst)

for line in \$(cat missing_cor.lst)
do
    if [ "\${line: -15}" == "_incomplete.cor"} ]; then
        echo "\$line already modified"
    else
        echo "\$line now being renamed"
        mv "\$line" "\${line/.cor/_incomplete.cor}"
    fi
done

echo "work on corr"
for line in \$(cat missing_cor.lst)
do
python $codedir/perco_generate_corr.py $option \$line \$line
done
echo "end corr"

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

