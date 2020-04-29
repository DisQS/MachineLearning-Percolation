#!/bin/bash

dir=${1:-../data}   
seed=${2:-1234567}
size=${3:-10}
perco_i=${4:-4.0}
perco_f=${5:-4.0}
dperco=${6:-0.1}
configs=${7:-2}

codedir=`pwd`

echo "PERCO: dir=" $dir ", seed=" $seed ", size=" $size \
", [Ti,Tf,dT]= [" $perco_i, $perco_f, $dperco "], configs=" $configs

mkdir -p $dir
cd $dir
mkdir -p "L"$size
cd "L"$size

for perco in $(seq $perco_i $dperco $perco_f) 
do

echo "--- making jobfile for p=" $perco

jobfile=`printf "$perco.sh"`
echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2012
#SBATCH --time=00:30:00

module load Anaconda3
#conda init --all; conda activate

pwd
echo "--- working on $perco"

python $codedir/perco_data_generate.py $seed $size $perco $perco 1 $configs

echo "--- finished with $perco"
EOD

cat ${jobfile}

chmod 755 ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
#(./${jobfile})

done
