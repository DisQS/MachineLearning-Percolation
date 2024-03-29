#!/bin/bash

dir=${1:-../data}   

#seed=${2:-1234567}
size=${2:-10}
perco_i=${3:-05927}
perco_f=${4:-06000}
dperco=${5:-02000}
configs=${6:-2}
py=${7:-perco_generate_data.py}
codedir=`pwd`

echo "PERCO: dir=" $dir ", size=" $size \
", [Ti,Tf,dT]= [" $perco_i, $perco_f, $dperco "], configs=" $configs",py=" $py

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
#SBATCH --time=48:00:00

module restore new_TorchGPU_1_7_1
#conda init --all; conda activate

pwd
echo "--- working on p=$perco"

python $codedir/$py $size $perco $perco 1 $configs

chmod -R g+w *

echo "--- finished with p=$perco"
EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
#(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
(./${jobfile})

done
