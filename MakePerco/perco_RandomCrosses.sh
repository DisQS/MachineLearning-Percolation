#!/bin/bash

dir=${1:-../Data}  

#seed=${2:-1234567}
size=${2:-10}
perco_i=${3:-05927}
perco_f=${4:-06000}
dperco=${5:-02000}
configs=${6:-2}

codedir=`pwd`

echo "PERCO: dir=" $dir ", size=" $size \
", [Ti,Tf,dT]= [" $perco_i, $perco_f, $dperco "], configs=" $configs", py_code=" $pyfile

mkdir -p $dir
cd $dir

mkdir -p "L"$size"_cross"
cd "L"$size"_cross"

for perco in $(seq $perco_i $dperco $perco_f) 
do

echo "--- making jobfile for p=" $perco

jobfile=`printf "mlp-$size-$perco.sh"`
echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2012
#SBATCH --time=48:00:00

module load Anaconda3
module list
#conda init --all; conda activate

pwd
echo "--- working on p=$perco"
echo "starting cmd:" $codedir/perco_RandomCrosses.py $size $perco $perco 1 $configs

python --version
echo -e "print(1+2)" | python
python $codedir/perco_RandomCrosses.py $size $perco $perco 1 $configs

chmod -R g+w *

echo "--- finished with p=$perco"
EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
#(./${jobfile})

done
