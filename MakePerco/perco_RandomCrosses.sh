#!/bin/bash

dir=${1:-../Data}  

#seed=${2:-1234567}
size=${2:-10}
perco_i=${3:-05927}
perco_f=${4:-06000}
dperco=${5:-02000}
configs=${6:-1}
hlines=${7:-0}
hthick=${8:-0}
vlines=${9:-0}
vthick=${10:-0}
ulines=${11:-0}
uthick=${12:-0}
dlines=${13:-0}
dthick=${14:-0}
typemod=${15:-1}

if [ $typemod -ge 0 ]
then
    typechar="L"
else
    typechar="A"
fi

codedir=`pwd`

echo "PERCO: dir=" $dir ", size=" $size \
", [Ti,Tf,dT]= [" $perco_i, $perco_f, $dperco "], configs=" $configs", py_code=" $pyfile

mkdir -p $dir
cd $dir

datadir="L"$size"_h"$hlines"-"$hthick"_v"$vlines"-"$vthick"-u"$ulines"-"$uthick"_d"$dlines"-"$dthick"_"$typechar

echo $datadir
mkdir -p $datadir
cd $datadir

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

module restore new_TorchGPU_1_7_1
#module list
#conda init --all; conda activate

pwd
echo "--- working on p=$perco"
echo "starting cmd:" $codedir/perco_RandomCrosses.py $size $perco $perco 1 $configs $hlines $hthick $vlines $vthick $ulines $uthick $dlines $dthick $typemod

#python --version
#echo -e "print(1+2)" | python
python $codedir/perco_RandomCrosses.py $size $perco $perco 1 $configs $hlines $hthick $vlines $vthick $ulines $uthick $dlines $dthick $typemod

chmod -R g+w *

echo "--- finished with p=$perco"
EOD

#cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
#(./${jobfile})

done
