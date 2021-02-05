#!/bin/bash

dir=${1:-../data}
size=${2:-10}
dir2=${3:-/p_occ}
#filename=${4:-filename}  

codedir=`pwd`

echo "PERCO: dir=" $dir" , size=" $size" , dir2=" $dir2


cd $dir
cd "L"$size
cd $dir2






EXT=pkl
EXT_txt=txt

for files_pkl, files_txt in *.${EXT} and *.${EXT_txt}
do 
echo "---  dir=" $dir2

jobfile=`printf "$dir2.sh"`
echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2012
#SBATCH --time=48:00:00

module load Anaconda3
#conda init --all; conda activate

pwd
echo "--- working in directory=$dir2"

python $codedir/perco_generate_im.py $files_pkl $files_txt

echo "--- finished in directory= " $dir2
EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
#(./${jobfile})

done
