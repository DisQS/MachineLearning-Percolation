#!/bin/bash

dir=${1:-../data}
size=${2:-10}
  

codedir=`pwd`

echo "PERCO: dir=" $dir ", size=" $size


cd $dir
cd "L"$size







EXT=pkl
EXT_txt=corr_func.txt


for directory in */
do

for files1 in $directory*.pkl
do 
echo $files1

for files2 in $directory*corr_func.txt
do
echo $files2


jobfile=`printf "$images.sh"`
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
echo "--- working in directory=$directory"

python $codedir/perco_generate_im.py $files1 $files2

echo "--- finished in directory=  $directory"
EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
#(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
(./${jobfile})

done
done
done
