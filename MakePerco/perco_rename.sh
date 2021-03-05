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



jobfile="rename-"$size-$directory".sh"

#echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2012
#SBATCH --time=48:00:00

if [ $option = 1 ]; then
  rename _a.png .apng *_a.png
  rename _b.png .bpng *_b.png
  rename _s.png .spng *_s.png
else
  rename .png .npng *.png
fi

#chmod -R g+w *
echo "--- finished in directory=$directory"
EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
#(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
(./${jobfile})

cd ..
done

cd $codedir

