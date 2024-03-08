#!/bin/bash

dir=${1:-../data}
seed=${2:-12345684}
py=${3=Train-Pytorch-class.py}




codedir=`pwd`

echo "PERCO: dir=" $dir ",seed:"$seed ",py="$py

cd $dir

mkdir $seed
cd $seed
	
jobfile="training-"$seed".sh"

#echo $jobfile

	cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1


module restore TorchGPU_1_7_1
#conda init --all; conda activate

pwd
echo "--- working in directory=$seed"


python $codedir/$py $seed



#echo "--- finished in directory=  $seed"
EOD


cat ${jobfile}
chmod 755 ${jobfile}
chmod g+w ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch ${jobfile})
#(sbatch ${jobfile})
#(./${jobfile})
cd ..


cd $codedir

