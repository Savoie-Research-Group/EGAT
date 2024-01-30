import argparse
import yaml
import os,sys

def write(name,queue,nodes,gpu,cpus,tasks,days,hours,configfile,train_only=False,generate_only=False,cpu=False):
    fname = name.split('/')[-1]
    file = open(fname+'.submit','w')
    file.write('#!/bin/bash\n')
    file.write('#\n')
    file.write('#SBATCH --job-name={}\n'.format(fname))
    file.write('#SBATCH --output={}.out\n'.format(fname))
    file.write('#SBATCH -A {}\n'.format(queue))
    file.write('#SBATCH --nodes={}\n'.format(nodes))
    if not cpu:
        file.write('#SBATCH --gpus-per-node={}\n'.format(gpu))
        file.write('#SBATCH --cpus-per-gpu={}\n'.format(cpus))
    else:
        file.write('#SBATCH --cpus-per-task={}\n'.format(cpus))
    file.write('#SBATCH --ntasks-per-node={}\n'.format(tasks))
    file.write('#SBATCH --mem=200G\n')
    if queue == 'standby': 
        hours = '04'
        days = '00'
    if queue != 'debug':
        if days == '00':
            file.write('#SBATCH -t {}:00:00\n'.format(hours))
        else:
            file.write('#SBATCH -t {}-{}:00:00\n'.format(days,hours))
    else:
        file.write('#SBATCH -t 00:30:00\n')
        
    file.write('echo Running on host `hostname`\n')
    file.write('echo Time is `date`\n')
    
    file.write('module load cuda/11.7.0\n')
    file.write('module load cudnn/cuda-11.7_8.6\n')
    file.write('module load gcc/12.3.0\n')
    
    file.write('source activate yarp \n')
    file.write('cd /depot/bsavoie/data/Mahit-TS-Energy-Project/GitHub/EGAT \n')
    if not train_only:
        file.write('python Generate.py --config {} \n'.format(configfile))
        #file.write('python Split.py --config {} \n'.format(configfile))
    if not generate_only:
        file.write('python Train.py --config {} \n'.format(configfile))


    file.close()
    
    os.system('sbatch '+fname+'.submit')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hydra-based script with a config file argument")
    parser.add_argument("--config", type=str, default="config/test.yaml", help="Path to the config file")
    parser.add_argument("--name", type=str, default="Test", help="Name of Slurm File")
    parser.add_argument("--queue", type=str, default="standby", help="Slurm queue")
    parser.add_argument("--nodes", type=int, default=1, help="nodes to use")
    parser.add_argument("--gpu", type=int, default=1, help="GPU to use")
    parser.add_argument("--cpus", type=int, default=16, help="Processors to use")
    parser.add_argument("--tasks", type=int, default=10, help="Tasks to use")
    parser.add_argument("--days", type=str, default='00', help="Days to Run")
    parser.add_argument("--hours", type=str, default='04', help="Hours to Run")
    parser.add_argument("--train_only", action = 'store_true', help="Only do training")
    parser.add_argument("--gen_only", action = 'store_true', help="Only do training")
    parser.add_argument("--bell", action = 'store_true', help="Only do training")


    args = parser.parse_args()
    write(args.name,args.queue,args.nodes,args.gpu,args.cpus,args.tasks,args.days,args.hours,args.config,args.train_only,args.gen_only,args.bell)

