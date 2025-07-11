import copy
import numpy as np
import os
import sys
import shutil
from glob import glob
import pdb
import time

from dlnpyutils import utils as dln
from astropy.io import fits
from astropy.table import Table

import subprocess
from os import getlogin, getuid, getgid, makedirs, chdir
from pwd import getpwuid
from grp import getgrgid
from datetime import datetime,timezone
import string
import random
import traceback

SLURMDIR = '/scratch/general/nfs1/'

def genkey(n=20):
    characters = string.ascii_lowercase + string.digits
    key =  ''.join(random.choice(characters) for i in range(n))
    return key

def slurmstatus(label,jobid,username=None):
    """
    Get status of the job from the slurm queue
    """

    if username is None:
        username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    
    # Check if the slurm job is still running
    #  this will throw an exception if the job finished already
    #  slurm_load_jobs error: Invalid job id specified
    try:
        # previously long jobnames got truncated, make sure to get the full name
        form = 'jobid,jobname%30,partition%30,account,alloccpus,state,exitcode,nodelist'
        res = subprocess.check_output(['sacct','-u',username,'-j',str(jobid),'--allclusters','--format='+form])
    except:
        print('Failed to get Slurm status')
        #traceback.print_exc()
        return []
    if type(res)==bytes: res=res.decode()
    lines = res.split('\n')
    # remove first two lines
    lines = lines[2:]
    # only keep lines with the label in it
    lines = [l for l in lines if ((l.find(label)>-1) and (l.find(str(jobid))>-1))]
    # JobID    JobName  Partition    Account  AllocCPUS      State ExitCode
    out = np.zeros(len(lines),dtype=np.dtype([('JobID',str,30),('JobName',str,30),('Partition',str,30),
                                              ('Account',str,30),('AllocCPUS',int),('State',str,30),
                                              ('ExitCode',str,20),('Nodelist',str,30),('done',bool)]))
    # Each node gets its own line/row
    for i in range(len(lines)):
        dum = lines[i].split()
        out['JobID'][i] = dum[0]
        out['JobName'][i] = dum[1]
        out['Partition'][i] = dum[2]
        out['Account'][i] = dum[3]
        out['AllocCPUS'][i] = int(dum[4])
        out['State'][i] = dum[5]
        out['ExitCode'][i] = dum[6]
        out['Nodelist'][i] = dum[7]        
        if dum[5] == 'COMPLETED' or dum[5]=='TIMEOUT' or dum[5]=='CANCELLED':
            out['done'][i] = True

    return out

def taskstatus(label,key,username=None):
    """
    Return tasks that completed
    """

    if username is None:
        username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    jobdir = slurmdir+label+'/'+key+'/'
    
    # Get number of tasks
    ntasks = dln.readlines(jobdir+label+'.ntasks')
    ntasks = int(ntasks[0])
    
    # Load the task inventory
    #tasks = Table.read(jobdir+label+'_inventory.txt',names=['task','node','proc'],format='ascii')

    # Check the log files
    outfiles = glob(jobdir+'*'+label+'_*.out*')
    # Load the files
    tasknum = 0
    completedtasks = []
    for i in range(len(outfiles)):
        lines = dln.readlines(outfiles[i])
        # Get the "completed" lines
        clines = [l for l in lines if l.find('ompleted')>-1]
        for j in range(len(clines)):
            # Task 127 node01 proc64 Completed Sun Oct 16 19:30:01 MDT 2022
            dum = clines[j].split()
            taskid = dum[1]
            nodeid = dum[2][4:]
            procid = dum[3][4:]
            stat = [taskid,nodeid,procid]
            completedtasks += [stat]
            tasknum += 1

    if len(completedtasks):
        tstatus = np.zeros(len(completedtasks),dtype=np.dtype([('task',int),('node',int),('proc',int),('done',bool)]))
        for i,stat in enumerate(completedtasks):
            tstatus['task'][i] = stat[0]
            tstatus['node'][i] = stat[1]   # node01
            tstatus['proc'][i] = stat[2]   # proc54
            tstatus['done'][i] = True
    else:
        tstatus = []
    
    return tstatus
    
def status(label,key,jobid,username=None):
    """
    Return the status of a job.
    """

    if username is None:
        username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    jobdir = slurmdir+label+'/'+key+'/'
    
    # Check if the slurm job is still running
    state = slurmstatus(label,jobid)
    if len(state)==0:
        return None,None,None
    node = len(state)
    ndone = np.sum(state['done'])
    noderunning = node-ndone
    # Get number of tasks
    ntasks = dln.readlines(jobdir+label+'.ntasks')
    ntasks = int(ntasks[0])
    # Check how many tasks have completed
    tstatus = taskstatus(label,key)
    ncomplete = len(tstatus)
    taskrunning = ntasks-ncomplete
    percent = 100*ncomplete/ntasks
    return noderunning,taskrunning,percent
    
def queue_wait(label,key,jobid,sleeptime=60,logger=None,verbose=True):
    """
    Wait until the job is done
    """

    if logger is None:
        logger = dln.basiclogger()

    username = getpwuid(getuid())[0]
    slurmdir = SLURMDIR+username+'/slurm/'
    jobdir = slurmdir+label+'/'+key+'/'

    # Get number of tasks
    ntasks = dln.readlines(jobdir+label+'.ntasks')
    ntasks = int(ntasks[0])
    
    # While loop
    done = False
    count = 0
    while (done==False):
        time.sleep(sleeptime)
        # Check that state
        noderunning,taskrunning,percent = status(label,key,jobid)
        if noderunning is not None:
            # Check if the slurm job is still running
            state = slurmstatus(label,jobid)
            node = len(state)
            ndone = np.sum(state['done'])
            noderunning = node-ndone
            # Get number of tasks
            ntasks = dln.readlines(jobdir+label+'.ntasks')
            ntasks = int(ntasks[0])
            # Check how many tasks have completed
            tstatus = taskstatus(label,key)
            ncomplete = len(tstatus)
            taskrunning = ntasks-ncomplete
            percent = 100*ncomplete/ntasks
            if verbose:
                logger.info('percent complete = %2d   %d / %d tasks' % (percent,ntasks-taskrunning,ntasks))
        else:
            # It claims to not be running, but let's check anyway
            tstatus = taskstatus(label,key)
            ncomplete = len(tstatus)
            percent = 100*ncomplete/ntasks
            if verbose:
                logger.info('NOT Running  percent complete = %2d   %d / %d tasks' % (percent,ncomplete,ntasks))                
                
        # Are we done
        #  if the slurm job was canceled then we need to stop even if some tasks are not done yet
        if noderunning==0 or taskrunning==0:
            done = True

    # Check Slurm exit status for failure
    # Slurm exit codes:
    # 0 -> success
    # non-zero -> failure
    # Exit code 1 indicates a general failure
    # Exit code 2 indicates incorrect use of shell builtins
    # Exit codes 3-124 indicate some error in job (check software exit codes)
    # Exit code 125 indicates out of memory
    # Exit code 126 indicates command cannot execute
    # Exit code 127 indicates command not found
    # Exit code 128 indicates invalid argument to exit
    # Exit codes 129-192 indicate jobs terminated by Linux signals
    # For these, subtract 128 from the number and match to signal code
    # Enter kill -l to list signal codes
    # Enter man signal for more information
    if state['ExitCode'][0] == '0:0':
        logger.info('Slurm job succeeded')
    else:
        msg = 'Slurm job failed with State='+str(state['State'][0])
        msg += ' and ExitCode='+str(state['ExitCode'][0])
        logger.info(msg)
        
def submit(tasks,label,nodes=1,cpus=64,ppn=None,account='priority-davidnidever',
           partition='priority',shared=True,walltime='12-00:00:00',notification=False,
           memory=7500,numpy_num_threads=2,stagger=True,nodelist=None,precommands=None,
           postcommands=None,slurmroot='/tmp',verbose=True,logger=None):
    """
    Submit a bunch of jobs

    tasks : table
      Table with the information on the tasks.  Must have columns of:
        cmd, outfile, errfile, dir (optional)

    """

    if logger is None:
        logger = dln.basiclogger()

    if ppn is None:
        ppn = np.minimum(64,cpus)
    slurmpars = {'nodes':nodes, 'account':account, 'shared':shared, 'ppn':ppn,
                 'cpus':cpus, 'walltime':walltime, 'partition':partition,
                 'notification':notification}

    username = getpwuid(getuid())[0]
    slurmdir = os.path.join(slurmroot,username,'slurm')
    if os.path.exists(slurmdir)==False:
        os.makedirs(slurmdir)

    # Generate unique key
    key = genkey()
    if verbose:
        logger.info('key = '+key)
    # make sure it doesn't exist yet

    # job directory
    jobdir = os.path.join(slurmdir,label,key)
    if os.path.exists(jobdir)==False:
        os.makedirs(jobdir)

    # Start .slurm files

    # nodeXX.slurm that sources the procXX.slurm files    
    # nodeXX_procXX.slurm files with the actual commands in them

    # Figure out number of tasks per cpu
    ntasks = len(tasks)

    # Add column to tasks table
    if isinstance(tasks,Table)==False:
        tasks = Table(tasks)
    tasks['task'] = -1
    tasks['node'] = -1
    tasks['proc'] = -1
    
    # Parcel out the tasks to the nodes+procs
    #  -try to use as many cpus as possible
    #  -loop over all the nodes+procs until we've
    #    exhausted all of the tasks
    count = 0
    while (count<ntasks):
        for i in range(nodes):
            node = i+1
            for j in range(ppn):
                proc = j+1
                if count>=ntasks: break
                tasks['task'][count] = count+1
                tasks['node'][count] = node
                tasks['proc'][count] = proc
                count += 1
                
    # Node loop
    node_index = dln.create_index(tasks['node'])    
    tasknum = 0
    inventory = []
    for i in range(len(node_index['value'])):
        nind = node_index['index'][node_index['lo'][i]:node_index['hi'][i]+1]
        nind = np.sort(nind)
        node = node_index['value'][i]
        nodefile = 'node%02d.slurm' % node
        nodename = 'node%02d' % node        
        # Number of proc files
        proc_index = dln.create_index(tasks['proc'][nind])
        nproc = len(proc_index['value'])

        # Create the lines
        lines = []
        lines += ['#!/bin/bash']
        lines += ['# Auto-generated '+datetime.now().ctime()+' -- '+label+' ['+nodefile+']']
        if account is not None:
            lines += ['#SBATCH --account='+account]
        if partition is not None:
            lines += ['#SBATCH --partition='+partition]
        lines += ['#SBATCH --nodes=1']
        lines += ['#SBATCH --ntasks='+str(nproc)]
        lines += ['#SBATCH --mem-per-cpu='+str(memory)]
        lines += ['#SBATCH --cpus-per-task=1']
        lines += ['#SBATCH --time='+walltime]
        lines += ['#SBATCH --job-name='+label]
        lines += ['#SBATCH --output='+label+'_%j.out']
        lines += ['#SBATCH --err='+label+'_%j.err']
        lines += ['# ------------------------------------------------------------------------------']
        lines += ['export OMP_NUM_THREADS=2']
        lines += ['export OPENBLAS_NUM_THREADS=2']
        lines += ['export MKL_NUM_THREADS=2']
        lines += ['export VECLIB_MAXIMUM_THREADS=2']
        lines += ['export NUMEXPR_NUM_THREADS=2']
        lines += ['# ------------------------------------------------------------------------------']
        lines += ['export CLUSTER=1']
        # Adding extra command to execute
        if precommands is not None:
            if type(precommands) is not list:
                precommands = [precommands]
            lines += precommands
        lines += [' ']
        for j in range(nproc):
            proc = proc_index['value'][j]
            procfile = 'node%02d_proc%02d.slurm' % (node,proc)
            lines += ['source '+os.path.join(jobdir,procfile)+' &']
        lines += ['wait']
        lines += ['echo "Done"']
        if verbose:
            logger.info('Writing '+os.path.join(jobdir,nodefile))
        dln.writelines(os.path.join(jobdir,nodefile),lines)
        
        # Create the proc files
        for j in range(nproc):
            proc = proc_index['value'][j]
            pind = proc_index['index'][proc_index['lo'][j]:proc_index['hi'][j]+1]   # proc index, subset of nind
            pind = np.sort(pind)
            procname = 'proc%02d' % proc
            procfile = 'node%02d_proc%02d.slurm' % (node,proc)
            lines = []
            lines += ['# Auto-generated '+datetime.now().ctime()+' -- '+label+' ['+procfile+']']
            if stagger:
                lines += ['sleep '+str(int(np.ceil(np.random.rand()*20)))]
            lines += ['cd '+jobdir]            
            # Loop over the tasks
            for k in range(len(pind)):
                tind = nind[pind][k]   # task index into "tasks" table
                lines += ['# ------------------------------------------------------------------------------']
                lines += ['echo "Running task '+str(tasks['task'][tind])+' '+nodename+' '+procname+'" `date`']                
                if 'dir' in tasks.colnames:
                    lines += ['cd '+tasks['dir'][tind]]
                cmd = tasks['cmd'][tind]+' > '+tasks['outfile'][tind]+' 2> '+tasks['errfile'][tind]
                lines += [cmd]
                lines += ['echo "Task '+str(tasks['task'][tind])+' '+nodename+' '+procname+' Completed" `date`']
                lines += ['echo "Done"']
                if os.path.exists(os.path.dirname(tasks['outfile'][tind]))==False:  # make sure output directory exists
                    try:
                        os.makedirs(os.path.dirname(tasks['outfile'][tind]))
                    except:
                        logger.info('Problems making directory '+os.path.dirname(tasks['outfile'][tind]))
                inventory += [str(tasks['task'][tind])+' '+str(node)+' '+str(proc)]
                tasknum += 1
            lines += ['cd '+jobdir]                            
            if verbose:
                logger.info('Writing '+os.path.join(jobdir,procfile))
            dln.writelines(os.path.join(jobdir,procfile),lines)

    # Create the "master" slurm file
    masterfile = label+'.slurm'
    lines = []
    lines += ['#!/bin/bash']
    lines += ['# Auto-generated '+datetime.now().ctime()+' ['+masterfile+']']
    if account is not None:
        lines += ['#SBATCH --account='+account]
    if partition is not None:
        lines += ['#SBATCH --partition='+partition]
    lines += ['#SBATCH --nodes=1']
    lines += ['#SBATCH --ntasks='+str(nproc)]
    lines += ['#SBATCH --mem-per-cpu='+str(memory)]
    lines += ['#SBATCH --cpus-per-task=1']
    lines += ['#SBATCH --time='+walltime]
    lines += ['#SBATCH --array=1-'+str(nodes)]
    lines += ['#SBATCH --job-name='+label]
    lines += ['#SBATCH --output='+label+'_%A[%a].out']
    lines += ['#SBATCH --err='+label+'_%A[%a].err']
    lines += ['# ------------------------------------------------------------------------------']
    lines += ['export OMP_NUM_THREADS=2']
    lines += ['export OPENBLAS_NUM_THREADS=2']
    lines += ['export MKL_NUM_THREADS=2']
    lines += ['export VECLIB_MAXIMUM_THREADS=2']
    lines += ['export NUMEXPR_NUM_THREADS=2']
    lines += ['# ------------------------------------------------------------------------------']
    # Adding extra command to execute at beginning
    if precommands is not None:
        if type(precommands) is not list:
            precommands = [precommands]
        lines += precommands
    lines += ['SBATCH_NODE=$( printf "%02d']
    lines += ['" "$SLURM_ARRAY_TASK_ID" )']
    lines += ['source '+jobdir+'/node${SBATCH_NODE}.slurm']
    # Adding extra command to execute at end
    if postcommands is not None:
        if type(postcommands) is not list:
            postcommands = [postcommands]
        lines += postcommands
    lines += [' ']

    if verbose:
        logger.info('Writing '+os.path.join(jobdir,masterfile))
    dln.writelines(os.path.join(jobdir,masterfile),lines)

    # Write the number of tasks
    dln.writelines(os.path.join(jobdir,label+'.ntasks'),ntasks)

    # Write the inventory file
    dln.writelines(os.path.join(jobdir,label+'_inventory.txt'),inventory)

    # Write the tasks list
    tasks.write(os.path.join(jobdir,label+'_tasks.fits'),overwrite=True)
    # Write the list of logfiles
    dln.writelines(os.path.join(jobdir,label+'_logs.txt'),list(tasks['outfile']))
    
    # Now submit the job
    if verbose:
        logger.info('Submitting '+os.path.join(jobdir,masterfile))
    # Change to the job directory, because that's where the outputs will go
    curdir = os.path.abspath(os.curdir)
    os.chdir(jobdir)
    # Sometimes sbatch can fail for some reason
    #  if that happens, retry
    scount = 0
    success = False
    res = ''
    while (success==False) and (scount < 30):
        if scount>0:
            logger.info('Trying to submit to SLURM again')
        try:
            res = subprocess.check_output(['sbatch',os.path.join(jobdir,masterfile)])
            success = True
        except:
            logger.info('Submitting job to SLURM failed with sbatch.')
            success = False
            tb = traceback.format_exc()
            logger.info(tb)
            if scount<10:
                time.sleep(60)    # 1 min wait time
            elif scount>=10 and scount<20:
                time.sleep(300)   # 5 min wait time
            else:
                time.sleep(1800)  # 30 min wait time       
        scount += 1
    os.chdir(curdir)   # move back
    if type(res)==bytes: res = res.decode()
    res = res.strip()  # remove \n
    if verbose:
        logger.info(res)
    # Get jobid
    #  Submitted batch job 5937773 on cluster notchpeak
    jobid = res.split()[3]

    if verbose:
        logger.info('key = '+key)
        logger.info('job directory = '+jobdir)
        logger.info('jobid = '+jobid)
        
    return key,jobid


def launcher(tasks,label,nodes=1,nparallel=None,cpus=64,ppn=None,account='priority-davidnidever',
             partition='priority',shared=True,walltime='12-00:00:00',notification=False,
             memory=7500,numpy_num_threads=2,stagger=True,nodelist=None,precommands=None,
             postcommands=None,slurmroot='/tmp',verbose=True,nosubmit=False,logger=None):
    """
    Submit a Launcher slurm job with many serial tasks

    tasks : table
      Table with the information on the tasks.  Must have columns of:
        cmd, outfile, errfile, dir (optional)

    """


    if logger is None:
        logger = dln.basiclogger()

    if ppn is None:
        ppn = np.minimum(64,cpus)
    slurmpars = {'nodes':nodes, 'account':account, 'shared':shared, 'ppn':ppn,
                 'cpus':cpus, 'walltime':walltime, 'partition':partition,
                 'notification':notification}

    username = getpwuid(getuid())[0]
    slurmdir = os.path.join(slurmroot,username,'slurm')
    if os.path.exists(slurmdir)==False:
        os.makedirs(slurmdir)

    # Run 50 tasks per node, by default
    if nparallel is None:
        nparallel = nodes*50

    # Generate unique key
    key = genkey()
    if verbose:
        logger.info('key = '+key)
    # make sure it doesn't exist yet

    # job directory
    jobdir = os.path.join(slurmdir,label,key)
    if os.path.exists(jobdir)==False:
        os.makedirs(jobdir)

    # Figure out number of tasks per cpu
    ntasks = len(tasks)

    # Make single list of commands

    # Task loop
    lines = []
    for i in range(ntasks):
        line = ''
        if 'dir' in tasks.colnames:
            line += 'cd '+tasks['dir'][i]+'; '
        cmd = tasks['cmd'][i]+' > '+tasks['outfile'][i]+' 2> '+tasks['errfile'][i]
        line += cmd
        if os.path.exists(os.path.dirname(tasks['outfile'][i]))==False:  # make sure output directory exists
            try:
                os.makedirs(os.path.dirname(tasks['outfile'][i]))
            except:
                logger.info('Problems making directory '+os.path.dirname(tasks['outfile'][tind]))
        lines += [line]
    jobsfile = os.path.join(jobdir,label+'_cmd.lst')
    if verbose:
        logger.info('Writing '+jobsfile)
    dln.writelines(jobsfile,lines)

    # Write the tasks list
    tasks.write(os.path.join(jobdir,label+'_tasks.fits'),overwrite=True)
    # Write the list of logfiles
    dln.writelines(os.path.join(jobdir,label+'_logs.txt'),list(tasks['outfile']))

    # Make Launcher slurm script 

    # Create the "master" slurm file
    masterfile = label+'.slurm'
    lines = []
    lines += ['#!/bin/bash']
    lines += ['# Auto-generated '+datetime.now().ctime()+' ['+masterfile+']']
    if account is not None:
        lines += ['#SBATCH --account='+account]
    if partition is not None:
        lines += ['#SBATCH --partition='+partition+'  # queue (partition)']
    lines += ['#SBATCH --job-name='+label]
    lines += ['#SBATCH -N '+str(nodes)+'           # number of nodes requested']
    lines += ['#SBATCH -n '+str(nparallel)+'          # total number of tasks to run in parallel']
    lines += ['#SBATCH -t '+str(walltime)+'        # run time (hh:mm:ss)']
    #lines += ['#SBATCH --mem-per-cpu='+str(memory)]
    lines += ['#SBATCH --output='+label+'-%j.out']
    lines += ['#SBATCH --err='+label+'-%j.err']
    lines += ['']
    # Adding extra command to execute
    if precommands is not None:
        if type(precommands) is not list:
            precommands = [precommands]
        lines += precommands
    lines += ['module load launcher']
    lines += ['']
    lines += ['export LAUNCHER_WORKDIR='+jobdir]
    lines += ['export LAUNCHER_JOB_FILE='+jobsfile]
    lines += ['']
    lines += ['${LAUNCHER_DIR}/paramrun']
    # Adding extra command to execute at end
    if postcommands is not None:
        if type(postcommands) is not list:
            postcommands = [postcommands]
        lines += postcommands
    lines += [' ']
    lines += ['echo "Done"']
    if verbose:
        logger.info('Writing '+os.path.join(jobdir,masterfile))
    dln.writelines(os.path.join(jobdir,masterfile),lines)

    if nosubmit==False:
        # Now submit the job
        logger.info('Submitting '+os.path.join(jobdir,masterfile))
        # Change to the job directory, because that's where the outputs will go
        curdir = os.path.abspath(os.curdir)
        os.chdir(jobdir)
        try:
            res = subprocess.check_output(['sbatch',os.path.join(jobdir,masterfile)])
            success = True

            if type(res)==bytes: res = res.decode()
            res = res.strip()  # remove \n
            if verbose:
                logger.info(res)
            # Get jobid
            #  Submitted batch job 5937773 on cluster notchpeak
            res = res.split('\n')[-1]
            jobid = res.split()[3]
            if verbose:
                logger.info('jobid = '+jobid)

        except:
            logger.info('Submitting job to SLURM failed with sbatch.')
            success = False
            tb = traceback.format_exc()
            logger.info(tb)
            jobid = -1

        return slurmdir,key,jobid

    return slurmdir,key
