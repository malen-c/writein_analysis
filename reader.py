print('Gathering corners of template ballots...')

from funs import *
import pandas as pd
from multiprocessing.pool import ThreadPool as Pool
import subprocess
import os
import numpy as np
import time

#Commented sections are configuration for continual downloading from shared drive,
#this script has been altered to instead pull from the optical drive

start = time.time()

pool_size = 16



image_folder = './templates/' #PATH TO FOLDER CONTAINING BALLOT SCANS GOES HERE. Make sure it ends in a slash ('/') !
template_folder = './templates/'

#shared_drive_path = 'EVIC_Data2\\MultCo_Drive_Feb2021\\writein_images\\election20201103\\'

df = pd.read_csv('writeins.csv')
df['teressa'] = 'NA'
df['min_lev'] = 'NA'
df['best_guess'] = 'NA'

chunks = np.array_split(df, 4700)

counter = 0


"""

def download(ids):
    for i in ids:
        subprocess.call(['rclone', 'copy',
                         'evic:{0}{1}.jpg'.format(shared_drive_path, i),
                         'C:\\Users\\Malen\\Downloads\\Ballot CSVs\\scary zone\\'],
                        shell = True)
"""

def worker(chunk):
    #download(chunk.iloc[0:,0])
    for i, row in chunk.iterrows():
        ballot_path = '{0}{1}.jpg'.format(image_folder, row[0])
        try:
            best_guess, min_lev = read_ballot(ballot_path, row[1])
            if min_lev > 8:
                tr_flag = 0
            else:
                tr_flag = 1
            chunk.at[i, 'teressa'] = tr_flag
            chunk.at[i, 'min_lev'] = min_lev
            chunk.at[i, 'best_guess'] = best_guess
        except:
            pass
        #try:
        #    os.remove(ballot_path)
        #except:
        #    print(ballot_path + ' failed to delete.')
    
    global counter
    counter += 1

    if counter % 100 == 0:
        print('{0}/4700 chunks processed, {1}% done'.format(counter, int(counter/47)))
        print('{0} minutes elapsed...'.format(str((time.time() - start)/60).split('.')[0]))

pool = Pool(pool_size)

#making sure image folder was configured before running
if image_folder != '':
    for chunk in chunks:
        pool.apply_async(worker, (chunk,))
else:
    print('Specify image_folder!')

pool.close()
pool.join()

results = pd.concat(chunks)
results.to_csv('results.csv', index = False)

#Sample of 1000 had 2/660 false positive, 46/331 false negative

