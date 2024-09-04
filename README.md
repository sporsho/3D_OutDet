# 3D_OutDet
## Experiments on WADS
We found a huge amount of duplicate points in WADS dataset which distorts the nearest neighborhood hence affecting the performance of all the algorithms that uses kNN e.g., SOR, ROR, DSOR, DROR, 3D OutDet, SalsaNext etc. 
Hence, we need to remove the duplicate data points as the first step of data processing. 

### Duplicate Removal
1. Download the original WADS from https://digitalcommons.mtu.edu/wads/ (thanks for Kurup & Bos). I put the sequences in a folder named ```WADS```
2. Find out the file named ```remove_duplicate.py```, I put it in ```3D_OutDet/dataset/``` folder. 
3. Open the ```remove_duplicate.py``` file and change the ```src_root``` variable pointing at your ```sequences``` folder from step 1. If you had used a different name for the folder, you need to modify the script accordingly. 
4. Run ```remove_duplicate.py``` script however you like, e.g., (from PyCharm, from Terminal or from some other IDE) 
5. You will find your duplicate free data in ```WADS2``` folder in the same parent directory. 

Since the KD Tree remains the same for the same point cloud during training, there is no point in calculating the KD Tree for every epoch. Rather we will save the nearest neighbors from KD tree for training. This would occupy some space in you HDD / SSD, so keep it in mind. 
### Pre-Compute kNN 
1. Find out the file named ```generate_knn_dist_wads.py```. I put it in ```3D_OutDet/dataset/utils``` folder. 
2. Open ```generate_knn_dist_wads.py``` and update ```data_dir``` in the argument parser pointing at the ```WADS2``` folder created in the previous major step. 
3. Run ```generate_knn_dist_wads.py```. It will pre-compute kNN for all the sequences. 

### Train on WADS 
1. Find and open the file ```train_wads.py```. Change ```data_dir``` and ```mode_save_path``` accordingly. 
2. Run ```train_wads.py```
3. Wait for the training to finish. 

### Evaluation on WADS
1. Find and open the file ```eval_wads.py```. 
2. Change ```data_dir```, ```model_save_path```, ```test_output_path``` accordingly.
3. Run ```eval_wads.py```
4. Wait for the evaluation to finish. 

# UPDATE 
We added RandLA-Net on our WADS benchmark. 

| | Precision | Recall | $F_1$ | mIOU |
| ------------ | -------- | --------- | ------ | ------- | 
| RandLa-Net | 69.99 | 96.86 | 81.26 | 68.43 | 
| 3D-OutDet |  96.78 | 92.76 | 94.73 | 90.00 | 
