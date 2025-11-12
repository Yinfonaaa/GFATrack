
## Install the environment
 Use the Anaconda (CUDA 10.2)
```
conda create -n gfatrack python=3.8
conda activate gfatrack
bash install.sh
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training

python tracking/train.py --script gfatrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/gfatrack`. Set `--use_wandb 0`.


## Evaluation

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py gfatrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py gfatrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name gfatrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
```
- TrackingNet
```
python tracking/test.py gfatrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name gfatrack --cfg_name vitb_384_mae_ce_32x4_ep300
```

## Visualization or Debug 
1. Alive visdom in the sever by running `visdom`:

2. Simply set `--debug 1` during inference for visualization, e.g.:
```
python tracking/test.py gfatrack gfatrack384_elimination_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1
```
3. Open `http://localhost:8097` in your browser (remember to change the ip address and port according to the actual situation).

4. Then you can visualize the candidate elimination process.







