When you want to run IPPO or MAPPO, use `trian_mpe_all.py`

Only params to change EVER for `train_mpe_all.py`:

* `device`, if you want to run on CPU, pass command "`--cuda`"
* `--algorithm_name`, this decides the algo. options:{ippo, mappo}
* `--experiment_name`, this helps track what the runs are, what params they have. Name wisely.
* `--n_rollout_threads`, this is number of parallel environments
* line 111: `share_policy`, False for separated, True for shared params (networks). 

How to run:

If you want to run a sequential seed list say from 1-5 inclusive (with same params), then you can run command:

```python "path-to-train_mpe_all.py" --start_seed 1 --end_seed 5```

This will run a loop starting at seed = 1, and end with the last run being seed = 5. This assumes the parameters you want to run are already saved in the file.

Another command to run one specific seed is to run

```python "path-to-train_mpe_all.py" --start_seed 1```

This is if you want to only run seed 1. 

These are all you need.
