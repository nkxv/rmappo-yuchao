When you want to run IPPO or MAPPO, use `trian_mpe_all.py`

Only params to change EVER for `train_mpe_all.py`:

* line 54: `device`, if you want to run on CPU, pass command "`--cuda`"
* line 56: `algo name`, this decides the algo. options:{ippo, mappo}
* line 57: `experim name`, this helps track what the runs are, what params they have. Name wisely.
* line 60: `rollout threads`, this is number of parallel environments
* line 110: `share_policy`, False for separated, True for shared params (networks). 

How to run:

If you want to run a sequential seed list say from 1-5 inclusive (with same params), then you can run command:

```python "path-to-train_mpe_all.py" --start_seed 1 --end_seed 5```

This will run a loop starting at seed = 1, and end with the last run being seed = 5. This assumes the parameters you want to run are already saved in the file.

Another command to run one specific seed is to run

```python "path-to-train_mpe_all.py" --start_seed 1```

This is if you want to only run seed 1. 

These are all you need.
