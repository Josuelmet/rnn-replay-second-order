The code to make figures is in the .ipynb notebook files.

To train the RNNs, run commands like these (with varying seeds):
```zsh
python train.py --dataset=triangle --seed=0 --unmask_every=3
python train.py --dataset=tmaze --seed=0 --unmask_every=3
```

Then, all analysis code is in the notebooks.