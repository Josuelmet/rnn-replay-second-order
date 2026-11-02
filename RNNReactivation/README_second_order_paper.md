To reproduce experiments from my paper, the commands to train models have syntax like this:
```zsh
# Train vanilla RNN on unbiased spatial navigation with seed=0 and unmasking every 6 observations
python train.py config=spatial_navigation/noisy_unbiased seed=0 rnn.bias=True trainer.n_epochs=15000 trainer.unmask_every=6
# Train vanilla RNN on biased spatial navigation with seed=0 and unmasking every 3 observations
python train.py config=spatial_navigation/noisy_biased seed=0 rnn.bias=True trainer.n_epochs=15000 trainer.unmask_every=3
```

Then, all analysis is in the "notebooks" folder.
