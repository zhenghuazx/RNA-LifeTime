RNA-LifeTIme: A deep learning framework for RNA lifetime prediction.
===========
[![zheng](https://img.shields.io/badge/Author-Zheng.H-yellow)](https://zhenghuazx.github.io/hua.zheng/)

vLab is a package for RNA lifetime prediction task written in pytorch.

![](assets/Network.png)
RNA-LifeTime Model architecture: Arrows show the information flow among the various components described in this paper. Tensor shapes are shown with $N$ representing the number of residues, $C_t$ representing the number of types of native contacts, $C_g$ representing the number of Gaussians.

If you have any questions, or just want to chat about using the package,
please feel free to contact me in the [Website](https://zhenghuazx.github.io/hua.zheng/).
For bug reports, feature requests, etc., please submit an issue to the email <zheng.hua1@northeastern.edu>.

Installation
======================================
If you would like to build the library from the source code, then at the root `/RNA-LifeTime`, run
```shell
pip install build
python -m build
```
Then install RNA-LifeTime and clean up the directory
```shell
pip install dist/RNA_LifeTime-1.0.0-py3-none-any.whl
rm -r dist
```
Usage
======================================
### Usage Help
the main.py script provides useful help messages that describe the available command-line arguments and their usage.
```shell
python main.py --help
```
Read the help messages to check out hyperparamters:
```shell
usage: main.py [-h] [-e NUM_EPOCHS] [-b BATCH_SIZE] [--dtype DTYPE] [-lr LR] [-c CP_DIR] [-m MODEL_TYPE] [-p PATH] [-f PREPROCESSED] [-t TRUNCATED] [-g NUM_GAUSSIANS] [-l MAX_LENGTH] [-fh FEEDFORWARD_HIDDEN_MULTIPLIER] [-d DROPOUT] [-r SEED] [--log_interval LOGGING_INTERVAL]
               [--patience PATIENCE] [--step STEP_SIZE] [--gamma GAMMA]

optional arguments:
  -h, --help            show this help message and exit
  -e NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        Number of epochs.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size.
  --dtype DTYPE         Data type.
  -lr LR, --learning-rate LR
                        Learning rate
  -c CP_DIR, --checkpoint_dir CP_DIR
                        Whether to run on the real test set (if not included, the validation set is used).
  -g NUM_GAUSSIANS, --num_gau NUM_GAUSSIANS
                        Number of Gaussian used for modeling solvent-mediated interaction.
  -l MAX_LENGTH, --max_len MAX_LENGTH
                        Max sequence length. Shorter sequences are padded.
  -fh FEEDFORWARD_HIDDEN_MULTIPLIER, --ff_hidden_mult FEEDFORWARD_HIDDEN_MULTIPLIER
                        Scaling factor that determines the size of the hidden layers relative to the input layers.
  -d DROPOUT, --dropout DROPOUT
                        Dropout rate.
  -r SEED, --random-seed SEED
                        RNG seed. Negative for random
  --log_interval LOGGING_INTERVAL
                        Logging interval.
  --patience PATIENCE   Patience in early stopping.
  --step STEP_SIZE      Period of learning rate decay.
  --gamma GAMMA         Multiplicative factor of learning rate decay.
```

### Example
```python
python main.py -e 15 -b 512 -lr 0.003 -c ./MD-simulation/models/ -m RNA-LifeTime -p ./MD-simulation/ -f False -g 3 -l 72 -t False -d 0.2 -r 1 --step 3 --gamma 0.3
```
![img.png](assets/training.png)

