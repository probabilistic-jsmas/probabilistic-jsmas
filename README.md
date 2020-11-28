# Probabilistic Jacobian-based Saliency Maps Attacks

Original implementation of WJSMA and TJSMA as decribed in this [paper](https://arxiv.org/abs/2007.06032).

## Installation

Please install the packages required by the `requirements.txt` file.

The complete repository including large files can be found [here](https://drive.google.com/file/d/1n_kpv64H4pdyFiWjilBu5qzpV8yOzbF_/view?usp=).

## How to use
### White box experiments
#### Train a model

A model can be trained by running the following command

```python start_train.py```

Arguments
 - `--model` the name of the joblib file
 - `--dataset` the dataset used (by default, mnist / cifar10 / gtsrb, but you can create augmented datasets as it is shown below)
 - `--epochs` the number of epochs (6 by default)
 - `--batchsize` the size of the training batches (128 by default)
 - `--lr` the learning rate used (0.001 by default)
 - `--smoothing` the label smoothing used (0.1 by default)
 
The trained models are stored in the `models/joblibs/` folder (see [File format](#file-format) for more details). 

#### Attack a model

You can perform the attacks presented in the article by using the following command

```python start_attack.py```

Arguments
 - `--attack` the name of the attack (either jsma / wjsma / tjsma / jsma_nt / wjsma_nt / tjsma_nt / mjsma / mwjsma)
 - `--model` the name of the model joblib file
 - `--dataset` the dataset of which the adversarial samples will be crafted
 - `--settype` the type of set used (either train or test)
 - `--firstindex` the index of the first attacked sample
 - `--lastindex` the index of the last attacked sample
 - `--batchsize` the size of the adversarial batches
 - `--theta` the amount by which the pixel are modified, which can either be positive or negative (not used for mjsma and mwjsma)
 - `--clipmin` minimum component value for clipping
 - `--clipmax` maximum component value for clipping
 - `--maxiter` maximum iteration before the attack stops
 - `--uselogits` uses the logits (Z variation) when set to true and the softmax (F variation) values otherwise (only used for non-targeted attacks)
 - `--nonstop` force the attack to keep going until the maximum iteration is reached when set to true

The crafted sample are stored in the `results/` folder (see [File format](#file-format) for more details). 

#### Evaluating the performances of an attack

You can evaluate the performances of an attack (like the average L0 distortion) by running the following command

```python start_stats.py```

Arguments
 - `--attack` the name of the folder containing the crafted adversarial samples
 - `--threshold` the L0 threshold below which an attack is considered successful (114 by default, corresponding to the threshold taken in the article)

#### Augment a dataset

Augmented datasets are used to train more robust models. You can add extra samples to the existing datasets by running the following command

```python start_augment.py```

Arguments
 - `--samples` the name of the folder containing the crafted adversarial samples
 - `--dataset` the dataset that will be augmented
 - `--name` the name of the new augmented dataset
 - `--spp` the number of extra samples per class (2000 by default as taken in the article)
 - `--threshold` the L0 threshold below which an attack is considered successful. Only successful samples will be added to the set (114 by default, corresponding to the threshold taken in the article)

The augmented datasets are stored in the `datasets/joblibs/` folder (see [File format](#file-format) for more details). 

### Black box experiments
#### Train a substitute model

A substitute model can be trained using Jacobian Based Dataset Augmentation (see https://arxiv.org/pdf/1602.02697.pdf) using the following command

```python start_train_substitute.py```

Arguments
 - `--model` the name of the joblib file of the substitute model
 - `--oracle` the name of the joblib file of the existing oracle model
 - `--dataset` the dataset used (gtsrb by default)
 - `--epochs` the number of epochs (25 by default)
 - `--batchsize` the size of the training batches (128 by default)
 - `--lr` the learning rate used (0.001 by default)
 - `--smoothing` the label smoothing used (0.1 by default)
 - `--jbdaepochs` the number of JBDA epochs (6 by default)
 - `--jbdabatchsize` the size of the JBDA batches (32 by default)
 - `--lamb` the lambda factor of the JBDA (0.1 by default)

The substitute models are also stored in the `models/joblibs/` folder (see [File format](#file-format) for more details).
Note that you can only train a substitute model for gtsrb. For other dataset, please look at `models/model_utils.py`.

#### Attack a substitute model

Attacks on substitute uses a different stop condition as described in the article.
Note that only non-targeted attacks can be used with the substitute models, even though through simple tweaking, the code of the targeted attacks can be adapted to substitute models.
To attack a substitute model, use the following command:

```python start_attack_substitute.py```

Arguments
 - `--attack` the name of the attack (either jsma_nt / wjsma_nt / tjsma_nt / mjsma / mwjsma)
 - `--model` the name of the substitute model joblib file
 - `--oracle` the name of the oracle model joblib file
 - `--dataset` the dataset of which the adversarial samples will be crafted
 - `--settype` the type of set used (either train or test)
 - `--firstindex` the index of the first attacked sample
 - `--lastindex` the index of the last attacked sample
 - `--batchsize` the size of the adversarial batches
 - `--theta` the amount by which the pixel are modified, which can either be positive or negative (not used for mjsma and mwjsma)
 - `--clipmin` minimum component value for clipping
 - `--clipmax` maximum component value for clipping
 - `--maxiter` maximum iteration before the attack stops
 - `--uselogits` uses the logits (Z variation) when set to true and the softmax (F variation) values otherwise (only used for non-targeted attacks)
 - `--nonstop` force the attack to keep going until the maximum iteration is reached when set to true (set to true by default for black box attacks)
 
The crafted sample are stored in the `results/` folder (see [File format](#file-format) for more details). 

#### Evaluating the performance of an attack against a substitute model and the oracle

Since the L0 is no longer a stop condition, the success and transferability rate are computed using the following command:

```python start_stats_substitute.py```

Arguments
 - `--attack` the name of the folder containing the crafted adversarial samples
 
### Example of use
#### White box

Let's say we want to test the performances of TJSMA against a model trained on CIFAR10. First we train a new model (or you can use the default model named cifar10)

```python start_train.py --model cifar10 --dataset cifar10```

This will create a joblib file containing the model in `models/joblibs/` named `cifar10.joblib`.
The we craft the adversarial samples on the first 10000 images of the train set using a batch size of 100:

```python start_attack.py --model cifar10 --dataset cifar10 --settype train --attack tjsma --firstindex 0 --lastindex 10000 --batchsize 100```

The adversarial samples will be saved in `results/tjsma_cifar10_cifar10_train_1.0/`. We can check the performances of the attack by running

```python start_stats.py --attack tjsma_cifar10_cifar10_train_1.0```

Now we will augment the CIFAR10 dataset with 20000 adversarial samples (2000 samples per class)

```python start_augment.py --samples tjsma_cifar10_cifar10_train_1.0 --dataset cifar10 --name cifar10_augmented --spp 2000```

This will create a joblib file containing the augmented CIFAR10 dataset in `datasets/joblibs/` named `cifar10_augmented`.
We can train on new model on the augmented dataset:

```python start_train.py --model cifar10_defense --datatset cifar10_augmented```

Again, this will create a joblib file containing the model in `models/joblibs/` named `cifar10_defense.joblib`.
We can then attack the new model and check the performances on the two models. This we will attack the first 1000 images of the test set

```
python start_attack.py --model cifar10 --dataset cifar10 --settype test --attack tjsma --firstindex 0 --lastindex 1000 --batchsize 100
python start_attack.py --model cifar10_defense --dataset cifar10 --settype test --attack tjsma --firstindex 0 --lastindex 1000 --batchsize 100
python start_stats.py --attack tjsma_cifar10_cifar10_test_1.0
python start_stats.py --attack tjsma_cifar10_defense_cifar10_test_1.0
```

#### Black box

First we will train a model which will be used as our oracle over gtsrb:

```python start_train.py --model gtsrb --dataset gtsrb```

Then we have to train a substitute model that will try to mimic our previously trained oracle

```python start_train_substitute.py --model gtsrb_substitute --oracle gtsrb --dataset gtsrb --jbdaepochs 6 --lamb 0.1```

As before, both models will be stored in `models/joblibs/` as `gtsrb.joblib` and `gtsrb_substitute.joblib`.
Now we can try to attack our substitute model and see if the adversarial samples are effectively fooling the oracle or not.
We will use Maximal WJSMA over the first 1000 images of the test set:

```python start_attack_substitute.py --model gtsrb_substitute --oracle gtsrb --dataset gtsrb --settype test --attack tjsma_nt --firstindex 0 --lastindex 1000 --batchsize 100```

The results will be saved in `results/tjsma_nt_z_gtsrb_substitute_gtsrb_test_1.0_non_stop/`. We can now evaluate the success rate and the transferability of the attack:

```python start_stats_substitute.py --attack tjsma_nt_z_gtsrb_substitute_gtsrb_test_1.0_non_stop```


## File format

The results are stored in the `results/` folder. The sub-folders contain the crafted adversarial samples saved as CSV.
The folders are named like `<attack>_<model>_<dataset>_<set_type>_<theta>_<non_stop>`. This is the name that you use to evaluate the attacks.

The attacks are save as CSV containing the adversarial samples in the firsts columns and the original images in the last column.
The columns name gives additional information about the attacked sample
 - for targeted attacks, the columns are named `<index>_<original_class>_<target_class>`
 - for non-targeted attacks, the columns are named `<index>_<original_class>`
 - for non-targeted attacks against a substitute model, the columns are named `<index>_<original_class>_<oracle_prediction>_<substitute_prediction>`
 
The model and dataset joblibs are stored in `models/joblibs/` and `datasets/joblibs/` respectively. The models used in our article are provided under the names mnist, cifar10 and gtsrb.
