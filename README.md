

### Parameter List

**Datasets**: MNIST, Cifar-10, Fashion-MNIST

**Model**: CNN, MLP
**DP Mechanism**: Laplace, Gaussian(Simple Composition), **Todo**: Gaussian(*moments* accountant)

**DP Parameter**: $\epsilon$ and $\delta$

**DP Clip**: In DP-based FL, we usually clip the gradients in training and the clip is an important parameter to calculate the sensitivity.

### No DP

You can run like this:

python main.py --dataset cifar --iid --model cnn --epochs 50 --frac 0.1 --dp_mechanism no_dp

### Laplace Mechanism

You can run like this:

python main.py --dataset mnist --iid --model cnn --epochs 300 --frac 0.1 --dp_mechanism Laplace --dp_epsilon 5
### Gaussian Mechanism

python main.py --dataset mnist --iid --model cnn --epochs 50 --frac 0.1 --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5

  
