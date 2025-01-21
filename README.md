# AdapLDP-FL

<!-- start intro -->

This repository provides the implementation of the paper ["AdapLDP-FL: An Adaptive Local DifferentialPrivacy for Federated Learning"](https://10.1109/TMC.2024.3374789), which is published in IEEE Transactions on Mobile Computing. This paper investigates FL under the scenario of noise optimization with LDP. Specifically, given a certain privacy budget, we design the adaptive LDP method via a noise scaler, which adaptively optimizes the noise size of every client. Secondly, we dynamically tailor the model direction after adding noise by the designed a direction matrix, to overcome the model drift problem caused by adding noises to the client model. Finally, our method achieves higher accuracy than some existing works with the same privacy level and the convergence speed is significantly improved.

<table>
  <tr>
    <td width="25%"><img src="Fig/github_loss76.png" width="300"></td>
    <td width="25%"><img src="Fig/githubspeed83.png" width="300"></td>
    <td width="25%"><img src="Fig/githubacc103.png" width="300" ></td>
  </tr>
  <tr>
    <td width="25%">Loss function on three public datasets.</td>
    <td width="25%">Convergence speed improvement compared with LDP-FL</td>
    <td width="25%">Comparison of model performance via CNN networks.</td>
  </tr>
</table>


We built a privacy-preserving FL based on our proposed adaptive LDP, mainly consisting of adaptive updating noise and transform noise direction. 

<p align="center">
<img src="Fig/framework.png" align="center" width="85%"/>
</p>

<!-- end intro -->

## 1. Optimizing Noise Constraints

<!-- start Noise -->

The code in the folder [models](models/Fed.py) is for enhancing the trade-off between privacy and performance.

`main.py` is the main function.

The input is the path of the dataset.

<!-- end Clipping -->

## 2. Skipping of Clipping Operation

<!-- start Clipping -->

The code in the folder [models](models/Fed.py) is for enhancing convergence speed and  model drift problems of LDP-FL approach.

`main.py` is the main function.

The input is the path of the dataset.

<!-- end Clipping -->


## Prerequisites

To run the code, it needs some libraies:

- Python >= 3.8
- Pytorch >= 1.10
- torchvision >= 0.11
- phe >= 1.5
- skfuzzy >= 0.4

Our environment is shown in the file, named `environment.yaml`.

## Parameter List

**Datasets**: MNIST, Cifar-10, Fashion-MNIST

**Model**: CNN, MLP

You can run like main.py this:

python main.py --dataset mnist --iid --model cnn --epochs 50 --frac 0.1 --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5

  


## Citing

<!-- start citation -->

If you use this repository, please cite:
```bibtex
@article{yue2025AdapLDP-FL,
  title={AdapLDP-FL: An Adaptive Local DifferentialPrivacy for Federated Learning},
  author={Han, Junhao and Yan, Li},
  journal={IEEE Internet of Things Journal},
  volume={Early Access},
  year={2023},
  publisher={IEEE}
}
```

<!-- end citation -->



