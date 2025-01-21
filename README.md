# Differential Privacy (DP) Based Federated Learning (FL) 
Everything about DP-based FL you need is here.

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

## Papers

- Reviews
  - Rodríguez-Barroso, Nuria, et al. "Federated Learning and Differential Privacy: Software tools analysis, the Sherpa. ai FL framework and methodological guidelines for preserving data privacy." *Information Fusion* 64 (2020): 270-292.
- Gaussian Mechanism
  - Wei, Kang, et al. "Federated learning with differential privacy: Algorithms and performance analysis." *IEEE Transactions on Information Forensics and Security* 15 (2020): 3454-3469.
  - Geyer, Robin C., Tassilo Klein, and Moin Nabi. "Differentially private federated learning: A client level perspective." *arXiv preprint arXiv:1712.07557* (2017).
  - Seif, Mohamed, Ravi Tandon, and Ming Li. "Wireless federated learning with local differential privacy." *2020 IEEE International Symposium on Information Theory (ISIT)*. IEEE, 2020.
  - Naseri, Mohammad, Jamie Hayes, and Emiliano De Cristofaro. "Toward robustness and privacy in federated learning: Experimenting with local and central differential privacy." *arXiv e-prints* (2020): arXiv-2009.
  - Truex, Stacey, et al. "A hybrid approach to privacy-preserving federated learning." *Proceedings of the 12th ACM workshop on artificial intelligence and security*. 2019.
  - Triastcyn, Aleksei, and Boi Faltings. "Federated learning with bayesian differential privacy." *2019 IEEE International Conference on Big Data (Big Data)*. IEEE, 2019.
- Laplace Mechanism
  - Wu, Nan, et al. "The value of collaboration in convex machine learning with differential privacy." *2020 IEEE Symposium on Security and Privacy (SP)*. IEEE, 2020.
  - Olowononi, Felix O., Danda B. Rawat, and Chunmei Liu. "Federated learning with differential privacy for resilient vehicular cyber physical systems." *2021 IEEE 18th Annual Consumer Communications & Networking Conference (CCNC)*. IEEE, 2021.
- Other Mechanism
  - Sun, Lichao, Jianwei Qian, and Xun Chen. "Ldp-fl: Practical private aggregation in federated learning with local differential privacy." *arXiv preprint arXiv:2007.15789* (2020).
  - Liu, Ruixuan, et al. "Fedsel: Federated sgd under local differential privacy with top-k dimension selection." *International Conference on Database Systems for Advanced Applications*. Springer, Cham, 2020.
  - Truex, Stacey, et al. "LDP-Fed: Federated learning with local differential privacy." *Proceedings of the Third ACM International Workshop on Edge Systems, Analytics and Networking*. 2020.
  - Zhao, Yang, et al. "Local differential privacy-based federated learning for internet of things." *IEEE Internet of Things Journal* 8.11 (2020): 8836-8853.

  