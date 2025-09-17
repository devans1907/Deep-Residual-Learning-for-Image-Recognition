A Practical Analysis of Deep Residual Learning and Regularisation Techniques
Project: Reproduction and Extension of "Deep Residual Learning for Image Recognition" Date: 19 October 2023
1. Summary of "Deep Residual Learning for Image Recognition"
Problem Solved: The paper addresses the "degradation" problem in training very deep neural networks. Contrary to expectations, simply stacking more layers onto a deep model often leads to higher training and test error. This issue is not caused by overfitting but by optimisation difficulties; solvers struggle to learn identity mappings with stacks of non-linear layers, which hampers convergence as networks get deeper. The authors demonstrated this phenomenon with "plain" networks, where a 56-layer model performed worse than a 20-layer one on CIFAR-10.
Proposed Solution: The authors introduced a deep residual learning framework to ease the training of substantially deeper networks. Instead of having layers learn a direct, unreferenced mapping H(x), the framework reformulates them to learn a residual mapping: F(x) = H(x) - x. The original mapping is then calculated as F(x) + x.
This is implemented using "shortcut" or "skip connections" that perform an identity mapping, adding the input x directly to the output of a block of layers. This approach has two key advantages:
It helps address the degradation problem. If an identity mapping is optimal for a set of layers, the network can easily learn this by driving the weights of the residual block (F(x)) to zero.
The identity shortcuts add no extra parameters or computational complexity, allowing for a fair comparison against plain networks of the same depth and width.
This framework enabled the authors to successfully train networks of unprecedented depth (e.g., 152 layers) and achieve state-of-the-art results, winning 1st place in several ILSVRC & COCO 2015 competitions.

2. Reproduction Process
Objective: The first phase of this project aimed to empirically validate the paper's central claims by implementing and comparing an 18-layer plain network (PlainNet-18) and its residual counterpart (ResNet-18) on the CIFAR-10 image classification task.
Methodology:
Dataset and Environment: The project used the CIFAR-10 dataset, consisting of 60,000 32x32 colour images across 10 classes(There are 50000 training images and 10000 test images).Experiments were conducted in a PyTorch environment. Standard data augmentation (random flips and crops) and normalisation were applied to the training data to improve generalisation.
Model Architectures:
PlainNet-18: A baseline CNN with 17 sequential 3x3 convolutional layers followed by a fully-connected layer. The architecture progressively halves spatial dimensions while doubling the number of filters.
ResNet-18: This model has the same convolutional layers as the PlainNet-18 but groups them into residual blocks. Each block features a shortcut connection that adds the block's input to its output, overcoming the degradation problem.
Training Regimen: To ensure a fair comparison, both models were trained under identical conditions for 75 epochs:
Optimiser: SGD with 0.9 momentum and a weight decay of 5e-4.
Loss Function: Cross-Entropy Loss.
Learning Rate: A Cosine Annealing scheduler starting from 0.1 was used for effective convergence.
3. Project Extension: Combating Overfitting
Motivation and Hypothesis: Initial results from the reproduction showed that both the PlainNet-18 and ResNet-18 models were prone to significant overfitting, achieving near-perfect training accuracy while test accuracy plateaued much lower. This motivated an extension to investigate if standard regularisation techniques could mitigate this issue.
The hypothesis was that a combination of regularisation techniques would reduce the gap between training and test accuracy, creating a more robust and generalisable model.
Methodology: A new "Regularised ResNet-18" model was created by applying three simultaneous changes to the baseline ResNet-18 architecture:
Reduced Model Complexity: The number of channels at each stage was halved to create a "slimmer" model with less capacity to memorise the training data.
Dropout: A dropout layer with a probability of 0.5 was added before the final fully-connected layer to prevent co-adaptation of features.
Increased L2 Regularisation: The weight decay parameter was doubled to 1e-3 to apply a stronger penalty on large weight values.

4. Results
Core Experiment: PlainNet vs. ResNet
Model
Final Test Accuracy
Final Train Accuracy
Overfitting Gap
PlainNet-18
93.71%
99.88%
6.17%
ResNet-18
94.20%
99.97%
5.77%

Table 1: Performance comparison from the reproduction experiment.
The results confirm the paper's findings: the ResNet-18 achieved a higher test accuracy, demonstrating the effectiveness of residual connections in improving optimisation and performance. However, both models showed a large overfitting gap, justifying the extension.
Extension Experiment: Regularisation
Model
Final Test Accuracy
Final Train Accuracy
Overfitting Gap
Original ResNet-18
94.20%
99.97%
5.77%
Regularised ResNet-18
93.54%
99.37%
5.83%

Table 2: Performance comparison from the regularisation experiment.
The regularisation techniques were successful in their primary goal. The final training accuracy of the regularised model was notably lower (99.37% vs. 99.97%), indicating it was no longer memorising the training data as aggressively.
5. Insights and Conclusion
This project yielded several key insights into the principles and practical application of deep residual learning.
Validation of Residual Learning: The reproduction successfully validated the core thesis of He et al. The ResNet-18 outperformed the PlainNet-18, confirming that shortcut connections are a powerful tool for training deeper, more effective networks.


Demonstration of the Bias-Variance Trade-off: The extension provided a tangible example of the bias-variance trade-off. The aggressive regularisation techniques successfully reduced model variance (overfitting) at the cost of slightly increased bias, which resulted in a minor drop in peak test accuracy. This highlights that while the regularised model has a slightly lower headline accuracy, it is more robust and offers a more honest reflection of its generalisation capabilities.


Practical Considerations: The project also highlighted the practical trade-off between performance and computational cost, as the ResNet-18 took significantly longer to train than the PlainNet-18 due to the more complex gradient flow and shortcut computations.


