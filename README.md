# MNIST PGD Adversarial Attack
This project investigates the robustness of a Convolutional Neural Network (CNN) trained on the MNIST handwritten digit dataset under adversarial perturbations generated using the Projected Gradient Descent (PGD) attack.
## Objective
The goal of this project is to analyze how adversarial perturbations affect the performance of neural network classifiers and to study how increasing attack strength impacts model accuracy.
## Model
A simple CNN architecture was used for digit classification:
- Conv layer (16 filters, 5×5) + ReLU + MaxPool
- Conv layer (32 filters, 5×5) + ReLU + MaxPool
- Fully connected layer (128 neurons)
- Output layer (10 classes)
The model was trained using:
- Cross-Entropy Loss
- Stochastic Gradient Descent (SGD)
- 3 training epochs
- Batch size = 64
## Dataset
The MNIST dataset containing 70,000 handwritten digit images (28×28 grayscale).
- Training samples: 60,000
- Test samples: 10,000
Images were normalized to the range [0,1].
## Adversarial Attack
The Projected Gradient Descent (PGD) attack was implemented using the **L2 norm constraint**.  
The attack iteratively perturbs input images to maximize the classification loss while keeping the perturbation within a specified budget.
Only correctly classified test samples were attacked to ensure that misclassifications were caused by adversarial perturbations.
## Experiment
A small experiment was conducted to analyze how increasing the **L2 perturbation budget (dmax)** affects model robustness.
Attack parameters:
- step size = 0.2
- iterations = 40
- varying parameter: dmax
### Results
| dmax | Adversarial Accuracy | Accuracy Drop |
|-----|----------------------|---------------|
| 0.5 | 0.99 | 0.01 |
| 1.0 | 0.915 | 0.085 |
| 1.5 | 0.78 | 0.22 |
| 2.0 | 0.53 | 0.47 |
| 2.5 | 0.33 | 0.67 |
| 3.0 | 0.21 | 0.79 |
As the perturbation strength increased, adversarial accuracy decreased significantly, demonstrating the vulnerability of neural networks to carefully crafted adversarial inputs.
## Visualization
The following plot shows the effect of increasing the L2 perturbation budget (dmax) on adversarial accuracy.
![PGD Attack Results](pgd_attack_results.png)
## Key Insight
Even a model that performs well on clean test data can experience severe performance degradation when exposed to adversarial perturbations.
This highlights the importance of evaluating machine learning systems under adversarial conditions to understand their robustness.
## File
`task_2_pgd_6.py`  
Implementation of CNN training and PGD adversarial attack evaluation.
## Author
Keerthija Bontu  
M.Eng Information Technology(Artificial Intelligence)
