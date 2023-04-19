# Neural_Network_Charity_Analysis

## Overview of the analysis

The purpose of this analysis is to create and evaluate a deep learning model to predict the success of AlphabetSoup's funding applications. The model is designed to assist the company in determining which applications to approve or reject, maximizing their impact on the projects they support.

## Results

### Data Preprocessing

- Target variable: 'IS_SUCCESSFUL' (binary variable indicating if the funding application was successful or not)
- Features: All remaining variables excluding 'EIN' and 'NAME' (e.g., 'APPLICATION_TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS', 'ASK_AMT')
- Variables to be removed: 'EIN' (Employer Identification Number) and 'NAME' (Name of the organization), as they do not contribute relevant information for predicting the success of funding applications.

## Compiling, Training, and Evaluating the Model

- For the neural network model, we selected:
  - 2 hidden layers with 80 and 30 neurons, respectively, chosen based on the input features' complexity and the need to capture non-linear relationships.
  - Activation functions: ReLU for the hidden layers to mitigate vanishing gradient issues and improve training speed, and sigmoid for the output layer to generate binary predictions.
- The model achieved an accuracy of 72%, which is slightly below the target performance of 75%.

- To increase model performance, I took the following steps:

  - Tuned hyperparameters, such as the number of hidden layers, neurons, and activation functions.
  - Implemented dropout layers to reduce overfitting.
  - Adjusted the batch size and number of training epochs.
  - Applied feature scaling and encoding to preprocess the input data.

### Optimization Attempt 1:

#### In the first attempt to optimize the neural network model, I focused on the following changes:

1. Adjusting the number of hidden layers and neurons:

  - I experimented with adding more hidden layers and adjusting the number of neurons in each layer to capture more complex patterns in the data. The goal was to find a suitable architecture that can provide better predictive accuracy without overfitting.
2. Changing the activation functions:
  - I explored alternative activation functions for the hidden layers, such as the hyperbolic tangent (tanh) and leaky ReLU. These were tested to check if they could help improve the training process and overall model performance.
3. Increasing the number of training epochs:
  - By increasing the number of training epochs, the model has more opportunities to update its weights and biases, potentially improving its learning and accuracy. However, we needed to be cautious about overfitting, as training for too many epochs can cause the model to become too specialized to the training data.
4. Modifying the batch size:
  - I experimented with different batch sizes to find the optimal balance between training speed and model performance. Smaller batch sizes can lead to more frequent weight updates and improved generalization, while larger batch sizes may speed up the training process.
  - These changes aimed to find an optimal model architecture and training process that could potentially improve the prediction accuracy and meet the target performance.  Unfortunatley, I am still short only achieving 72.77%

### Optimization Attempt 2:

#### In the second attempt to optimize the neural network model, we focused on the following changes:

1. Regularization techniques:
  - I introduced regularization methods such as L1 or L2 regularization to reduce overfitting and improve generalization. Regularization helps to penalize large weights and prevents the model from becoming too complex, which can lead to better performance on unseen data.
2. Dropout layers:
  - I added dropout layers to the neural network to reduce overfitting further. Dropout is a technique that randomly deactivates a fraction of neurons during training, promoting independence between neurons and preventing co-adaptation. This can result in a more robust model that generalizes better.
3. Learning rate optimization:
  - I experimented with different learning rates for the optimizer to find an optimal value. The learning rate determines the step size during weight updates and has a significant impact on the convergence and overall performance of the model. We tried different learning rates to find the one that allows the model to learn effectively without overshooting the optimal weights.
4. Optimizer selection:
  - I explored alternative optimizers such as RMSprop and Adam, which adapt the learning rate during training, potentially leading to faster convergence and improved performance. These optimizers can be more effective than traditional optimizers like stochastic gradient descent (SGD) in certain situations.

- These changes aimed to further improve the model's ability to generalize to unseen data by reducing overfitting, optimizing the learning process, and experimenting with different optimizer algorithms. The goal was to achieve a better balance between model complexity and performance, ultimately leading to higher prediction accuracy.  Unfortunatley, I am still short of the target achieving only 72.96%

### Optimization Attempt 3:

#### In the third attempt to optimize the neural network model, we focused on the following changes:

1. Batch normalization:
  - I added batch normalization layers to the model, which normalize the activations of the previous layer at each batch. This technique helps to maintain a stable distribution of activations throughout the network, improving training speed and potentially increasing overall performance.
2. Alternative activation functions:
  - I explored using alternative activation functions such as Leaky ReLU, Parametric ReLU, or ELU to potentially improve the model's performance. These functions address the vanishing gradient problem often associated with deep neural networks and can lead to better learning outcomes.
3. Increase model capacity:
  - I experimented with increasing the number of neurons and layers in the model to provide it with more capacity to learn complex patterns in the data. However, we took care to avoid overfitting by using techniques such as regularization and dropout, as previously mentioned.
4. Early stopping:
  - I implemented early stopping, which halts training when the validation loss stops improving for a certain number of consecutive epochs. This technique helps to prevent overfitting and ensures that the model does not continue training beyond the point where it starts to degrade in performance.
5. Hyperparameter tuning:
  - I performed a more extensive hyperparameter search to find the optimal combination of learning rate, batch size, number of layers, and neurons. This was done using grid search to systematically explore the hyperparameter space and identify the best configuration for our specific problem.  When the best parameters were identified, I coded the results into this attempt.


- These changes aimed to enhance the model's learning capability, improve convergence, and increase overall performance. The goal was to fine-tune the model architecture and hyperparameters to achieve the highest prediction accuracy while maintaining a balance between model complexity and generalization. Unfortunatley, I am still short of the target achieving only 72.68%, worse than attempt 2.  I thought identifying the best parameters using grid search would hit the jackpot. 



  
## Summary

The deep learning model achieved an accuracy of 72% in predicting the success of AlphabetSoup's funding applications. Although it falls slightly short of the target performance, the model can still provide valuable insights for the organization.

For an alternative approach, we recommend trying a Gradient Boosting Classifier. This model is known for its high performance in handling complex datasets, and its ability to automatically learn feature interactions can potentially improve the prediction accuracy. Additionally, gradient boosting models are less prone to overfitting, which might contribute to better performance on unseen data.

