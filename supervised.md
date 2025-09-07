# Supervised Learning

Supervised learning is a type of machine learning where a model is trained using **labeled data**, meaning each input has a corresponding correct output. The goal is for the model to learn the mapping between inputs and outputs so it can make accurate predictions on new, unseen data.  

### Key Idea
In supervised learning, the algorithm is provided with pairs of inputs and outputs. The model identifies the relationship between them and generalizes it to predict outcomes for future inputs.

- **Input (X):** Features or independent variables  
- **Output (Y):** Target or dependent variable  
- **Model:** Learns the function \( f(X) \approx Y \)  

---

# Regression in Supervised Learning
Regression is used when the **output variable is continuous** and real-valued. The objective is to predict numerical outcomes based on input features.  

**Examples:**  
- Predicting house prices  
- Estimating rainfall amounts  
- Forecasting stock prices  

---

### Types of Regression Algorithms

- **Linear Regression**  
  Models the relationship between input \(x\) and output \(y\) as a straight line:  
  \[ y = mX + b \]  
  where:  
  - \(m\) = slope of the line (weight)  
  - \(b\) = intercept  

- **Logistic Regression**  
  Despite its name, it is mainly used for classification tasks. It predicts the probability of an outcome that can only take discrete values (e.g., 0 or 1).  

- **Support Vector Regression (SVR)**  
  Based on Support Vector Machines. It tries to fit the best regression line within a margin of tolerance around the data points.  

- **Decision Tree Regression**  
  Splits data into regions based on decision thresholds and predicts outcomes by averaging values within each region.  

- **Random Forest Regression**  
  An ensemble of decision trees combined together. It improves accuracy and reduces overfitting compared to a single decision tree.  

# Loss Function vs Cost Function

In machine learning, **loss function** and **cost function** are closely related concepts but have subtle differences:

- **Loss Function:**  
  Measures the error for a **single training example**. It quantifies how far the model's prediction is from the actual output for one data point.

- **Cost Function:**  
  Represents the **average (or aggregate) error** over all training examples in the dataset. It is typically calculated by averaging the loss values of all individual samples. The cost function is what the model tries to minimize during training.

---

### Summary

| Term          | Definition                                               | Scope                    |
|---------------|----------------------------------------------------------|--------------------------|
| Loss Function | Error for one data point                                  | Single training example  |
| Cost Function | Average (or total) loss over entire dataset              | Whole dataset            |

---

### Why This Matters

- During training, the model parameters are optimized to minimize the **cost function**, which indirectly reduces the individual **losses**.
- Loss functions provide detailed feedback per example, while cost functions summarize overall model performance.
- Both terms are often used interchangeably, but the distinction is important in understanding model optimization.

---

### Example (Mean Squared Error)

- Loss for one example:  
  \[
  L = (y_{\text{pred}} - y_{\text{actual}})^2
  \]

- Cost (average loss) over \(m\) examples:  
  \[
  J(\theta) = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2
  \]

Where \(h_\theta(x^{(i)})\) is the prediction from the model, and \(y^{(i)}\) is the corresponding true output.


