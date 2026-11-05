# CART Decision Tree: Student Exam Result Prediction

A beginner-friendly Jupyter notebook that builds a **CART (Classification and Regression Tree)** model to predict whether a student will **Pass** or **Fail** an exam based on their study habits.

The dataset is very small (18 students), so this project is meant for **learning how CART works**, not for real-world prediction.

## Notebook

`Mulla_Javeed_05_11_CART.ipynb`

## What the notebook covers

1. Import the required packages
2. Create the student exam dataset and inspect it
3. Visualize the Pass/Fail distribution
4. Find the root split by hand using **Gini impurity**
5. Split the data into training and test sets and one-hot encode the features
6. Train a CART classifier (`criterion="gini"`, `max_depth=3`)
7. Evaluate the model (accuracy, classification report, confusion matrix)
8. Draw the trained decision tree
9. Predict results for new students
10. Summary of key takeaways

## Dataset

The data is created directly inside the notebook (no external file needed). Each row is one student.

| Feature | Values |
|---|---|
| `StudyHours` | Low, Medium, High |
| `Attendance` | Poor, Average, Good |
| `Sleep` | Short, Adequate |
| `Assignments` | Incomplete, Complete |
| `Result` (target) | Pass, Fail |

The two classes are evenly balanced (9 Pass, 9 Fail).

## Method in brief

- **Gini impurity** measures how mixed a node is:

  `G = 1 - Σ (p_c)²`

  where `p_c` is the fraction of class `c` in that node.
- For every possible rule (e.g. `StudyHours is High`), the notebook computes the weighted Gini of the true and false branches and the **Gini gain**. The rule with the lowest weighted Gini becomes the root split.
- Categorical features are converted to numbers with `OneHotEncoder`, fitted on the **training data only** so the test set stays unseen.
- The train/test split is 80/20, stratified by `Result`, with `random_state=21`.
- The tree depth is limited to 3 to keep it readable and reduce overfitting.

## Requirements

- Python 3.8+
- pandas
- matplotlib
- scikit-learn (1.2 or newer, because `OneHotEncoder(sparse_output=False)` is used)
- Jupyter Notebook or JupyterLab

Install the dependencies:

```bash
pip install pandas matplotlib scikit-learn notebook
```

## How to run

1. Clone or download this repository.
2. Open a terminal in the project folder and start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `Mulla_Javeed_05_11_CART.ipynb`.
4. Run all cells from top to bottom (**Kernel → Restart & Run All**).

## Outputs

Running the notebook produces:

- A bar chart of the Pass/Fail distribution
- A table ranking every candidate split by weighted Gini and Gini gain
- Test accuracy and a classification report (precision, recall, F1-score)
- A confusion matrix
- A plot of the trained decision tree
- Predictions with certainty scores for three new example students

## Limitations

- With only 18 records, results change a lot depending on the train/test split. Treat the scores as a demonstration only.
- Real exam prediction would need a much larger and more varied dataset.

## Key takeaways

- CART learns a hierarchy of if/else rules directly from labeled examples.
- Gini impurity drives the choice of every split.
- One-hot encoding lets the tree work with categorical features.
- A shallow tree is easier to interpret and less likely to overfit.

## Author

Mulla Javeed
