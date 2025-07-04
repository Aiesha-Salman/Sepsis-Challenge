# Phase 2 Submission

### This is the Phase 2 example submission. In this phase, we have made few improvements and updates over the previous submission. These changes enhance reproducibility, consistency, and robustness of our evaluation. 
---

## What's New in Phase 2

Overall major function arguments remains same, some new functionality have been added as follows.

- **Enhanced Data Loading I/O:**  
  - The `load_challenge_testdata` function accepts an optional list of selected columns. This ensures that during inference, only the necessary columns are loaded from the CSV.
  - New helper functions have been added to read selected variables either from the model dictionary or from a file (e.g., `selected_variables.txt`) in the model folder or root directory.

- **Threshold Handling Improvements:**  
  - A new mechanism is implemented to read the classification threshold from a file. If the file is missing or cannot be read, the system computes an approximate threshold from the prediction probabilities and binary predictions.

- **Parsimony Score Calculation:**  
  - The parsimony score is computed as the number of selected variables divided by the total number of available predictors (136 by default after excluding identifiers). This metric helps quantify model complexity.

- **Compute Calculation:**  
  - We now track and report compute resource usage—including memory usage and CPU time—using `psutil`. This data is recorded during inference and included in the evaluation metrics.

## Scoring Criteria

1. **Sensitivity Guard**  
   - Your model’s **sensitivity must be ≥ 0.8** at the chosen classification threshold.  
   - If sensitivity < 0.8, we **do not compute** a weighted score; both `weighted_score` and `scaled_weighted_score` will be **null** in the JSON output and the submission will not appear on the leaderboard.

2. **Threshold Selection**  
   - We have added an example code to calculate the threshold ensuring a sensitivity  ≥ 0.8 during training, you  may also supply your own threshold via `threshold.txt` using your run model function, ensuring a sensitivity ≥ 0. 8.   

3. **Weighted Scoring via Factor Loadings**  
   - The four performance metrics—`F1`, `AUPRC`, `Net.Benefit`, and `ECE`—are standardized (using `scale_params.json`) and then combined using the factor‐analysis loadings in `factor_loadings.json`. 
   - Parsimony (inverted: 1 − score) and inference-time (inverted z-normalized “speed”) each contribute a fixed 5 % weight.

4. **Final Z-Score Transform**  
   - The raw weighted score is transformed via a z-score using μ and σ in `zscore_params.json`. 
   - Both `weighted_score` and `scaled_weighted_score` are rounded to **4 decimal places**.

5. **Leaderboard Ranking**  
   - Submissions that satisfy sensitivity ≥ 0.8 will be ranked by their `scaled_weighted_score` (higher is better).  
   - Submissions with sensitivity < 0.8 will receive null scores and will not be ranked.  


---


## Submission Structure

```bash
submission/
├── Dockerfile                # REQUIRED: Defines the container image and entry point.
├── requirements.txt          # REQUIRED: Lists all Python dependencies.
├── run_model.py              # Not required: Script to run inference and evaluation.
├── team_code.py              # REQUIRED: Contains your training and inference functions.
├── helper_code.py            # Not required: Contains helper functions used by your code.
├── threshold.txt             # REQUIRED: Used as classification threshold.
├── selected_variables.txt     # OPTIONAL: Contains the raw selected variables used for training, if not given/calculated explicitely all features with be considered as used for parsimony.
├── dummy_columns.txt         # OPTIONAL: Contains the dummy‐encoded column names (if not stored in model folder).
├── scale_params.json # centers & scales for the four factor metrics + inference time.
├── factor_loadings.json # numeric loadings for F1, AUPRC, Net.Benefit, ECE.
├── zscore_params.json #  µ and σ for z-scoring the raw weighted score.
└── model/
    ├── model.sav             # REQUIRED: Serialized trained model (includes imputer, prediction_model, etc.).
    ├── dummy_columns.txt     # OPTIONAL: List of dummy‐encoded columns (used to align test data).
    └──selected_variables.txt  # OPTIONAL: Contains the raw selected variables used for training, if not given/calculated explicitely all features with be considered as used for parsimony.
```

## File Descriptions

- **Dockerfile**  
  Defines your container image. It installs dependencies from `requirements.txt` and sets the entry point (for example, to run `run_model.py`).

- **requirements.txt**  
  Lists all necessary Python libraries (e.g., numpy, pandas, scikit-learn, joblib, mne, psutil).

- **run_model.py**  
  Loads the trained model, runs inference on test data, computes inference time and resource usage, and writes outputs including the parsimony score.

- **team_code.py**  
  Contains your team's training and inference functions. The training code now saves both the raw selected features and the dummy‑encoded columns, while the inference code uses these files for consistency.

- **helper_code.py**  
  Contains utility functions for loading data, saving outputs, and computing resource usage.

- **threshold.txt**  
  Contains the classification probability threshold (e.g., `0.5`). This value can be computed during training or hard coded.

- **selected_variables.txt**  
  Contains a list of the raw variables selected for training. This file is used during inference to load only the necessary columns and to compute the parsimony score. If this file is absent, all features will be used.

- **dummy_columns.txt**  
  Contains the list of dummy‑encoded feature names, which are used to align test data with the training features. Alternatively, these may be stored in `columns.txt` within the model folder.

- **model/**  
  - **model.sav:** The serialized model including the imputer and prediction model.
  - **dummy_columns.txt:** Required file listing the dummy‑encoded columns used during training.
  - **selected_variables.txt:** (Optional) A copy of `selected_variables.txt` for reference.

---
---

## How to Run the Phase-2 Challenge Example Code in Docker

Follow these steps to run the example code in a Docker environment: 

1. **Prepare Your Local Directory Structure:**

   Create a directory (e.g., `~/example`) with the required subfolders:

   ```bash
   mkdir -p ~/example/{mkdir training_data test_data model test_outputs}

2. **Clone or Download the Repository:**
  
    Clone the updated repository containing the example submission submission:
    ```bash
    git clone --branch Phase2 https://github.com/Kamaleswaran-Lab/The-2024-Pediatric-Sepsis-Challenge.git
    cd The-2024-Pediatric-Sepsis-Challenge
    ```
    We have provided example train_data.csv, test_data.csv and labels.csv on the above cloned example repo.
    You may consider placing them or your own CSVs in those folders for testing and debuggig. 
    - Place train_data.csv files into ~/example/training_data folder
    - Place test_data.csv files into ~/example/test_data folder
    - Place labels.csv files into ~/example/test_data folder
3. Build the Docker Image:
    
    In the root of the repository, build the Docker image using the provided Dockerfile:
    ```bash
    docker build -t image .
    ```
3. Run the Docker Container with Volume Mappings:
   
   Start the Docker container and map your local directories into the container. For example:
    ```bash
    docker run -it \
    -v ~/example/model:/challenge/model \
    -v ~/example/test_data:/challenge/test_data \
    -v ~/example/test_outputs:/challenge/test_outputs \
    -v ~/example/training_data:/challenge/training_data \
    image bash
    ```

4. Inside the Container:
  
    Once inside the container, verify that the directory structure is correct:
    ```bash  
    root@[...]:/challenge# ls
        Dockerfile             README.md         test_outputs
        evaluate_2024.py      requirements.txt  training_data
        helper_code.py         team_code.py      train_model.py
        LICENSE                run_model.py    dummy_data_split.py

    ```

    **Train Your Model:**
    Run the training script to build and save your model:
    ```bash
    python train_model.py training_data/training_data.csv model
    ```
    **Run Your Trained Model**:
    Execute the inference script to generate predictions:
    ```bash
    python run_model.py model test_data/test_data.csv test_outputs
    ```
    **Evaluate Your Model:**
    Finally, run the evaluation script to compute performance metrics (e.g., AUC, AUPRC, Sensitivity parsimony score, compute usage):
    ```bash
    python evaluate_2024.py test_data/labels.csv test_outputs/outputs.txt test_outputs/inference_time.txt threshold.txt scale_params.json factor_loadings.json zscore_params.json score.json
    ```
   The final score.json will include weighted_score and scaled_weighted_score only when sensitivity ≥ 0.8; otherwise they will be null.
    
    **Exit the Container:**
    Once finished, simply type:
    ```bash
    exit
    ```
These steps ensure that you run the complete Phase 2 example code in a reproducible Docker environment. The code includes flexible thresholding, compute resource tracking, and parsimony score calculation.
