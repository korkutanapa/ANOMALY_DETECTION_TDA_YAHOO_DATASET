This notebook implements a sophisticated, unsupervised pipeline for detecting anomalies in time series data (specifically the Yahoo A3 Benchmark) using **Topological Data Analysis (TDA)**.

The core philosophy here is that anomalies in time series often manifest as **changes in the "shape" or connectivity** of the data when embedded into a higher-dimensional space, rather than just changes in amplitude.

Here is a detailed breakdown of the workflow implemented in your code:

---

### 1. Data Pre-processing: Segmentation (Change Point Detection)
**Cell:** `# @title segmentation`

Before applying TDA, the code addresses the issue of **non-stationarity** (where the statistical properties of the signal change over time). TDA works best on "stable" regimes.

* **Logic:** It iterates through files `A3Benchmark-TS1.csv` to `TS100.csv`.
* **Feature:** It calculates `rolling_variance`. Variance is often a better indicator of structural change than the raw mean.
* **Algorithm:** It uses the **PELT** algorithm (Pruned Exact Linear Time) from the `ruptures` library.
* **Outcome:** It detects structural change points (e.g., a sudden jump in noise level or frequency) and saves them in a new column `new_cp`.
* **Why this matters:** TDA features will be normalized *per segment*. This prevents a high-variance segment from drowning out anomalies in a low-variance segment of the same time series.

### 2. TDA Feature Extraction (The "Shape" of Data)
**Cell:** `# @title TDA Feature Extractor (RAW values only...)`

This is the mathematical engine of the pipeline. It converts a 1D time series into topological features.

* **Takens' Embedding:**
    The code converts a sliding window of the time series into a "point cloud" in $d$-dimensional space.
              
$$
            \vec{v}_t = [x_t, x_{t-\tau}, x_{t-2\tau}, \dots, x_{t-(d-1)\tau}]
$$
  
    * **Parameters:** Window ($W=14$), Dimension ($d=7$), Time Delay ($\tau=1$).
    * **Result:** A small cloud of points representing the dynamics of that window.

* **Persistent Homology ($H_0$):**
    The code uses `ripser` to calculate **0-dimensional persistence ($H_0$)**.
    * $H_0$ measures **connected components**. As the filtration parameter (radius) grows, points connect to form clusters.
    * **Lifetimes:** The "persistence" of a component is how long it stays separate before merging with another. Long lifetimes = significant gaps/jumps in data; Short lifetimes = noise.

* **Topological Summaries:**
    Instead of using the raw diagrams, the code calculates scalar features describing the distribution of these lifetimes. Key features include:
    * **`H0_bottleneck`:** The maximum lifetime (amplitude of the biggest jump).
    * **`H0_ratio_auc_L1_to_sum`:** Measures the "area under the curve" of the persistence landscape relative to the total sum of lifetimes. This is a shape descriptor.
    * **`H0_gini`:** Measures the inequality of lifetimes. A high Gini coefficient means one "gap" dominates (likely an anomaly); low Gini means uniform noise.
    * **`H0_energy_concentration`:** The $L^2$ norm divided by the sum.

### 3. VAAD Scoring (Velocity & Acceleration)
**Cell:** `# @title TDA Feature Extractor and Anomaly Scoring Algorithm (VAAD)`

This section implements a meta-algorithm. Raw TDA values might drift slowly. Anomalies are often characterized by **how fast** the topology changes.

* **Concept:** It treats the TDA features stream as a physical object moving through space.
* **Derivatives:**
    * $V$ (Velocity): First derivative of the TDA feature (rate of change).
    * $A$ (Acceleration): Second derivative (rate of change of the rate).
* **The Score Formula:**
    $$Score = (k_v \times V_{norm}) \times (k_a \times A_{norm})$$
    where $V_{norm}$ and $A_{norm}$ are robustly normalized using the Median Absolute Deviation (MAD).
* **Interpretation:** A high score means the topological shape of the data changed suddenly and violently—a strong indicator of a point anomaly.

### 4. Thresholding and Evaluation (Making the Decision)
The notebook implements four different strategies to convert the continuous `anomalyscore` into binary Yes/No predictions.

#### A. Top-K Analysis
**Cell:** `# @title by ground truth K value`
* **Logic:** "Cheating" slightly for analysis. If the ground truth says there are 5 anomalies, it picks the top 5 highest scores.
* **Purpose:** Establishes the **theoretical maximum performance** of your features. If Top-K performs poorly, the feature engineering is bad.

#### B. Quantile Thresholding
**Cell:** `# @title thresholding by quantile`
* **Logic:** Calculates the 99th percentile ($0.99$) of scores within each segment. Anything above this is an anomaly.
* **Evaluation:** Uses a "dilated" matching window (`NTOL=3`). If the prediction is within 3 steps of the real anomaly, it counts as a True Positive (TP).

#### C. POT (Peaks Over Threshold)
**Cell:** `# @title POT`
* **Logic:** Uses signal processing (`scipy.signal.find_peaks`).
* **Adaptive Threshold:** $Threshold = \mu_{prominence} + K \cdot \sigma_{prominence}$.
* **Purpose:** Good for finding spikes in the score that stand out locally, even if the absolute score isn't globally high.

#### D. Extreme Value Theory (EVT)
**Cell:** `# @title EVT a3` / `# @title finding the best parameters`
* **Logic:** This is the most advanced statistical method.
* **Method:** It fits a **Generalized Pareto Distribution (GPD)** to the "tail" of the anomaly scores (values above a "Gate" quantile, e.g., 80%).
* **Probability:** It calculates a threshold such that the probability of a score exceeding it is extremely low (e.g., $1 - 0.993$).
* **Grid Search:** The final cell brute-forces combinations of Gate Quantiles and Final Quantiles to find the optimal sensitivity.

### Summary of Results (Based on your output)

Looking at the output of the **EVT Grid Search**, your best performing features are:

1.  **`anomalyscore_h0_auc_over_l2`**: F1 Score ~ **0.89**
2.  **`anomalyscore_bottleneck`**: F1 Score ~ **0.88**
3.  **`anomalyscore_h0_auc`**: F1 Score ~ **0.87**

This indicates that **Area Under the Curve (AUC)** based topological features and the **Bottleneck distance** (Max lifetime) are highly effective for the Yahoo A3 dataset.

### Technical Recommendations

1.  **Feature Selection:** Since `h0_auc_over_l2` is performing best (~0.89 F1), you might want to simplify your production pipeline to calculate only that feature to save compute time.
2.  **Visualization:** Use the interactive plotting cell provided. Select `ts11` or `ts78` (which show detected changepoints in the logs) to visually inspect how the `h0_auc_over_l2` score spikes exactly where the anomaly occurs.
3.  **Tolerance (NTOL):** Your code uses `NTOL=3`. Ensure this aligns with your business use case. If you need exact timestamp precision, set `NTOL=0` (though F1 scores will drop significantly).

**Next Step:** Would you like me to extract the specific mathematical formula used for the `_auc_tri_max` (Triangular Landscape AUC) function to include in your report?

