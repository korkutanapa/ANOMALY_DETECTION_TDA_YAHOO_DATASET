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
    * **Parameters:** Window ($W=14$), Dimension ($d=7$), Time Delay ($\tau=1$).
    * **Result:** A small cloud of points representing the dynamics of that window.
              
$$
            \vec{v}_t = [x_t, x_{t-\tau}, x_{t-2\tau}, \dots, x_{t-(d-1)\tau}]
$$
  


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


Based on the code provided in the A4 notebook, we can analyze the configuration and expected results for the Yahoo A4 dataset (Real Web Traffic), specifically focusing on how the **Extreme Value Theory (EVT)** parameters were tuned compared to the A3 (Synthetic) benchmark.

Although the explicit output tables for A4 were not visible in the file snippet, the code configuration reveals significant analytical insights into the nature of the A4 dataset.

### 1. Analysis of EVT Parameter Tuning (A3 vs. A4)

The most telling difference lies in how the **VAAD (Velocity Acceleration Anomaly Detection)** algorithm was tuned for the A4 dataset compared to the previous A3 analysis.

| Parameter | A3 (Synthetic) | A4 (Real Traffic) | Analysis |
| :--- | :--- | :--- | :--- |
| **Gate Quantile** | `0.87` (87th %) | `0.78` (78th %) | **Significant Drop.** A4 required a much lower "gate" to collect enough tail samples. This indicates the A4 dataset is **noisier**. Real web traffic has high variance in "normal" behavior, so the algorithm needs a broader sample of "high" values to accurately model the statistical tail (GPD fit). |
| **Final Quantile** | `0.995` | `0.993` | **More Aggressive.** The threshold for flagging an anomaly was lowered slightly (99.5% $\to$ 99.3%). This suggests anomalies in A4 are **less distinct** from normal data than in A3. Synthetic anomalies are often obvious spikes; real anomalies (like server timeouts or traffic dips) can be subtler, requiring a more sensitive trigger. |

### 2. Topological Feature Performance (Expected Behavior)

The A4 dataset consists of real web traffic, which typically includes **seasonality** (daily/weekly cycles) and **point anomalies** (spikes/drops). Based on the TDA features extracted, we can infer which "shapes" define the anomalies:

* **Bottleneck Distance (`anomalyscore_bottleneck`):**
    * **Function:** Measures the single largest "gap" or "loop" in the data topology.
    * **A4 Analysis:** This is likely the **top performer**. In web traffic, a sudden spike (DDoS or viral event) or drop (server crash) creates a massive, short-lived topological loop. The Bottleneck distance captures exactly this "max amplitude" deviation, ignoring smaller noise.

* **AUC over L2 (`anomalyscore_h0_auc_over_l2`):**
    * **Function:** Measures the "total area" of topological activity relative to the signal energy.
    * **A4 Analysis:** This feature is robust against baseline shifts. Since web traffic fluctuates day-to-night, the raw energy ($L2$) changes. By normalizing the topological signal (AUC) by the energy ($L2$), this feature detects anomalies regardless of whether they happen during peak traffic (day) or low traffic (night).

### 3. Change Point Detection (Segmentation)

The code re-runs **PELT** segmentation on A4.
* **A4 Context:** Unlike A3, A4 data often shifts behavior drastically (e.g., a website goes from "growth phase" to "stable phase").
* **Impact:** The TDA scoring resets at each segment. This is crucial for A4. Without segmentation, a "busy week" would look like one giant anomaly compared to a "quiet week." The segmentation ensures the model adapts to the **current** traffic baseline.

### 4. Summary of Results Structure

The notebook implements a **Grid Search** for A4 (Cell 17) to optimize the EVT thresholds.
* **Search Space:** It tested Gate Quantiles from `0.60` to `0.90` and Final Quantiles from `0.980` to `0.997`.
* **Selection Criterion:** The model selects the parameters maximizing the **F1 Score**.
* **Result Interpretation:** If the best F1 score comes from a **low Gate Quantile (e.g., 0.75-0.80)**, it confirms that A4 is "heavy-tailed"—meaning "extreme" events are relatively common, and distinguishing true anomalies requires modeling that heavy tail carefully.

**Next Step:**
To provide the precise F1 scores for A4, please copy and paste the **"FINAL PERFORMANCE SUMMARY"** table output from the notebook. I can then compare the exact accuracy of the TDA method on real data (A4) versus the synthetic benchmark (A3).

```python?code_reference&code_event_index=2
import json

# Load the notebook content
file_path = 'TDA_codes_for_YAHOO_A4.ipynb'
with open(file_path, 'r') as f:
    notebook_content = json.load(f)

# Iterate through cells to find outputs containing "F1" or "Performance"
found_scores = []
for cell in notebook_content['cells']:
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'text' in output:
                text_content = "".join(output['text'])
                if "F1" in text_content or "Performance" in text_content or "SUMMARY" in text_content:
                    found_scores.append(text_content)

# Print the found content to analyze
for i, score_text in enumerate(found_scores):
    print(f"--- Output {i+1} ---")
    print(score_text[:2000]) # Print first 2000 chars to avoid too much output
    print("\n")

```

```text?code_stdout&code_event_index=2
--- Output 1 ---
Starting EXTREME VALUE THEORY (EVT) adaptive detection...
Using data file: /content/anomaly_scores_a3.csv
Evaluation NTOL: 3
Rule: Gate at 0.87, calculate threshold for 0.995

Found 13 score columns to analyze.

Pre-calculating dilated ground truth...
Dilated ground truth calculated.

------------------------- Analyzing: anomalyscore_h0_auc -------------------------
Calculating adaptive EVT thresholds...
Prediction generation complete.
Applying NTOL to predictions...
--- Results for anomalyscore_h0_auc (EVT) ---
Total Actual Anomaly Points: 934
Total Predicted Anomaly Points: 893

Custom Confusion Matrix (Point-Adjusted):
                 Predicted Negative | Predicted Positive
Actual Negative:            160493 |                74
Actual Positive:               157 |               777

Performance Metrics (Point-Adjusted):
  F1 Score:  0.870588
  Precision: 0.913043
  Recall:    0.831906

------------------------- Analyzing: anomalyscore_h0_auc_over_max -------------------------
Calculating adaptive EVT thresholds...
Prediction generation complete.
Applying NTOL to predictions...
--- Results for anomalyscore_h0_auc_over_max (EVT) ---
Total Actual Anomaly Points: 934
Total Predicted Anomaly Points: 869

Custom Confusion Matrix (Point-Adjusted):
                 Predicted Negative | Predicted Positive
Actual Negative:            160655 |                47
Actual Positive:               167 |               767

Performance Metrics (Point-Adjusted):
  F1 Score:  0.877574
  Precision: 0.942260
  Recall:    0.821199

------------------------- Analyzing: anomalyscore_h0_auc_over_l2 -------------------------
Calculating adaptive EVT thresholds...
Prediction generation complete.
Applying NTOL to predictions...
--- Results for anomalyscore_h0_auc_over_l2 (EVT) ---
Total Actual Anomaly Points: 934
Total Predicted Anomaly Points: 895

Custom Confusion Matrix (Point-Adjusted):
                 Predicted Negative | Predicted Positive
Actual Negative:            160462 |         


--- Output 2 ---
Starting EXTREME VALUE THEORY (EVT) adaptive detection...
Using data file: /content/anomaly_scores_a4.csv
Evaluation NTOL: 3
Rule: Gate at 0.78, calculate threshold for 0.993

Found 13 score columns to analyze.

Pre-calculating dilated ground truth...
Dilated ground truth calculated.

------------------------- Analyzing: anomalyscore_h0_auc -------------------------
Calculating adaptive EVT thresholds...
Prediction generation complete.
Applying NTOL to predictions...
--- Results for anomalyscore_h0_auc (EVT) ---
Total Actual Anomaly Points: 829
Total Predicted Anomaly Points: 790

Custom Confusion Matrix (Point-Adjusted):
                 Predicted Negative | Predicted Positive
Actual Negative:            159711 |               128
Actual Positive:               186 |               643

Performance Metrics (Point-Adjusted):
  F1 Score:  0.803750
  Precision: 0.833982
  Recall:    0.775633

------------------------- Analyzing: anomalyscore_h0_auc_over_max -------------------------
Calculating adaptive EVT thresholds...
Prediction generation complete.
Applying NTOL to predictions...
--- Results for anomalyscore_h0_auc_over_max (EVT) ---
Total Actual Anomaly Points: 829
Total Predicted Anomaly Points: 545

Custom Confusion Matrix (Point-Adjusted):
                 Predicted Negative | Predicted Positive
Actual Negative:            161162 |                50
Actual Positive:               339 |               490

Performance Metrics (Point-Adjusted):
  F1 Score:  0.715851
  Precision: 0.907407
  Recall:    0.591074

------------------------- Analyzing: anomalyscore_h0_auc_over_l2 -------------------------
Calculating adaptive EVT thresholds...
Prediction generation complete.
Applying NTOL to predictions...
--- Results for anomalyscore_h0_auc_over_l2 (EVT) ---
Total Actual Anomaly Points: 829
Total Predicted Anomaly Points: 787

Custom Confusion Matrix (Point-Adjusted):
                 Predicted Negative | Predicted Positive
Actual Negative:            159716 |         



```

```python?code_reference&code_event_index=6
# Iterate through cells again to find the specific A4 summary
a4_summary = ""
for cell in notebook_content['cells']:
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'text' in output:
                text_content = "".join(output['text'])
                if "Using data file: /content/anomaly_scores_a4.csv" in text_content:
                    # found the correct cell execution for A4
                    # now find the summary table
                    if "FINAL EVT" in text_content and "PERFORMANCE SUMMARY" in text_content:
                         a4_summary = text_content

# Print the extracted summary part
if a4_summary:
    # Find the start of the summary table
    start_index = a4_summary.find("FINAL EVT")
    print(a4_summary[start_index:])
else:
    print("A4 Summary table not found in the expected cell.")

```

```text?code_stdout&code_event_index=6
FINAL EVT (q=0.993) PERFORMANCE SUMMARY -------------------------
                               f1_score  precision   recall   TP   FN   FP  total_predicted
score_column                                                                               
anomalyscore_h0_auc_over_l2    0.814026   0.846354 0.784077  650  179  118              787
anomalyscore_h0_auc            0.803750   0.833982 0.775633  643  186  128              790
anomalyscore_tail_q90          0.742732   0.681087 0.816647  677  152  317              992
anomalyscore_h0_dom_share      0.742732   0.681087 0.816647  677  152  317              992
anomalyscore_sum_centroid      0.722777   0.640858 0.828709  687  142  385             1356
anomalyscore_bottleneck        0.715851   0.907407 0.591074  490  339   50              545
anomalyscore_h0_auc_over_max   0.715851   0.907407 0.591074  490  339   50              545
anomalyscore_pete              0.714661   0.633147 0.820265  680  149  394             1371
anomalyscore_h0_l2norm         0.703940   0.630124 0.797346  661  168  388             1355
anomalyscore_h0_energy_conc    0.604651   0.528933 0.705669  585  244  521             1105
anomalyscore_h0_cen_to_energy  0.604651   0.528933 0.705669  585  244  521             1105
anomalyscore_h0_gini           0.530144   0.439334 0.668275  554  275  707             1253
anomalyscore_h0_tail_curve     0.209081   0.185915 0.238842  198  631  867             1091


```

Based on the analysis of the Yahoo A4 (Real Web Traffic) dataset in the provided notebook (`TDA_codes_for_YAHOO_A4.ipynb`), here are the **F1 Scores** for the various topological features.

These scores were achieved using **Extreme Value Theory (EVT)** for adaptive thresholding, with a **Gate Quantile of 0.78** and a **Final Quantile of 0.993**.

### Top Performing Features (A4 Dataset)

The best-performing feature for the real traffic data is **AUC over L2**, achieving an F1 score of **0.814**.

| Rank | Feature Name | **F1 Score** | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **`anomalyscore_h0_auc_over_l2`** | **0.814026** | 0.846 | 0.784 |
| 2 | `anomalyscore_h0_auc` | **0.803750** | 0.834 | 0.776 |
| 3 | `anomalyscore_tail_q90` | **0.742732** | 0.681 | 0.817 |
| 4 | `anomalyscore_h0_dom_share` | **0.742732** | 0.681 | 0.817 |
| 5 | `anomalyscore_sum_centroid` | **0.722777** | 0.641 | 0.829 |

### Full Performance Summary Table

| Feature Name | F1 Score | Precision | Recall | TP | FN | FP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `anomalyscore_h0_auc_over_l2` | 0.814026 | 0.846354 | 0.784077 | 650 | 179 | 118 |
| `anomalyscore_h0_auc` | 0.803750 | 0.833982 | 0.775633 | 643 | 186 | 128 |
| `anomalyscore_tail_q90` | 0.742732 | 0.681087 | 0.816647 | 677 | 152 | 317 |
| `anomalyscore_h0_dom_share` | 0.742732 | 0.681087 | 0.816647 | 677 | 152 | 317 |
| `anomalyscore_sum_centroid` | 0.722777 | 0.640858 | 0.828709 | 687 | 142 | 385 |
| `anomalyscore_bottleneck` | 0.715851 | 0.907407 | 0.591074 | 490 | 339 | 50 |
| `anomalyscore_h0_auc_over_max` | 0.715851 | 0.907407 | 0.591074 | 490 | 339 | 50 |
| `anomalyscore_pete` | 0.714661 | 0.633147 | 0.820265 | 680 | 149 | 394 |
| `anomalyscore_h0_l2norm` | 0.703940 | 0.630124 | 0.797346 | 661 | 168 | 388 |
| `anomalyscore_h0_energy_conc` | 0.604651 | 0.528933 | 0.705669 | 585 | 244 | 521 |
| `anomalyscore_h0_cen_to_energy` | 0.604651 | 0.528933 | 0.705669 | 585 | 244 | 521 |
| `anomalyscore_h0_gini` | 0.530144 | 0.439334 | 0.668275 | 554 | 275 | 707 |
| `anomalyscore_h0_tail_curve` | 0.209081 | 0.185915 | 0.238842 | 198 | 631 | 867 |

### Key Insights for A4

  * **Best Feature:** `anomalyscore_h0_auc_over_l2` (Area Under Curve normalized by Energy). This feature is particularly robust for real web traffic because it detects topological changes relative to the signal's total energy, making it effective even when the baseline traffic volume changes (e.g., day vs. night cycles).
  * **Bottleneck Performance:** Interestingly, the `bottleneck` feature (max lifetime), which often performs well on synthetic data, had a lower F1 score (**0.716**) here. It had very high precision (**0.907**) but poor recall (**0.591**), meaning it is very sure when it finds an anomaly but misses almost 40% of them.
  * **Comparison to A3:** The F1 scores for A4 are generally lower than those for the A3 dataset (where the best score was \~0.89). This is expected as real-world data (A4) is noisier and more complex than synthetic benchmarks (A3).
