# 🎬 Movie Recommendation System

A deep learning project that recommends personalized movies using **Neural Collaborative Filtering (NCF)**, trained on the **MovieLens 20M** dataset.

**Authors:** Aya Aburahmeh · Hala AlJafary

---

## 📌 Overview

| | |
|---|---|
| **Dataset** | MovieLens 20M (~20 million ratings) |
| **Technique** | Neural Collaborative Filtering (Deep Learning) |
| **Problem type** | Regression (predict a user's rating for a movie) |
| **Framework** | TensorFlow / Keras |
| **Test RMSE** | **0.2057** (on the 0–1 normalized scale, ≈ 0.93 stars on the original 0.5–5 scale) |

**Pipeline:** Data Preparation → Model Design → Training → Evaluation → Recommendations

---

## 📂 Repository Structure

```
.
├── movie-Recommendation.ipynb   # Full code: preprocessing, model, training, evaluation, recommendations
├── Movie-Recommendation.pptx    # Project presentation slides
└── README.md
```

---

## 📊 Dataset

- **Name:** MovieLens 20M
- **Source:** [Kaggle – MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) (GroupLens)
- **Files used:** `rating.csv`, `movie.csv`
- **Features:** `userId`, `movieId`, `timestamp`
- **Target:** `rating` (normalized to 0–1)

> The dataset is **not included** in this repository because of its size. Download it from Kaggle and place it next to the notebook.

---

## 🧹 Data Preparation

1. **Remove duplicates**: keep the most recent rating for each user–movie pair.
2. **Filter sparse data**: keep users with **≥ 50 ratings** and movies with **≥ 10 ratings**.
   - 20,000,263 → **18,300,248** ratings
   - **85,307** users and **15,359** movies remain
3. **Normalize ratings**: Min-Max scaling from 0.5–5.0 to 0.0–1.0.
4. **Encode IDs**: map user and movie IDs to sequential integers for the embedding layers.
5. **Chronological split**: sort by timestamp, then **80% train / 20% test** to avoid data leakage.
   - Train: 14,640,198 samples
   - Test: 3,660,050 samples

---

## 🧠 Model Architecture

```
User ID  → Embedding(32) → Flatten ─┐
                                    ├→ Concatenate(64) → Dense(64, ReLU) → Dropout(0.5)
Movie ID → Embedding(32) → Flatten ─┘                  → Dense(32, ReLU) → Dropout(0.3)
                                                       → Dense(1, Sigmoid)
```

| Design choice | Value | Why |
|---|---|---|
| Embedding size | 32 | Balance between expressiveness and generalization |
| Hidden activation | ReLU (+ He Normal init) | Stable gradients |
| Output activation | Sigmoid | Constrains predictions to [0, 1] |
| Dense layers | 64 → 32 → 1 | Progressive compression |
| Dropout | 0.5 / 0.3 | Prevents memorization |
| L2 regularization | 1e-5 on embeddings | Limits embedding weight growth |

---

## 🏋️ Training

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Loss | MSE |
| Metric | RMSE |
| Batch size | 2048 |
| Max epochs | 10 |

**Callbacks**
- `EarlyStopping`: monitors `val_loss`, patience = 2, restores best weights.
- `ReduceLROnPlateau`: if `val_loss` stalls for 1 epoch, LR × 0.2 (min LR = 1e-6).

---

## ✅ Results

- **Final Test RMSE: 0.2057** (normalized scale)
- Multiplying by the rating range (4.5) gives an average error of roughly **0.93 stars**.

### Example: recommendations for User 1

The model recommends the top-10 movies the user has **not** watched yet, ranked by predicted rating. For example, for user 1 it suggests titles such as *Braveheart*, *Forrest Gump*, *Schindler's List*, *The Matrix*, and *Gladiator*.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install pandas numpy tensorflow matplotlib jupyter
```

### 3. Download the dataset

Download `rating.csv` and `movie.csv` from [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) and put them in the project folder.

### 4. Update the file paths

In the notebook, replace the local paths with your own, for example:

```python
df = pd.read_csv("rating.csv")
movies_metadata = pd.read_csv("movie.csv")
```

### 5. Run the notebook

```bash
jupyter notebook movie-Recommendation.ipynb
```

The trained model is saved at the end as `my_final_ncf_model.keras`.

> ⚠️ Training on the full dataset is heavy (each epoch took ~7–10 minutes on CPU in our run). A GPU is recommended.

---

## 🔮 Possible Improvements

- Add side information (genres, tags, release year) to the model.
- Compare against baselines such as Matrix Factorization or a hybrid model.
- Evaluate with ranking metrics (Precision@K, Recall@K, NDCG).
- Use a separate validation set in addition to the test set.

---

## 🛠️ Tech Stack

Python · TensorFlow / Keras · Pandas · NumPy · Matplotlib

---

## 📄 License

Add your preferred license here (e.g., MIT). Note that the MovieLens dataset has its own [usage license](https://grouplens.org/datasets/movielens/).
