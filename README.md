# ⚽ Stochastic Football

Premier League match simulator built with **Python**, **Streamlit**, and **Google BigQuery**. This tool uses a dynamic Bayesian-inspired Elo system to model team performance as probability distributions rather than static ratings.

## Key Features

* **Dynamic Elo System:** Ratings that evolve based on "Match Surprise" (Actual vs. Expected points).
* **Uncertainty Modeling (σ):** Incorporates a volatility parameter (Sigma) that expands during shock results and converges during predictable runs.
* **Interactive Simulation:** Adjust Base Elo, Home Field Advantage (HFA), and Momentum (Form) in real-time to see how win probabilities shift.
* **Performance Analytics:** Visualizes the "Performance Overlap" between two teams using Normal Distribution PDF curves.
* **Historical Context:** Displays the last 5 games, comparing Actual Points (Pts) against Model Expectations (xPts).

---

## Technical Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Warehouse:** [Google BigQuery](https://cloud.google.com/bigquery)
* **Math/Stats:** `NumPy` & `SciPy` (Normal Distribution CDF/PDF modeling)
* **Visuals:** `Plotly` for interactive distribution charts

---

## How the Model Works

Unlike standard Elo, this engine models a match as the difference between two random variables:
1.  **Home Performance** ~ $N(Elo_h + HFA + Form_h, \sigma_h)$
2.  **Away Performance** ~ $N(Elo_a + Form_a, \sigma_a)$

The probability of a **Home Win**, **Draw**, or **Away Win** is calculated by finding the area under the resulting difference curve relative to a calibrated `draw_margin`.



---

## 📝 Research & Methodology

The mathematical foundation and calibration of this stochastic model are based on extensive research into Bayesian rating systems and football analytics. For a deep dive into the modelling and hyperparameter tuning process, refer to the full research notebook:

**👉 [Explore the Full Research on Kaggle](https://www.kaggle.com/code/justinus/stochastic-football)**