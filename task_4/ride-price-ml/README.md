# Ride Price Estimation System

## Project Overview

This mini project builds an end-to-end **machine learning system to estimate ride prices** based on trip and contextual information, similar to a taxi or ride-hailing platform.

The main notebook is:

- `notebook/ride_price_model.ipynb`

The notebook walks through:

- Problem framing and ML mindset  
- Synthetic dataset design and feature justification  
- Data exploration and cleaning  
- Regression model for price prediction  
- Classification model for high-cost vs low-cost rides  
- Model evaluation, feature importance, and ethical reflection  

## Dataset Description

- Location: `data/rides.csv`  
- Rows: 600 synthetic rides (I can change this number by changing the generator)  
- Target (continuous): `ride_price` — final ride price in an arbitrary currency.

I **created this dataset myself** using the code in this project (see `generate_rides.py` and the notebook).  
It is not downloaded or copied from any external dataset.

### Features Used and Justification

1. `distance_km` (numeric)  
   - Distance of the trip in kilometers. Longer trips usually cost more, so this is a core pricing driver.

2. `duration_min` (numeric)  
   - Trip duration in minutes. Captures time-based pricing (e.g., waiting in traffic or slow routes).

3. `time_of_day` (categorical: `morning`, `afternoon`, `evening`, `night`)  
   - Models peak vs off-peak pricing; morning and evening rush hours often have higher prices.

4. `traffic_level` (categorical: `low`, `medium`, `high`)  
   - Higher traffic can increase travel time and sometimes introduce congestion charges.

5. `weather` (categorical: `clear`, `rainy`, `stormy`)  
   - Bad weather can increase risk and slow down traffic, which can be reflected in higher prices.

6. `demand_level` (categorical: `low`, `normal`, `high`)  
   - Simulates surge pricing when many passengers are requesting rides at the same time.

7. `pickup_zone` (categorical: `city_center`, `suburbs`, `airport`)  
   - Some zones, especially airports or busy city centers, often have extra surcharges or higher base fares.

**Excluded feature (for justification):**

- `driver_rating` (1–5 stars) was considered but **not** included. Ratings can be subjective and biased, and using them directly for pricing might unfairly charge more to or for certain drivers/passengers. For this educational project, it is safer to exclude rating-based pricing.

## How to Run the Notebook

### Option 1: Local (Python)

1. Clone the repository and move into the project:

   ```bash
   git clone <your-repo-url>.git
   cd ride-price-ml
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

4. Make sure the dataset exists (if needed, regenerate):

   ```bash
   python generate_rides.py
   ```

   This creates/overwrites `data/rides.csv`.

5. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `notebook/ride_price_model.ipynb` and run all cells.

### Option 2: Google Colab

1. Upload the `ride-price-ml` folder (or at least `notebook/ride_price_model.ipynb` and `data/rides.csv`) to your Drive or Colab workspace.
2. Open `notebook/ride_price_model.ipynb` in Colab.
3. Make sure the path to the CSV is correct (for example, `../data/rides.csv` if the notebook is in `notebook/`).
4. Run all cells (Runtime → Run all).

## Key Findings (Summary)

- **Regression:** A Linear Regression model can learn the synthetic pricing rule, with ride price increasing mainly with `distance_km` and `duration_min`, and adjusted by demand, traffic, time of day, weather, and pickup zone.
- **Classification:** A Logistic Regression model can classify rides into **high-cost** vs **low-cost** using the same features with reasonable accuracy, based on a median price threshold.
- **Most influential feature:** Distance (`distance_km`) is typically the most important feature in the regression model, with duration and high-demand/peak-time indicators also contributing.
- **Data quality:** Handling missing values, encoding categorical variables correctly, and scaling numerical features are all important for stable model performance, especially when moving from synthetic data to real-world data.

## Notes and Limitations

- The dataset is **synthetic** and relatively small, so it does not capture all real-world complexity (holidays, events, different cities, etc.).
- Real deployment would require much larger and real data, careful monitoring, and fairness considerations to avoid unfair pricing behavior.

