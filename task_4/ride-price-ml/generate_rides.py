import numpy as np
import pandas as pd
from pathlib import Path


def generate_rides(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic ride dataset consistent with the notebook description.
    """
    rng = np.random.default_rng(random_state)

    # Numeric features
    distance_km = np.round(rng.uniform(1, 25, size=n_samples), 2)
    base_speed_kmph = rng.normal(30, 5, size=n_samples)
    base_speed_kmph = np.clip(base_speed_kmph, 10, 60)

    true_duration = distance_km / base_speed_kmph * 60
    noise_duration = rng.normal(0, 5, size=n_samples)
    duration_min = np.clip(true_duration + noise_duration, 5, 90)

    # Categorical features
    time_of_day = rng.choice(
        ["morning", "afternoon", "evening", "night"],
        size=n_samples,
        p=[0.25, 0.3, 0.3, 0.15],
    )
    traffic_level = rng.choice(
        ["low", "medium", "high"],
        size=n_samples,
        p=[0.3, 0.4, 0.3],
    )
    weather = rng.choice(
        ["clear", "rainy", "stormy"],
        size=n_samples,
        p=[0.7, 0.25, 0.05],
    )
    demand_level = rng.choice(
        ["low", "normal", "high"],
        size=n_samples,
        p=[0.2, 0.5, 0.3],
    )
    pickup_zone = rng.choice(
        ["city_center", "suburbs", "airport"],
        size=n_samples,
        p=[0.4, 0.4, 0.2],
    )

    # Price construction
    base_fare = 2.0
    price_per_km = 0.8
    price_per_min = 0.3

    price = base_fare + distance_km * price_per_km + duration_min * price_per_min

    # Time of day effect
    for i, tod in enumerate(time_of_day):
        if tod in ["morning", "evening"]:
            price[i] *= 1.15
        elif tod == "night":
            price[i] *= 1.05

    # Traffic effect
    for i, tl in enumerate(traffic_level):
        if tl == "high":
            price[i] *= 1.2
        elif tl == "medium":
            price[i] *= 1.05

    # Weather effect
    for i, w in enumerate(weather):
        if w == "rainy":
            price[i] *= 1.05
        elif w == "stormy":
            price[i] *= 1.15

    # Demand effect
    for i, d in enumerate(demand_level):
        if d == "high":
            price[i] *= 1.3
        elif d == "low":
            price[i] *= 0.9

    # Pickup zone effect
    for i, z in enumerate(pickup_zone):
        if z == "airport":
            price[i] += 5
        elif z == "city_center":
            price[i] += 1.5

    # Noise and clipping
    price = price + rng.normal(0, 2.5, size=n_samples)
    price = np.round(np.clip(price, 5, None), 2)

    df = pd.DataFrame(
        {
            "distance_km": distance_km,
            "duration_min": duration_min,
            "time_of_day": time_of_day,
            "traffic_level": traffic_level,
            "weather": weather,
            "demand_level": demand_level,
            "pickup_zone": pickup_zone,
            "ride_price": price,
        }
    )
    return df


def main() -> None:
    df = generate_rides()
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "rides.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset with shape {df.shape} to {out_path}")


if __name__ == "__main__":
    main()

