import os
import joblib
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Predict ride price using a trained model.")
    parser.add_argument("--distance", type=float, default=10.0, help="Distance in km")
    parser.add_argument("--duration", type=float, default=25.0, help="Duration in min")
    parser.add_argument("--time", type=str, default="afternoon", choices=["morning", "afternoon", "evening", "night"])
    parser.add_argument("--traffic", type=str, default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--weather", type=str, default="clear", choices=["clear", "rainy", "stormy"])
    parser.add_argument("--demand", type=str, default="normal", choices=["low", "normal", "high"])
    parser.add_argument("--zone", type=str, default="city_center", choices=["city_center", "suburbs", "airport"])
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'random_forest.joblib')
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run train.py first.")
        return
        
    model = joblib.load(model_path)
    
    input_df = pd.DataFrame([{
        "distance_km": args.distance,
        "duration_min": args.duration,
        "time_of_day": args.time,
        "traffic_level": args.traffic,
        "weather": args.weather,
        "demand_level": args.demand,
        "pickup_zone": args.zone,
    }])
    
    price = model.predict(input_df)[0]
    print(f"\n--- Ride Price Prediction ---")
    print(f"Features: Distance={args.distance}km, Duration={args.duration}m, Time={args.time}, Traffic={args.traffic}")
    print(f"Features: Weather={args.weather}, Demand={args.demand}, Zone={args.zone}")
    print(f"\nPredicted Price: ${price:.2f}")

if __name__ == "__main__":
    main()
