import csv
import numpy as np

from src.linear_regression import LinearRegression
from src.preprocessing import StandardScalarScratch
from src.metrics import mse, rmse, mae, r2_score
from src.utils import train_test_split

def load_csv_data(file_path):
    xs, ys = [], []
    with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.reader(f) if row and any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError("CSV is empty.")
        header = [h.strip() for h in rows[0]]
        try:
            x_idx = header.index("x")
            y_idx = header.index("y")
        except ValueError as exc:
            raise ValueError("CSV header must include 'x' and 'y'.") from exc
        for row in rows[1:]:
            xs.append(float(row[x_idx]))
            ys.append(float(row[y_idx]))
    return np.array(xs), np.array(ys)

def main():
    # Load data
    X, y = load_csv_data("data/sample.csv")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Preprocess data
    scaler = StandardScalarScratch()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression(lr=0.05, epochs=1000, verbose=True)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Evaluate
    print("\n=== Learned Parameters ===")
    print(f"w: {model.w}")
    print(f"b: {model.b:.6f}")

    print("\n=== Train Metrics ===")
    print(f"MSE : {mse(y_train, y_pred_train):.6f}")
    print(f"RMSE: {rmse(y_train, y_pred_train):.6f}")
    print(f"MAE : {mae(y_train, y_pred_train):.6f}")
    print(f"R2  : {r2_score(y_train, y_pred_train):.6f}")

    print("\n=== Test Metrics ===")
    print(f"MSE : {mse(y_test, y_pred_test):.6f}")
    print(f"RMSE: {rmse(y_test, y_pred_test):.6f}")
    print(f"MAE : {mae(y_test, y_pred_test):.6f}")
    print(f"R2  : {r2_score(y_test, y_pred_test):.6f}")

if __name__ == "__main__":
    main()
