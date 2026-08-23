import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred):
    """Calculates performance evaluation metrics as defined in the manuscript.

    Equations (14)-(19)
    """
    n = len(y_true)

    # Root Mean Square Error (RMSE) - Eq. (14)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Pearson Correlation Coefficient (CC) - Eq. (15)
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    num = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    den = np.sqrt(
        np.sum((y_true - y_true_mean) ** 2)
        * np.sum((y_pred - y_pred_mean) ** 2)
    )
    cc = num / den if den != 0 else 0.0

    # Mean Absolute Error (MAE) - Eq. (16)
    mae = np.mean(np.abs(y_true - y_pred))

    # Mean Bias Error (MBE) - Eq. (17)
    mbe = np.mean(y_pred - y_true)

    # Mean Absolute Percentage Error (MAPE) - Eq. (18)
    # Avoid division by zero by filtering or handling near-zero true values
    mape = np.mean(np.abs((y_true - y_pred) / y_true))

    # Coefficient of Determination (R2) - Eq. (19)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        "CC": cc,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "MBE": mbe,
    }


def main():
    # Load dataset
    data_path = "../Data/SpurDike.tsv"
    df = pd.read_csv(data_path, sep="\t")

    # Extract target and input features from the dataset
    # target: ds/l (dsl)
    y_true = df["dsl"].values

    # Predictors:
    # VVc: V/Vc, yl: y/l, ld50: l/d50, Fd50: Fd50
    VVc = df["VVc"].values
    yl = df["yl"].values
    ld50 = df["ld50"].values
    Fd50 = df["Fd50"].values

    # Physical Constants
    S = 2.65  # Sediment specific gravity
    delta = S - 1  # Relative submerged density (~1.65)
    g = 9.81  # Gravitational acceleration (m/s^2)

    # ----------------------------------------------------
    # 1. Dey2005 Method
    # Equation: dsl = 7.281 * (Fe ** 0.314) * (yl ** 0.128) * (ld50 ** -0.167)
    # where Fe = (V - 0.5 * Vc) / sqrt(delta * g * d50)
    # Transformed: Fe = Fd50 * (VVc - 0.5)/VVc
    # ----------------------------------------------------
    Fe = Fd50 * (VVc - 0.5)/VVc
    # Handle non-positive Fe if V < 0.5 * Vc to avoid complex numbers
    # Fe_safe = Fe
    Fe_safe = np.maximum(Fe, 1e-6)
    pred_Dey2005 = (
        7.281 * (Fe_safe**0.314) * (yl**0.128) * (ld50**-0.167)
    )

    # ----------------------------------------------------
    # 2. Pandey2016 Method
    # Equation: dsl = 5.686 * (Fz ** 0.276) * (yl ** 0.248) * (d50l ** 0.163)
    # where Fz = (V / sqrt(g * l)) * (V / Vc - 0.5)
    # Note: d50l = 1 / ld50
    # ----------------------------------------------------
    # Reconstructing V / sqrt(g * l) = Fd50 * sqrt(delta * d50 / l) = Fd50 * sqrt(delta / ld50)
    Fz = Fd50 * np.sqrt(delta / ld50) * (VVc - 0.5)
    # Fz_safe = Fz
    Fz_safe = np.maximum(Fz, 1e-6)
    d50l = 1.0 / ld50
    pred_Pandey2016 = (
        5.686 * (Fz_safe**0.276) * (yl**0.248) * (d50l**0.163)
    )

    # ----------------------------------------------------
    # 3. Singh2024 (Linear version) Method
    # Equation: dsl = 0.2404 - 0.1132*Fd50 + 0.7289*VVc + 0.5271*yl - 0.0001*ld50
    # ----------------------------------------------------
    pred_Singh2024_Linear = (
        0.2404 - 0.1132 * Fd50 + 0.7289 * VVc + 0.5271 * yl - 0.0001 * ld50
    )

    # ----------------------------------------------------
    # 4. Singh2024 (Non-Linear version) Method
    # Equation: dsl = 1.9670 * (Fd50 ** -0.0419) * (VVc ** 0.4464) * (yl ** 0.5161) * (ld50 ** -0.0897)
    # ----------------------------------------------------
    pred_Singh2024_NonLinear = (
        1.9670
        * (Fd50**-0.0419)
        * (VVc**0.4464)
        * (yl**0.5161)
        * (ld50**-0.0897)
    )

    # Store predictions in a dictionary
    predictions = {
        "Dey2005": pred_Dey2005,
        "Pandey2016": pred_Pandey2016,
        "Singh2024-Linear": pred_Singh2024_Linear,
        "Singh2024-NonLinear": pred_Singh2024_NonLinear,
    }

    # Evaluate metrics for each model
    results = []
    for model_name, y_pred in predictions.items():
        metrics = compute_metrics(y_true, y_pred)
        metrics["Method"] = model_name
        results.append(metrics)

    # Format output as DataFrame
    results_df = pd.DataFrame(results)[
        ["Method", "CC", "R2", "RMSE", "MAE", "MAPE", "MBE"]
    ]

    print("\nModel Performance Evaluation Results:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()