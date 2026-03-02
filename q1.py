from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class FeatureSpec:
	numeric: tuple[str, ...]
	categorical: tuple[str, ...]


def load_data(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	required = {
		"Mthly_HH_Income",
		"Mthly_HH_Expense",
		"No_of_Fly_Members",
		"Emi_or_Rent_Amt",
		"Highest_Qualified_Member",
		"No_of_Earning_Members",
	}
	missing = required.difference(df.columns)
	if missing:
		raise ValueError(f"missing required columns: {sorted(missing)}")

	numeric_cols = [
		"Mthly_HH_Income",
		"Mthly_HH_Expense",
		"No_of_Fly_Members",
		"Emi_or_Rent_Amt",
		"No_of_Earning_Members",
	]
	for col in numeric_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")

	df["Highest_Qualified_Member"] = df["Highest_Qualified_Member"].astype("string")

	df = df.dropna(
		subset=[
			"Mthly_HH_Income",
			"Mthly_HH_Expense",
			"No_of_Fly_Members",
			"Emi_or_Rent_Amt",
			"No_of_Earning_Members",
			"Highest_Qualified_Member",
		]
	).reset_index(drop=True)

	return df


def define_problem(df: pd.DataFrame):
	feature_spec = FeatureSpec(
		numeric=(
			"Mthly_HH_Income",
			"No_of_Fly_Members",
			"No_of_Earning_Members",
			"Emi_or_Rent_Amt",
		),
		categorical=("Highest_Qualified_Member",),
	)

	X = df[list(feature_spec.numeric + feature_spec.categorical)].copy()
	y = df["Mthly_HH_Expense"].copy()
	return X, y, feature_spec


def make_preprocessor(feature_spec: FeatureSpec, *, scale_numeric: bool):
	numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
	if scale_numeric:
		numeric_steps.append(("scaler", StandardScaler()))

	numeric_pipe = Pipeline(steps=numeric_steps)
	categorical_pipe = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			(
				"onehot",
				OneHotEncoder(handle_unknown="ignore", sparse_output=False),
			),
		]
	)

	return ColumnTransformer(
		transformers=[
			("num", numeric_pipe, list(feature_spec.numeric)),
			("cat", categorical_pipe, list(feature_spec.categorical)),
		],
		remainder="drop",
	)


def evaluate_models(
	X: pd.DataFrame,
	y: pd.Series,
	feature_spec: FeatureSpec,
	*,
	random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
	cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=random_state)
	scoring = {
		"mae": "neg_mean_absolute_error",
		"rmse": "neg_root_mean_squared_error",
		"r2": "r2",
	}

	candidates: list[tuple[str, Pipeline]] = [
		(
			"dummy_median",
			Pipeline(
				steps=[
					("pre", make_preprocessor(feature_spec, scale_numeric=False)),
					("model", DummyRegressor(strategy="median")),
				]
			),
		),
		(
			"ridge",
			Pipeline(
				steps=[
					("pre", make_preprocessor(feature_spec, scale_numeric=True)),
					("model", Ridge(alpha=1.0, random_state=random_state)),
				]
			),
		),
		(
			"hgb",
			Pipeline(
				steps=[
					("pre", make_preprocessor(feature_spec, scale_numeric=False)),
					(
						"model",
						HistGradientBoostingRegressor(
							learning_rate=0.08,
							max_depth=4,
							max_leaf_nodes=31,
							min_samples_leaf=5,
							random_state=random_state,
						),
					),
				]
			),
		),
	]

	rows: list[dict[str, Any]] = []
	for name, pipe in candidates:
		scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=None)
		rows.append(
			{
				"model": name,
				"mae_mean": float(-np.mean(scores["test_mae"])),
				"mae_std": float(np.std(-scores["test_mae"])),
				"rmse_mean": float(-np.mean(scores["test_rmse"])),
				"rmse_std": float(np.std(-scores["test_rmse"])),
				"r2_mean": float(np.mean(scores["test_r2"])),
				"r2_std": float(np.std(scores["test_r2"])),
			}
		)

	results = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)
	return results


def fit_best_model(
	X: pd.DataFrame,
	y: pd.Series,
	feature_spec: FeatureSpec,
	best_model_name: str,
	*,
	random_state: int = RANDOM_STATE,
) -> Pipeline:
	if best_model_name == "dummy_median":
		pipe = Pipeline(
			steps=[
				("pre", make_preprocessor(feature_spec, scale_numeric=False)),
				("model", DummyRegressor(strategy="median")),
			]
		)
	elif best_model_name == "ridge":
		pipe = Pipeline(
			steps=[
				("pre", make_preprocessor(feature_spec, scale_numeric=True)),
				("model", Ridge(alpha=1.0, random_state=random_state)),
			]
		)
	elif best_model_name == "hgb":
		pipe = Pipeline(
			steps=[
				("pre", make_preprocessor(feature_spec, scale_numeric=False)),
				(
					"model",
					HistGradientBoostingRegressor(
						learning_rate=0.08,
						max_depth=4,
						max_leaf_nodes=31,
						min_samples_leaf=5,
						random_state=random_state,
					),
				),
			]
		)
	else:
		raise ValueError(f"Unknown model: {best_model_name}")

	pipe.fit(X, y)
	return pipe


def build_demo_profiles() -> pd.DataFrame:
	"""Creates a variety of demographic groups for demonstration."""
	return pd.DataFrame(
		[
			{
				"group": "Young single (renting)",
				"salary_monthly": 20000,
				"age": 25,
				"No_of_Fly_Members": 1,
				"No_of_Earning_Members": 1,
				"Highest_Qualified_Member": "Under-Graduate",
				"Emi_or_Rent_Amt": 6000,
			},
			{
				"group": "Mid-career couple",
				"salary_monthly": 45000,
				"age": 38,
				"No_of_Fly_Members": 2,
				"No_of_Earning_Members": 2,
				"Highest_Qualified_Member": "Graduate",
				"Emi_or_Rent_Amt": 10000,
			},
			{
				"group": "Family of 4 (mortgage)",
				"salary_monthly": 70000,
				"age": 42,
				"No_of_Fly_Members": 4,
				"No_of_Earning_Members": 2,
				"Highest_Qualified_Member": "Post-Graduate",
				"Emi_or_Rent_Amt": 18000,
			},
			{
				"group": "Large household, single earner",
				"salary_monthly": 30000,
				"age": 33,
				"No_of_Fly_Members": 6,
				"No_of_Earning_Members": 1,
				"Highest_Qualified_Member": "Illiterate",
				"Emi_or_Rent_Amt": 4000,
			},
			{
				"group": "High income, low dependents",
				"salary_monthly": 100000,
				"age": 45,
				"No_of_Fly_Members": 2,
				"No_of_Earning_Members": 2,
				"Highest_Qualified_Member": "Professional",
				"Emi_or_Rent_Amt": 20000,
			},
		]
	)


def predict_disposable_income(
	model: Pipeline,
	profiles: pd.DataFrame,
	*,
	clip_at_zero: bool = False,
) -> pd.DataFrame:
	"""Predicts monthly disposable income for profiles.

	Notes:
	- `age` is accepted for the prompt requirement, but this dataset has no age column;
	  it is not used by the trained model.
	- If `salary_annual` is provided, it is converted to `salary_monthly`.
	"""
	df = profiles.copy()
	if "salary_monthly" not in df.columns and "salary_annual" in df.columns:
		df["salary_monthly"] = df["salary_annual"] / 12.0
	if "salary_monthly" not in df.columns:
		raise ValueError("Provide `salary_monthly` or `salary_annual` in profiles")

	# Map API-friendly names to training feature names
	X = pd.DataFrame(
		{
			"Mthly_HH_Income": pd.to_numeric(df["salary_monthly"], errors="coerce"),
			"No_of_Fly_Members": pd.to_numeric(df["No_of_Fly_Members"], errors="coerce"),
			"No_of_Earning_Members": pd.to_numeric(
				df["No_of_Earning_Members"], errors="coerce"
			),
			"Highest_Qualified_Member": df["Highest_Qualified_Member"].astype("string"),
			"Emi_or_Rent_Amt": pd.to_numeric(df["Emi_or_Rent_Amt"], errors="coerce"),
		}
	)

	pred_non_housing_expense = model.predict(X)
	essentials = pred_non_housing_expense + X["Emi_or_Rent_Amt"].to_numpy()
	disposable = X["Mthly_HH_Income"].to_numpy() - essentials
	if clip_at_zero:
		disposable = np.maximum(disposable, 0)

	out = df.copy()
	out["pred_non_housing_expense"] = pred_non_housing_expense
	out["pred_essentials_total"] = essentials
	out["pred_disposable_monthly"] = disposable
	return out


def main() -> None:
	data_path = Path(__file__).resolve().parent / "data" / "Inc_Exp_Data.csv"
	df = load_data(data_path)
	X, y, feature_spec = define_problem(df)

	results = evaluate_models(X, y, feature_spec)
	best_name = str(results.loc[0, "model"])
	model = fit_best_model(X, y, feature_spec, best_name)

	print("Model selection (5x10 RepeatedKFold; target = Mthly_HH_Expense):")
	with pd.option_context("display.max_columns", 20, "display.width", 140):
		print(results)
	print(f"\nSelected model: {best_name}")

	demo = build_demo_profiles()
	demo_pred = predict_disposable_income(model, demo, clip_at_zero=False)

	print("\nDemo: predicted monthly disposable income by demographic group")
	cols = [
		"group",
		"salary_monthly",
		"age",
		"No_of_Fly_Members",
		"No_of_Earning_Members",
		"Highest_Qualified_Member",
		"Emi_or_Rent_Amt",
		"pred_non_housing_expense",
		"pred_essentials_total",
		"pred_disposable_monthly",
	]
	with pd.option_context("display.max_columns", 50, "display.width", 160):
		print(demo_pred[cols].round(2))

	print(
		"\nNote: The dataset has no age column; age is accepted in the demo but is not used by the trained model."
	)


if __name__ == "__main__":
	main()

