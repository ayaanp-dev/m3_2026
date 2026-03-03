from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def _try_import_joblib():
	try:
		import joblib
		return joblib
	except Exception:
		return None


def _clip(v: float, lo: float, hi: float) -> float:
	return float(min(max(v, lo), hi))


def _as_float(x):
	if x is None:
		return None
	if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
		return float(x)
	s = str(x).strip()
	if s == "" or s.lower() in {"na", "nan", "null"}:
		return None
	try:
		return float(s)
	except Exception:
		return None


def _normalize_excel_table(df: pd.DataFrame) -> pd.DataFrame:
	if df.shape[0] < 2:
		return df

	col0 = str(df.columns[0])
	if "Unnamed:" in col0 or col0.strip() == "":
		return df

	first_row = df.iloc[0].tolist()
	if not any(isinstance(v, str) and v.strip() for v in first_row):
		return df

	out = df.iloc[1:].copy()
	out.columns = [str(v).strip() if v is not None else "" for v in first_row]
	return out


def _drop_non_data_rows(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
	if label_col not in df.columns:
		return df
	labels = df[label_col].astype(str).str.strip()
	mask = (
		labels.ne("")
		& ~labels.str.startswith("Base:")
		& ~labels.str.startswith("Return to")
		& ~labels.str.startswith("i ")
		& ~labels.str.startswith("ii ")
		& ~labels.str.startswith("iii ")
		& ~labels.str.contains("Excluding participants", case=False, na=False)
		& ~labels.str.contains("could not be derived", case=False, na=False)
	)
	return df.loc[mask].copy()


def load_gsgb_tables(excel_path: str) -> Dict[str, pd.DataFrame]:
	if not os.path.exists(excel_path):
		raise FileNotFoundError(f"Excel file not found: {excel_path}")
	try:
		a3_raw = pd.read_excel(excel_path, sheet_name="Table A.3")
		a9_raw = pd.read_excel(excel_path, sheet_name="Table A.9")
		a13_raw = pd.read_excel(excel_path, sheet_name="Table A.13")
	except ImportError as e:
		raise ImportError(
			"Reading .xlsx requires 'openpyxl'. Install with: pip install openpyxl"
		) from e

	a3 = _drop_non_data_rows(_normalize_excel_table(a3_raw), "Age group (years)")
	a9 = _drop_non_data_rows(_normalize_excel_table(a9_raw), "Sex and age group (years)")
	a13 = _drop_non_data_rows(_normalize_excel_table(a13_raw), "Type of betting participation i")
	return {"a3": a3, "a9": a9, "a13": a13}


def _parse_age_group_label(label: str) -> Optional[str]:
	s = str(label).strip()
	if s.lower().startswith("all "):
		s = s[4:].strip()
	if s.endswith("plus"):
		base = s.replace("plus", "+").strip()
		return base
	if "to" in s:
		parts = [p.strip() for p in s.split("to")]
		if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
			return f"{parts[0]}-{parts[1]}"
	return None


def parse_population_weights(a3: pd.DataFrame) -> List[Tuple[str, str, float]]:
	age_col = "Age group (years)"
	m_col = "Total Great Britain population ii: \nadult males\n(percentage)"
	f_col = "Total Great Britain population ii: \nadult females\n(percentage)"

	if age_col not in a3.columns:
		raise ValueError("Table A.3 parse failed: missing age column")
	if m_col not in a3.columns or f_col not in a3.columns:
		cands = list(a3.columns)
		raise ValueError(f"Table A.3 parse failed: missing population share columns. Columns: {cands}")

	rows: List[Tuple[str, str, float]] = []
	for _, r in a3.iterrows():
		age = _parse_age_group_label(r.get(age_col))
		if not age:
			continue
		m = _as_float(r.get(m_col))
		f = _as_float(r.get(f_col))
		if m is None or f is None:
			continue
		rows.append(("M", age, float(m)))
		rows.append(("F", age, float(f)))

	total = sum(w for _, _, w in rows)
	if total <= 0:
		raise ValueError("Table A.3 parse failed: no usable rows")
	return [(sex, age, w / total) for sex, age, w in rows]


def parse_online_participation_by_sex_age(a9: pd.DataFrame) -> Dict[Tuple[str, str], float]:
	label_col = "Sex and age group (years)"
	p_col = "Participation in the past four weeks excluding lottery draw only players i\n(percentage)"
	if label_col not in a9.columns or p_col not in a9.columns:
		raise ValueError("Table A.9 parse failed: expected columns not found")

	out: Dict[Tuple[str, str], float] = {}
	for _, r in a9.iterrows():
		label = str(r.get(label_col)).strip()
		p = _as_float(r.get(p_col))
		if p is None:
			continue
		if label.startswith("Males "):
			age = _parse_age_group_label(label.replace("Males", "").strip())
			if age:
				out[("M", age)] = float(p)
		elif label.startswith("Females "):
			age = _parse_age_group_label(label.replace("Females", "").strip())
			if age:
				out[("F", age)] = float(p)

	if not out:
		raise ValueError("Table A.9 parse failed: no (sex, age) rows found")
	return out


def derive_default_sports_share_of_online(a9: pd.DataFrame, a13: pd.DataFrame) -> float:
	label_col = "Sex and age group (years)"
	online_col = "Participation in the past four weeks excluding lottery draw only players i\n(percentage)"
	all_online = None
	for _, r in a9.iterrows():
		if str(r.get(label_col)).strip() == "All participants":
			all_online = _as_float(r.get(online_col))
			break
	if all_online is None or all_online <= 0:
		return 0.40

	bet_col = "Type of betting participation i"
	all_col = "All participants\n(percentage)"
	football = None
	for _, r in a13.iterrows():
		if str(r.get(bet_col)).strip().lower() == "live football":
			football = _as_float(r.get(all_col))
			break
	if football is None or football <= 0:
		return 0.40

	share = float(football / float(all_online))
	return _clip(share, 0.20, 0.80)


def _sample_age_within_group(rng: np.random.Generator, age_group: str) -> float:
	if age_group.endswith("+"):
		lo = int(age_group[:-1])
		hi = 90
		return float(rng.integers(lo, hi + 1))
	if "-" in age_group:
		lo_s, hi_s = age_group.split("-", 1)
		lo, hi = int(lo_s), int(hi_s)
		return float(rng.integers(lo, hi + 1))
	return float("nan")


@dataclass(frozen=True)
class Assumptions:
	country: str
	house_edge: float
	sports_share_of_online: float
	risk_alpha: float
	risk_beta: float
	base_bet_median: float
	base_sigma_per_sqrt_wager: float
	t_df: float


def default_assumptions(country: str, sports_share_of_online: float) -> Assumptions:
	c = country.lower().strip()
	if c not in {"uk", "us"}:
		raise ValueError("--country must be 'uk' or 'us'")
	house_edge = 0.065 if c == "uk" else 0.085
	base_bet_median = 8.0 if c == "uk" else 12.0
	return Assumptions(
		country=c,
		house_edge=house_edge,
		sports_share_of_online=float(sports_share_of_online),
		risk_alpha=2.2,
		risk_beta=3.3,
		base_bet_median=base_bet_median,
		base_sigma_per_sqrt_wager=1.8,
		t_df=4.0,
	)


def _sex_multiplier(sex: str) -> float:
	return 1.15 if sex == "M" else 0.95


def _age_multiplier(age: float) -> float:
	if math.isnan(age):
		return 1.0
	if age <= 24:
		return 1.10
	if age <= 34:
		return 1.05
	if age <= 44:
		return 1.00
	if age <= 54:
		return 0.98
	if age <= 64:
		return 0.92
	return 0.85


def simulate_individual_year(
	rng: np.random.Generator,
	*,
	sex: str,
	age_group: str,
	p_online_4w_ex_lottery: float,
	assumptions: Assumptions,
):
	age = _sample_age_within_group(rng, age_group)

	p_sports_4w = _clip(p_online_4w_ex_lottery * assumptions.sports_share_of_online, 0.0, 1.0)
	is_sports = float(rng.random() < p_sports_4w)

	risk = float(rng.beta(assumptions.risk_alpha, assumptions.risk_beta))
	risk = _clip(risk + (0.03 if sex == "M" else -0.01) + (0.02 if age <= 34 else -0.01), 0.0, 1.0)

	if is_sports <= 0.0:
		return {
			"age": age,
			"risk_tolerance": risk,
			"p_sports_4w": p_sports_4w,
			"is_sports_gambler_4w": 0.0,
			"sessions_year": 0.0,
			"bets_per_session": 0.0,
			"avg_bet_size": 0.0,
			"total_wager": 0.0,
			"expected_net": 0.0,
			"realized_net": 0.0,
		}

	mean_4w = 1.2 + 6.0 * risk
	mean_4w *= _sex_multiplier(sex) * _age_multiplier(age)
	sessions_4w = int(max(1, rng.poisson(lam=max(0.2, mean_4w))))
	sessions_year = int(_clip(13.0 * sessions_4w, 1, 2500))

	mean_bps = 1.8 + 5.0 * risk
	mean_bps *= (1.05 if sex == "M" else 0.98)
	bets_per_session = int(_clip(rng.poisson(lam=max(0.5, mean_bps)) + 1, 1, 40))

	median = assumptions.base_bet_median
	median *= _sex_multiplier(sex) * _age_multiplier(age)
	median *= 0.65 + 1.10 * risk
	mu = math.log(max(0.5, median))
	sigma = 0.65 + 0.55 * risk
	avg_bet_size = float(rng.lognormal(mean=mu, sigma=sigma))
	avg_bet_size = _clip(avg_bet_size, 1.0, 500.0)

	total_wager = float(sessions_year * bets_per_session) * avg_bet_size

	expected = -assumptions.house_edge * total_wager

	base_sigma = assumptions.base_sigma_per_sqrt_wager
	stdev = base_sigma * math.sqrt(max(0.0, total_wager)) * (0.7 + 1.8 * risk)
	noise = float(rng.standard_t(df=assumptions.t_df))
	realized = expected + stdev * noise
	realized = max(realized, -total_wager)

	return {
		"age": age,
		"risk_tolerance": risk,
		"p_sports_4w": p_sports_4w,
		"is_sports_gambler_4w": 1.0,
		"sessions_year": float(sessions_year),
		"bets_per_session": float(bets_per_session),
		"avg_bet_size": float(avg_bet_size),
		"total_wager": float(total_wager),
		"expected_net": float(expected),
		"realized_net": float(realized),
	}


def simulate_population(
	*,
	n: int,
	seed: int,
	pop_weights: Sequence[Tuple[str, str, float]],
	online_p: Dict[Tuple[str, str], float],
	assumptions: Assumptions,
) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	sexes = [s for s, _, _ in pop_weights]
	age_groups = [a for _, a, _ in pop_weights]
	weights = np.array([w for _, _, w in pop_weights], dtype=float)
	weights = weights / float(weights.sum())

	choices = rng.choice(len(pop_weights), size=int(n), p=weights)
	rows: List[Dict[str, object]] = []
	for idx in choices.tolist():
		sex = sexes[idx]
		age_group = age_groups[idx]
		p_online = float(online_p.get((sex, age_group), 0.0))
		sim = simulate_individual_year(
			rng,
			sex=sex,
			age_group=age_group,
			p_online_4w_ex_lottery=p_online,
			assumptions=assumptions,
		)
		rows.append(
			{
				"sex": sex,
				"age_group": age_group,
				**sim,
			}
		)

	df = pd.DataFrame(rows)
	df["country"] = assumptions.country
	df["house_edge"] = assumptions.house_edge
	df["sports_share_of_online"] = assumptions.sports_share_of_online
	return df


def _train_test_split(rows: List[Dict[str, object]], seed: int, test_frac: float = 0.2):
	rng = random.Random(seed)
	items = rows[:]
	rng.shuffle(items)
	n_test = max(1, int(len(items) * test_frac))
	return items[n_test:], items[:n_test]


class RidgeModel:
	def __init__(
		self,
		*,
		num_features: Sequence[str],
		cat_features: Sequence[str],
		cat_maps: Dict[str, Dict[str, int]],
		num_means: Dict[str, float],
		num_stds: Dict[str, float],
		weights: np.ndarray,
		l2: float,
	):
		self.num_features = list(num_features)
		self.cat_features = list(cat_features)
		self.cat_maps = cat_maps
		self.num_means = num_means
		self.num_stds = num_stds
		self.weights = weights
		self.l2 = float(l2)

	def _transform_one(self, row: Dict[str, object]) -> np.ndarray:
		parts: List[float] = [1.0]

		for name in self.num_features:
			v = _as_float(row.get(name))
			mu = self.num_means.get(name, 0.0)
			sd = self.num_stds.get(name, 1.0) or 1.0
			z = 0.0 if v is None else (float(v) - mu) / sd
			parts.append(z)
			parts.append(z * z)

		for name in self.cat_features:
			val = str(row.get(name) or "Unknown")
			m = self.cat_maps.get(name, {})
			k = len(m)
			vec = [0.0] * k
			if val in m:
				vec[m[val]] = 1.0
			parts.extend(vec)

		return np.array(parts, dtype=float)

	def predict_many(self, rows: Sequence[Dict[str, object]]) -> np.ndarray:
		X = np.vstack([self._transform_one(r) for r in rows])
		return X @ self.weights

	def expanded_linear_expression(self) -> Tuple[float, List[Tuple[str, float]]]:
		terms: List[Tuple[str, float]] = []
		idx = 0
		intercept = float(self.weights[idx])
		idx += 1
		for name in self.num_features:
			terms.append((f"z({name})", float(self.weights[idx])))
			idx += 1
			terms.append((f"z({name})^2", float(self.weights[idx])))
			idx += 1
		for name in self.cat_features:
			m = self.cat_maps.get(name, {})
			inv = {v: k for k, v in m.items()}
			for j in range(len(m)):
				val = inv.get(j, "?")
				terms.append((f"I({name}={val})", float(self.weights[idx])))
				idx += 1
		return intercept, terms


def train_ridge_model(
	*,
	rows: List[Dict[str, object]],
	label: str,
	seed: int,
	l2: float,
	test_frac: float,
):
	train_rows, test_rows = _train_test_split(rows, seed=seed, test_frac=test_frac)
	if not train_rows or not test_rows:
		raise ValueError("Not enough rows to train/test")

	num_features = ["age", "risk_tolerance", "sessions_year", "bets_per_session", "avg_bet_size"]
	cat_features = ["sex", "age_group"]

	num_means: Dict[str, float] = {}
	num_stds: Dict[str, float] = {}
	for name in num_features:
		vals = np.array([_as_float(r.get(name)) for r in train_rows if _as_float(r.get(name)) is not None], dtype=float)
		if vals.size == 0:
			num_means[name] = 0.0
			num_stds[name] = 1.0
		else:
			num_means[name] = float(np.mean(vals))
			num_stds[name] = float(np.std(vals)) if float(np.std(vals)) > 1e-9 else 1.0

	cat_maps: Dict[str, Dict[str, int]] = {}
	for name in cat_features:
		counts: Dict[str, int] = {}
		for r in train_rows:
			val = str(r.get(name) or "Unknown")
			counts[val] = counts.get(val, 0) + 1
		top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:64]
		cat_maps[name] = {k: i for i, (k, _) in enumerate(top)}

	dummy = RidgeModel(
		num_features=num_features,
		cat_features=cat_features,
		cat_maps=cat_maps,
		num_means=num_means,
		num_stds=num_stds,
		weights=np.zeros(1),
		l2=l2,
	)
	X_train = np.vstack([dummy._transform_one(r) for r in train_rows])
	y_train = np.array([_as_float(r.get(label)) or 0.0 for r in train_rows], dtype=float)

	n_features = int(X_train.shape[1])
	XtX = X_train.T @ X_train
	reg = np.eye(n_features)
	reg[0, 0] = 0.0
	A = XtX + float(l2) * reg
	b = X_train.T @ y_train
	w = np.linalg.solve(A, b)

	model = RidgeModel(
		num_features=num_features,
		cat_features=cat_features,
		cat_maps=cat_maps,
		num_means=num_means,
		num_stds=num_stds,
		weights=w,
		l2=l2,
	)

	pred = model.predict_many(test_rows)
	y_test = np.array([_as_float(r.get(label)) or 0.0 for r in test_rows], dtype=float)
	mae = float(np.mean(np.abs(pred - y_test)))
	rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
	var = float(np.var(y_test))
	r2 = float(1.0 - (np.mean((pred - y_test) ** 2) / var)) if var > 1e-9 else float("nan")

	metrics: Dict[str, object] = {
		"mae": mae,
		"rmse": rmse,
		"r2": r2,
		"n_train": len(train_rows),
		"n_test": len(test_rows),
		"_test_rows": test_rows,
		"_test_y": y_test,
		"_test_pred": pred,
	}
	return model, metrics


def save_model_artifacts(model: RidgeModel, out_dir: str, prefix: str) -> None:
	os.makedirs(out_dir, exist_ok=True)
	joblib = _try_import_joblib()
	artifact_path = os.path.join(out_dir, f"{prefix}_model_artifact.joblib" if joblib else f"{prefix}_model_artifact.json")
	if joblib:
		joblib.dump(model, artifact_path)
	else:
		payload = {
			"num_features": model.num_features,
			"cat_features": model.cat_features,
			"cat_maps": model.cat_maps,
			"num_means": model.num_means,
			"num_stds": model.num_stds,
			"weights": model.weights.tolist(),
			"l2": model.l2,
		}
		with open(artifact_path, "w") as f:
			json.dump(payload, f, indent=2)

	coef_csv = os.path.join(out_dir, f"{prefix}_model_coefficients.csv")
	b0, terms = model.expanded_linear_expression()
	with open(coef_csv, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["term", "coefficient"])
		w.writerow(["intercept", f"{b0:.10g}"])
		for name, coef in terms:
			w.writerow([name, f"{coef:.10g}"])

	expr_path = os.path.join(out_dir, f"{prefix}_model_expression.txt")
	with open(expr_path, "w") as f:
		f.write("AnnualNetGamblingOutcome_hat = intercept")
		for name, coef in terms:
			sign = "+" if coef >= 0 else "-"
			f.write(f" {sign} {abs(coef):.6g}*{name}")
		f.write("\n\n")
		f.write("z(x) indicates standardized numeric features: (x - mean)/std from training.\n")
		f.write("I(cat=value) is an indicator for one-hot features.\n")

	print(f"Saved model artifact: {artifact_path}")
	print(f"Saved coefficients CSV: {coef_csv}")
	print(f"Saved expression: {expr_path}")


def save_test_predictions(out_dir: str, prefix: str, test_rows: Sequence[Dict[str, object]], y: np.ndarray, pred: np.ndarray) -> str:
	os.makedirs(out_dir, exist_ok=True)
	path = os.path.join(out_dir, f"{prefix}_test_predictions.csv")
	fields = [
		"sex",
		"age_group",
		"age",
		"risk_tolerance",
		"p_sports_4w",
		"is_sports_gambler_4w",
		"sessions_year",
		"bets_per_session",
		"avg_bet_size",
		"total_wager",
		"expected_net",
		"pred_expected_net",
		"error",
	]
	with open(path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for r, yy, pp in zip(test_rows, y.tolist(), pred.tolist()):
			row = {k: r.get(k) for k in fields if k in r}
			row["expected_net"] = float(yy)
			row["pred_expected_net"] = float(pp)
			row["error"] = float(pp - yy)
			w.writerow(row)
	print(f"Saved test predictions: {path}")
	return path


def save_demographic_summary(df: pd.DataFrame, out_dir: str, prefix: str) -> str:
	os.makedirs(out_dir, exist_ok=True)
	path = os.path.join(out_dir, f"{prefix}_demographic_summary.csv")
	grouped = (
		df.groupby(["sex", "age_group"], dropna=False)
		.agg(
			n=("expected_net", "size"),
			expected_mean=("expected_net", "mean"),
			expected_median=("expected_net", "median"),
			realized_mean=("realized_net", "mean"),
			realized_median=("realized_net", "median"),
			avg_wager=("total_wager", "mean"),
			p_gambler_4w=("is_sports_gambler_4w", "mean"),
		)
		.reset_index()
	)
	grouped.to_csv(path, index=False)
	print(f"Saved demographic summary: {path}")
	return path


def profile_distribution_report(
	*,
	seed: int,
	assumptions: Assumptions,
	online_p: Dict[Tuple[str, str], float],
	age_groups: Sequence[str],
	sexes: Sequence[str],
	n_sims: int=4000,
	lose_threshold: float=1000.0,
) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	rows: List[Dict[str, object]] = []
	for sex in sexes:
		for age_group in age_groups:
			p_online = float(online_p.get((sex, age_group), 0.0))
			for risk_level, risk_override in [("low", 0.15), ("mid", 0.45), ("high", 0.80)]:
				nets = []
				exp_nets = []
				wagers = []
				for _ in range(int(n_sims)):
					risk = _clip(float(rng.normal(loc=risk_override, scale=0.07)), 0.0, 1.0)
					sim = simulate_individual_year(
						rng,
						sex=sex,
						age_group=age_group,
						p_online_4w_ex_lottery=p_online,
						assumptions=assumptions,
					)
					sim["risk_tolerance"] = risk
					if sim["is_sports_gambler_4w"] <= 0.0:
						nets.append(0.0)
						exp_nets.append(0.0)
						wagers.append(0.0)
						continue
					sessions_year = float(sim["sessions_year"])
					bps = float(sim["bets_per_session"])
					median = assumptions.base_bet_median * _sex_multiplier(sex) * _age_multiplier(float(sim["age"]))
					median *= 0.65 + 1.10 * risk
					mu = math.log(max(0.5, median))
					sigma = 0.65 + 0.55 * risk
					avg_bet_size = float(rng.lognormal(mean=mu, sigma=sigma))
					avg_bet_size = _clip(avg_bet_size, 1.0, 500.0)
					total_wager = sessions_year * bps * avg_bet_size
					expected = -assumptions.house_edge * total_wager
					stdev = assumptions.base_sigma_per_sqrt_wager * math.sqrt(max(0.0, total_wager)) * (0.7 + 1.8 * risk)
					realized = expected + float(rng.standard_t(df=assumptions.t_df)) * stdev
					realized = max(realized, -total_wager)
					nets.append(float(realized))
					exp_nets.append(float(expected))
					wagers.append(float(total_wager))

				nets_arr = np.array(nets, dtype=float)
				exp_arr = np.array(exp_nets, dtype=float)
				w_arr = np.array(wagers, dtype=float)

				rows.append(
					{
						"sex": sex,
						"age_group": age_group,
						"risk_level": risk_level,
						"p_sports_4w": float(_clip(p_online * assumptions.sports_share_of_online, 0.0, 1.0)),
						"expected_net_mean": float(np.mean(exp_arr)),
						"realized_net_p05": float(np.quantile(nets_arr, 0.05)),
						"realized_net_p50": float(np.quantile(nets_arr, 0.50)),
						"realized_net_p95": float(np.quantile(nets_arr, 0.95)),
						"prob_profit": float(np.mean(nets_arr > 0.0)),
						"prob_lose_gt_threshold": float(np.mean(nets_arr < -abs(float(lose_threshold)))),
						"avg_wager": float(np.mean(w_arr)),
					}
				)

	return pd.DataFrame(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Model annual net online sports-gambling outcome from demographics + assumptions")
	parser.add_argument("--excel", default="data/GSGB_W3_2025_FINAL.xlsx", help="Path to GSGB Wave 3 Excel tables")
	parser.add_argument("--out", default="outputs", help="Output directory")
	parser.add_argument("--sample", type=int, default=50000, help="Number of synthetic individuals to simulate")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--country", choices=["uk", "us"], default="uk")
	parser.add_argument("--house-edge", type=float, default=None, help="Override sportsbook hold (e.g. 0.07)")
	parser.add_argument(
		"--sports-share-of-online",
		type=float,
		default=None,
		help="Fraction of online gamblers who are sports bettors (defaults derived from Table A.13 / A.9)",
	)
	parser.add_argument("--l2", type=float, default=25.0, help="Ridge regularization")
	parser.add_argument("--test-frac", type=float, default=0.20)
	parser.add_argument("--profile-sims", type=int, default=4000, help="Monte Carlo sims per profile for distribution report")
	parser.add_argument("--lose-threshold", type=float, default=1000.0)
	parser.add_argument("--prefix", default="q2", help="Filename prefix for artifacts")
	args = parser.parse_args(list(argv) if argv is not None else None)

	tables = load_gsgb_tables(args.excel)
	pop_weights = parse_population_weights(tables["a3"])
	online_p = parse_online_participation_by_sex_age(tables["a9"])

	if args.sports_share_of_online is None:
		sports_share = derive_default_sports_share_of_online(tables["a9"], tables["a13"])
	else:
		sports_share = float(args.sports_share_of_online)

	assumptions = default_assumptions(args.country, sports_share_of_online=sports_share)
	if args.house_edge is not None:
		assumptions = Assumptions(
			country=assumptions.country,
			house_edge=float(args.house_edge),
			sports_share_of_online=assumptions.sports_share_of_online,
			risk_alpha=assumptions.risk_alpha,
			risk_beta=assumptions.risk_beta,
			base_bet_median=assumptions.base_bet_median,
			base_sigma_per_sqrt_wager=assumptions.base_sigma_per_sqrt_wager,
			t_df=assumptions.t_df,
		)

	print("Assumptions:")
	print(f"  country={assumptions.country}  house_edge={assumptions.house_edge:.3f}  sports_share_of_online={assumptions.sports_share_of_online:.3f}")

	df = simulate_population(
		n=int(args.sample),
		seed=int(args.seed),
		pop_weights=pop_weights,
		online_p=online_p,
		assumptions=assumptions,
	)

	rows = df.to_dict(orient="records")
	model, metrics = train_ridge_model(rows=rows, label="expected_net", seed=int(args.seed), l2=float(args.l2), test_frac=float(args.test_frac))
	print(f"Ridge holdout: MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}  n_train={metrics['n_train']} n_test={metrics['n_test']}")

	save_model_artifacts(model, args.out, prefix=args.prefix)
	_test_rows = metrics["_test_rows"]
	_test_y = metrics["_test_y"]
	_test_pred = metrics["_test_pred"]
	save_test_predictions(args.out, args.prefix, _test_rows, _test_y, _test_pred)
	save_demographic_summary(df, args.out, args.prefix)

	age_groups = sorted({a for _, a, _ in pop_weights}, key=lambda s: (999 if s.endswith("+") else int(s.split("-")[0])))
	prof = profile_distribution_report(
		seed=int(args.seed) + 7,
		assumptions=assumptions,
		online_p=online_p,
		age_groups=age_groups,
		n_sims=int(args.profile_sims),
		lose_threshold=float(args.lose_threshold),
	)
	prof_path = os.path.join(args.out, f"{args.prefix}_profile_report.csv")
	os.makedirs(args.out, exist_ok=True)
	prof.to_csv(prof_path, index=False)
	print(f"Saved profile report: {prof_path}")

	avg_expected = float(df["expected_net"].mean())
	avg_realized = float(df["realized_net"].mean())
	p_active = float(df["is_sports_gambler_4w"].mean())
	print(f"Population average expected annual net: {avg_expected:.2f}")
	print(f"Population average realized annual net: {avg_realized:.2f}")
	print(f"P(online sports bettor in last 4w): {p_active:.3f}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

