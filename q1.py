from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _try_import_matplotlib():
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		return plt
	except Exception:
		return None


def _try_import_tabulate():
	try:
		from tabulate import tabulate
		return tabulate
	except Exception:
		return None


def _try_import_joblib():
	try:
		import joblib
		return joblib
	except Exception:
		return None


def _to_int(value: Optional[str]):
	if value is None:
		return None
	s = str(value).strip()
	if s == "" or s.lower() in {"na", "nan", "null"}:
		return None
	try:
		return int(float(s))
	except Exception:
		return None


def _to_float(value: Optional[str]):
	if value is None:
		return None
	s = str(value).strip()
	if s == "" or s.lower() in {"na", "nan", "null"}:
		return None
	try:
		return float(s)
	except Exception:
		return None


def _clip(v: float, lo: float, hi: float) -> float:
	return float(min(max(v, lo), hi))


def _sigmoid(x: float) -> float:
	return 1.0 / (1.0 + math.exp(-x))


def _safe_get(row: Dict[str, str], name: str) -> Optional[str]:
	return row.get(name) or row.get(name.lower()) or row.get(name.upper())

NO_STATE_INCOME_TAX_FIPS = {
	2,   # AK
	12,  # FL
	32,  # NV
	46,  # SD
	48,  # TX
	53,  # WA
	56,  # WY
}


STATE_TAX_RATE_BY_FIPS = {
	# rough effective state income tax rates (not marginal).
	6: 0.055,   # CA
	36: 0.060,  # NY
	34: 0.050,  # NJ
	17: 0.045,  # IL
	25: 0.045,  # MA
	26: 0.042,  # MI
	27: 0.045,  # MN
	39: 0.040,  # OH
	42: 0.045,  # PA
	8: 0.042,   # CO
	41: 0.045,  # OR
	9: 0.050,   # CT
	24: 0.050,  # MD
	51: 0.050,  # VA
	37: 0.050,  # NC
	47: 0.040,  # TN
	40: 0.040,  # OK
	29: 0.040,  # MO
	20: 0.040,  # KS
	19: 0.035,  # IA
	55: 0.045,  # WI
	18: 0.035,  # IN
	1: 0.040,   # AL
	4: 0.030,   # AZ
	5: 0.050,   # AR
	16: 0.045,  # ID
	21: 0.045,  # KY
	22: 0.030,  # LA
	28: 0.030,  # MS
	30: 0.035,  # MT
	31: 0.030,  # NE
	33: 0.000,  # NH
	35: 0.045,  # NM
	44: 0.038,  # RI
	45: 0.030,  # SC
	50: 0.045,  # VT
	54: 0.040,  # WV
	10: 0.045,  # DE
	11: 0.060,  # DC
	23: 0.040,  # ME
	15: 0.040,  # HI
	38: 0.030,  # ND
	49: 0.045,  # UT
}


HOUSING_MULTIPLIER_BY_STATE_FIPS = {
	6: 1.65,   # CA
	36: 1.70,  # NY
	34: 1.45,  # NJ
	25: 1.35,  # MA
	53: 1.30,  # WA
	41: 1.25,  # OR
	24: 1.25,  # MD
	9: 1.30,   # CT
	15: 1.40,  # HI
	8: 1.10,   # CO
	48: 1.05,  # TX
	12: 1.05,  # FL
	13: 1.00,  # GA
	17: 1.00,  # IL
	4: 1.00,   # AZ
	32: 1.10,  # NV
}


REGION_COST_MULTIPLIER = {
	1: 1.15,  # Northeast
	2: 0.98,  # Midwest
	3: 0.95,  # South
	4: 1.08,  # West
}


METRO_STATUS_MULTIPLIER = {
	# gtmetsta: 1/2 etc vary by CPS; treat 1 as metro, else non-metro.
	1: 1.10,
	2: 0.95,
	3: 1.05,
	4: 0.95,
}


CBSA_SIZE_MULTIPLIER = {
	# gtcbsasz: metro area size category; larger -> higher costs.
	1: 0.95,
	2: 1.00,
	3: 1.05,
	4: 1.10,
	5: 1.15,
	6: 1.20,
	7: 1.28,
}


def gross_annual_income(row: Dict[str, str]) -> Optional[float]:
	weekly = _to_float(_safe_get(row, "pternwa"))
	if weekly is not None and weekly > 0:
		return weekly * 52.0

	bracket = _to_int(_safe_get(row, "hefaminc"))
	if bracket is None:
		return None

	bracket_midpoints = {
		1: 2500,
		2: 7500,
		3: 12500,
		4: 17500,
		5: 22500,
		6: 27500,
		7: 32500,
		8: 37500,
		9: 45000,
		10: 55000,
		11: 67500,
		12: 82500,
		13: 100000,
		14: 125000,
		15: 175000,
		16: 250000,
	}
	if bracket not in bracket_midpoints:
		return None

	return float(bracket_midpoints[bracket]) * 0.60


def _num_children(row: Dict[str, str]) -> int:
	n = _to_int(_safe_get(row, "prnmchld"))
	if n is None or n < 0:
		return 0
	return int(_clip(float(n), 0, 10))


def _household_size(row: Dict[str, str]) -> int:
	n = _to_int(_safe_get(row, "hrnumhou"))
	if n is None or n <= 0:
		n = 1 + _num_children(row)
	return int(_clip(float(n), 1, 12))


def effective_federal_tax(income: float, marital_code: Optional[int], kids: int) -> float:

	if income <= 0:
		return 0.0

	is_married = marital_code in {1, 2}
	deduction = 30000.0 if is_married else 15000.0
	taxable = max(0.0, income - deduction)

	if is_married:
		brackets = [
			(22000.0, 0.10),
			(89450.0, 0.12),
			(190750.0, 0.22),
			(364200.0, 0.24),
			(462500.0, 0.32),
			(693750.0, 0.35),
			(float("inf"), 0.37),
		]
	else:
		brackets = [
			(11000.0, 0.10),
			(44725.0, 0.12),
			(95375.0, 0.22),
			(182100.0, 0.24),
			(231250.0, 0.32),
			(578125.0, 0.35),
			(float("inf"), 0.37),
		]

	tax = 0.0
	prev = 0.0
	remaining = taxable
	for cap, rate in brackets:
		if remaining <= 0:
			break
		band = min(remaining, cap - prev)
		tax += band * rate
		remaining -= band
		prev = cap

	ctc = 2000.0 * max(0, kids)
	phaseout_start = 200000.0 if not is_married else 400000.0
	if income > phaseout_start:
		reduction = 50.0 * ((income - phaseout_start) / 1000.0)
		ctc = max(0.0, ctc - reduction)

	return max(0.0, tax - ctc)


def effective_state_tax(income: float, gestfips: Optional[int]) -> float:
	if income <= 0:
		return 0.0
	if gestfips in NO_STATE_INCOME_TAX_FIPS:
		return 0.0
	rate = STATE_TAX_RATE_BY_FIPS.get(gestfips, 0.04)
	if income > 150000:
		rate += 0.01
	elif income > 75000:
		rate += 0.005
	rate = _clip(rate, 0.0, 0.10)
	return income * rate


def housing_cost_annual(
	row: Dict[str, str],
	gestfips: Optional[int],
	region_code: Optional[int],
	household_size: int,
) -> float:
	base_monthly = 950.0
	state_mult = HOUSING_MULTIPLIER_BY_STATE_FIPS.get(gestfips, 1.0)
	region_mult = REGION_COST_MULTIPLIER.get(region_code, 1.0)

	metro = _to_int(_safe_get(row, "gtmetsta"))
	cbsa_size = _to_int(_safe_get(row, "gtcbsasz"))
	metro_mult = METRO_STATUS_MULTIPLIER.get(metro, 1.0)
	cbsa_mult = CBSA_SIZE_MULTIPLIER.get(cbsa_size, 1.0)

	size_mult = 1.0 + 0.25 * max(0, household_size - 1)
	size_mult = _clip(size_mult, 1.0, 2.25)

	monthly = base_monthly * state_mult * region_mult * metro_mult * cbsa_mult * size_mult
	return monthly * 12.0


def food_cost_annual(region_code: Optional[int], household_size: int, kids: int) -> float:
	region_mult = REGION_COST_MULTIPLIER.get(region_code, 1.0)
	adults = max(1, household_size - kids)
	monthly = adults * 360.0 + kids * 260.0
	return monthly * 12.0 * region_mult


def healthcare_cost_annual(row: Dict[str, str], age: int) -> float:
	if age < 18:
		base = 1200.0
	elif age < 35:
		base = 2200.0
	elif age < 50:
		base = 3200.0
	elif age < 65:
		base = 4200.0
	else:
		base = 5200.0

	prdisflg = _to_int(_safe_get(row, "prdisflg"))
	pedisphy = _to_int(_safe_get(row, "pedisphy"))
	pedisrem = _to_int(_safe_get(row, "pedisrem"))
	disability = any(v == 1 for v in [prdisflg, pedisphy, pedisrem])
	if disability:
		base += 2200.0

	peerncov = _to_int(_safe_get(row, "peerncov"))
	if peerncov == 1:
		base -= 450.0

	peio1cow = _to_int(_safe_get(row, "peio1cow"))
	if peio1cow in {1, 2}:
		base -= 250.0

	peafever = _to_int(_safe_get(row, "peafever"))
	if peafever == 1:
		base -= 200.0

	return max(600.0, base)


def childcare_cost_annual(row: Dict[str, str], kids: int) -> float:
	if kids <= 0:
		return 0.0

	prchld = _to_int(_safe_get(row, "prchld"))
	if prchld in {1, 2}:
		per_child = 12000.0
	elif prchld in {3, 4}:
		per_child = 9000.0
	elif prchld in {5, 6, 7}:
		per_child = 4500.0
	elif prchld is None:
		per_child = 4500.0
	else:
		per_child = 2000.0

	total = per_child * (1.0 + 0.75 * max(0, kids - 1))

	pemaritl = _to_int(_safe_get(row, "pemaritl"))
	if pemaritl in {1, 2}:
		total *= 0.90

	return total


def transportation_cost_annual(row: Dict[str, str]) -> float:
	base = 4200.0
	metro = _to_int(_safe_get(row, "gtmetsta"))
	cbsa_size = _to_int(_safe_get(row, "gtcbsasz"))
	if metro == 1:
		base += 300.0
	if cbsa_size is not None:
		base += 120.0 * max(0, cbsa_size - 2)

	occ = _to_int(_safe_get(row, "prdtocc1"))
	if occ is not None:
		if occ >= 6000:
			base += 600.0
		elif occ >= 4000:
			base += 350.0

	return max(1800.0, base)


def tuition_or_student_loan_annual(row: Dict[str, str], age: int) -> float:
	enrolled = _to_int(_safe_get(row, "peschenr"))
	if enrolled != 1:
		return 0.0

	full_time = _to_int(_safe_get(row, "peschft"))
	base = 9800.0 if full_time == 1 else 4000.0

	if age >= 30:
		base *= 0.70
	elif age < 22:
		base *= 1.10

	edu = _to_int(_safe_get(row, "peeduca"))
	if edu is not None and edu >= 44:
		base *= 1.10

	return base


def compute_disposable_income(row: Dict[str, str]) -> Optional[Dict[str, float]]:
	income = gross_annual_income(row)
	if income is None or not (income > 0):
		return None

	age = _to_int(_safe_get(row, "prtage"))
	if age is None or age <= 0 or age > 95:
		return None

	gestfips = _to_int(_safe_get(row, "gestfips"))
	region = _to_int(_safe_get(row, "gereg"))
	pemaritl = _to_int(_safe_get(row, "pemaritl"))
	kids = _num_children(row)
	household_size = _household_size(row)

	fed_tax = effective_federal_tax(income, pemaritl, kids)
	state_tax = effective_state_tax(income, gestfips)
	housing = housing_cost_annual(row, gestfips, region, household_size)
	food = food_cost_annual(region, household_size, kids)
	healthcare = healthcare_cost_annual(row, age)
	childcare = childcare_cost_annual(row, kids)
	transport = transportation_cost_annual(row)
	tuition = tuition_or_student_loan_annual(row, age)

	prsjmj = _to_int(_safe_get(row, "prsjmj"))
	peernuot = _to_int(_safe_get(row, "peernuot"))
	income_adj = 1.0
	if prsjmj in {2, 3}:
		income_adj += 0.03
	if peernuot == 1:
		income_adj += 0.02
	income_adj = _clip(income_adj, 0.95, 1.08)

	gross = income * income_adj
	taxes = fed_tax + state_tax
	expenses = housing + food + healthcare + childcare + transport + tuition
	disposable = gross - taxes - expenses

	return {
		"gross": float(gross),
		"taxes": float(taxes),
		"housing": float(housing),
		"food": float(food),
		"healthcare": float(healthcare),
		"childcare": float(childcare),
		"transport": float(transport),
		"tuition": float(tuition),
		"disposable": float(max(0.0, disposable)),
	}

CORE_INPUTS = [
	"annual_salary",
	"prtage",
	"pesex",
	"pemaritl",
	"peeduca",
	"ptdtrace",
	"pehspnon",
]


@dataclass
class RidgeModel:
	feature_names: List[str]
	cat_maps: Dict[str, Dict[int, int]]
	num_means: Dict[str, float]
	num_stds: Dict[str, float]
	weights: np.ndarray
	l2: float

	def predict_one(self, features: Dict[str, Optional[float]]) -> float:
		x = self._transform_one(features)
		return float(x @ self.weights)

	def predict_many(self, rows: Sequence[Dict[str, Optional[float]]]) -> np.ndarray:
		X = np.vstack([self._transform_one(r) for r in rows])
		return X @ self.weights

	def _transform_one(self, f: Dict[str, Optional[float]]) -> np.ndarray:
		vec: List[float] = [1.0]

		for name in ["annual_salary", "prtage"]:
			v = f.get(name)
			if v is None:
				v = self.num_means[name]
			mu = self.num_means[name]
			sd = self.num_stds[name] if self.num_stds[name] > 1e-6 else 1.0
			vec.append((float(v) - mu) / sd)

		age = f.get("prtage")
		if age is None:
			age = self.num_means["prtage"]
		age2 = (float(age) ** 2)
		mu2 = self.num_means["prtage"] ** 2
		sd2 = max(1.0, (self.num_stds["prtage"] ** 2))
		vec.append((age2 - mu2) / sd2)

		for name in ["pesex", "pemaritl", "peeduca", "ptdtrace", "pehspnon"]:
			mapping = self.cat_maps[name]
			dim = len(mapping)
			onehot = [0.0] * dim
			raw = f.get(name)
			if raw is None:
				vec.extend(onehot)
				continue
			key = int(raw)
			idx = mapping.get(key)
			if idx is not None:
				onehot[idx] = 1.0
			vec.extend(onehot)

		return np.array(vec, dtype=float)


	def expanded_linear_expression(self) -> Tuple[float, List[Tuple[str, float]]]:
		w = self.weights
		w0 = float(w[0])

		salary_mu = self.num_means["annual_salary"]
		salary_sd = self.num_stds["annual_salary"] if self.num_stds["annual_salary"] > 1e-6 else 1.0
		age_mu = self.num_means["prtage"]
		age_sd = self.num_stds["prtage"] if self.num_stds["prtage"] > 1e-6 else 1.0

		mu2 = age_mu ** 2
		sd2 = max(1.0, (age_sd ** 2))

		b_salary = float(w[1]) / float(salary_sd)
		b_age = float(w[2]) / float(age_sd)
		b_age2 = float(w[3]) / float(sd2)

		b0 = w0 - (float(w[1]) * salary_mu / salary_sd) - (float(w[2]) * age_mu / age_sd) - (float(w[3]) * mu2 / sd2)

		terms: List[Tuple[str, float]] = [
			("annual_salary", b_salary),
			("age", b_age),
			("age^2", b_age2),
		]

		offset = 4
		for name in ["pesex", "pemaritl", "peeduca", "ptdtrace", "pehspnon"]:
			mapping = self.cat_maps[name]
			inv = [None] * len(mapping)
			for code, idx in mapping.items():
				if 0 <= idx < len(inv):
					inv[idx] = code
			for idx, code in enumerate(inv):
				if code is None:
					continue
				coef = float(w[offset + idx])
				terms.append((f"I({name}={code})", coef))
			offset += len(mapping)

		return float(b0), terms


def _diverse_holdout_split(
	items: List[Tuple[Dict[str, Optional[float]], float]],
	*,
	seed: int,
	test_min: int,
	test_frac: float,
) -> Tuple[List[Tuple[Dict[str, Optional[float]], float]], List[Tuple[Dict[str, Optional[float]], float]]]:
	rng = random.Random(seed)
	items = items[:]
	rng.shuffle(items)

	n = len(items)
	if n == 0:
		return [], []

	test_size = max(int(n * test_frac), int(test_min))
	test_size = min(test_size, max(1, n // 2))

	group_to_indices: Dict[Tuple[str, object, object, object], List[int]] = {}
	for i, (x, _) in enumerate(items):
		age_bin = _bucket_age(x.get("prtage"))
		sex = int(x["pesex"]) if x.get("pesex") is not None else "U"
		race = int(x["ptdtrace"]) if x.get("ptdtrace") is not None else "U"
		mar = int(x["pemaritl"]) if x.get("pemaritl") is not None else "U"
		key = (age_bin, sex, race, mar)
		group_to_indices.setdefault(key, []).append(i)

	for idxs in group_to_indices.values():
		rng.shuffle(idxs)

	selected: List[int] = []
	for key, idxs in group_to_indices.items():
		if len(selected) >= test_size:
			break
		if idxs:
			selected.append(idxs.pop())

	if len(selected) < test_size:
		pool = [i for i in range(n) if i not in set(selected)]
		rng.shuffle(pool)
		need = test_size - len(selected)
		selected.extend(pool[:need])

	selected_set = set(selected)
	test = [items[i] for i in range(n) if i in selected_set]
	train = [items[i] for i in range(n) if i not in selected_set]
	return train, test


def _train_test_split(items: List[Tuple[Dict[str, Optional[float]], float]], seed: int, test_frac: float = 0.2):
	rng = random.Random(seed)
	items = items[:]
	rng.shuffle(items)
	n_test = int(len(items) * test_frac)
	test = items[:n_test]
	train = items[n_test:]
	return train, test


def _fit_ridge(
	feature_dicts: Sequence[Dict[str, Optional[float]]],
	y: np.ndarray,
	l2: float,
	cat_maps: Dict[str, Dict[int, int]],
	num_means: Dict[str, float],
	num_stds: Dict[str, float],
) -> np.ndarray:
	dummy_model = RidgeModel(
		feature_names=[],
		cat_maps=cat_maps,
		num_means=num_means,
		num_stds=num_stds,
		weights=np.zeros(1),
		l2=l2,
	)
	X = np.vstack([dummy_model._transform_one(f) for f in feature_dicts])
	n_features = X.shape[1]

	XtX = X.T @ X
	reg = np.eye(n_features)
	reg[0, 0] = 0.0
	A = XtX + l2 * reg
	b = X.T @ y
	w = np.linalg.solve(A, b)
	return w


def train_model(
	samples: List[Tuple[Dict[str, Optional[float]], float]],
	*,
	seed: int = 42,
	l2: float = 25.0,
	test_min: int = 1000,
	test_frac: float = 0.2,
) -> Tuple[RidgeModel, Dict[str, object]]:
	train, test = _diverse_holdout_split(samples, seed=seed, test_min=test_min, test_frac=test_frac)
	train_x = [x for x, _ in train]
	train_y = np.array([yy for _, yy in train], dtype=float)
	test_x = [x for x, _ in test]
	test_y = np.array([yy for _, yy in test], dtype=float)

	salary_vals = np.array([x["annual_salary"] for x in train_x if x.get("annual_salary") is not None], dtype=float)
	age_vals = np.array([x["prtage"] for x in train_x if x.get("prtage") is not None], dtype=float)
	num_means = {
		"annual_salary": float(np.mean(salary_vals)) if salary_vals.size else 60000.0,
		"prtage": float(np.mean(age_vals)) if age_vals.size else 40.0,
	}
	num_stds = {
		"annual_salary": float(np.std(salary_vals)) if salary_vals.size else 20000.0,
		"prtage": float(np.std(age_vals)) if age_vals.size else 12.0,
	}

	cat_maps: Dict[str, Dict[int, int]] = {}
	for name in ["pesex", "pemaritl", "peeduca", "ptdtrace", "pehspnon"]:
		counts: Dict[int, int] = {}
		for x in train_x:
			v = x.get(name)
			if v is None:
				continue
			key = int(v)
			counts[key] = counts.get(key, 0) + 1

		top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:32]
		cat_maps[name] = {k: i for i, (k, _) in enumerate(top)}

	w = _fit_ridge(train_x, train_y, l2=l2, cat_maps=cat_maps, num_means=num_means, num_stds=num_stds)
	model = RidgeModel(
		feature_names=CORE_INPUTS,
		cat_maps=cat_maps,
		num_means=num_means,
		num_stds=num_stds,
		weights=w,
		l2=l2,
	)

	pred = model.predict_many(test_x)
	mae = float(np.mean(np.abs(pred - test_y)))
	rmse = float(np.sqrt(np.mean((pred - test_y) ** 2)))
	var = float(np.var(test_y))
	r2 = float(1.0 - (np.mean((pred - test_y) ** 2) / var)) if var > 1e-9 else float("nan")

	metrics: Dict[str, object] = {"mae": mae, "rmse": rmse, "r2": r2, "n_train": len(train), "n_test": len(test)}
	metrics["_test_y"] = test_y
	metrics["_test_pred"] = pred
	metrics["_test_x"] = test_x
	return model, metrics


def save_model_artifacts(model: RidgeModel, out_dir: str):
	os.makedirs(out_dir, exist_ok=True)

	joblib = _try_import_joblib()
	artifact_path = os.path.join(out_dir, "model_artifact.joblib" if joblib else "model_artifact.json")
	if joblib:
		joblib.dump(model, artifact_path)
	else:
		payload = {
			"feature_names": model.feature_names,
			"cat_maps": model.cat_maps,
			"num_means": model.num_means,
			"num_stds": model.num_stds,
			"weights": model.weights.tolist(),
			"l2": model.l2,
		}
		with open(artifact_path, "w") as f:
			json.dump(payload, f, indent=2)

	b0, terms = model.expanded_linear_expression()
	coef_csv = os.path.join(out_dir, "model_coefficients.csv")
	with open(coef_csv, "w", newline="") as f:
		w = csv.writer(f)
		w.writerow(["term", "coefficient"])
		w.writerow(["intercept", f"{b0:.10g}"])
		for name, coef in terms:
			w.writerow([name, f"{coef:.10g}"])

	expr_path = os.path.join(out_dir, "model_expression.txt")
	with open(expr_path, "w") as f:
		f.write("DisposableIncome_hat = intercept")
		for name, coef in terms:
			sign = "+" if coef >= 0 else "-"
			f.write(f" {sign} {abs(coef):.6g}*{name}")
		f.write("\n\n")
		f.write("Where I(var=code) is an indicator (1 if true else 0).\n")
		f.write("age^2 means (age in years)^2. annual_salary is gross annual dollars.\n")

	print(f"Saved model artifact: {artifact_path}")
	print(f"Saved coefficients CSV: {coef_csv}")
	print(f"Saved expression: {expr_path}")


def save_test_predictions(
	out_dir: str,
	test_x: Sequence[Dict[str, Optional[float]]],
	test_y: np.ndarray,
	pred: np.ndarray,
):
	os.makedirs(out_dir, exist_ok=True)
	path = os.path.join(out_dir, "test_predictions.csv")
	fields = [
		"annual_salary",
		"prtage",
		"pesex",
		"pemaritl",
		"peeduca",
		"ptdtrace",
		"pehspnon",
		"actual_disposable",
		"pred_disposable",
		"error",
	]
	with open(path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=fields)
		w.writeheader()
		for x, y, p in zip(test_x, test_y.tolist(), pred.tolist()):
			row = {
				"annual_salary": x.get("annual_salary"),
				"prtage": x.get("prtage"),
				"pesex": x.get("pesex"),
				"pemaritl": x.get("pemaritl"),
				"peeduca": x.get("peeduca"),
				"ptdtrace": x.get("ptdtrace"),
				"pehspnon": x.get("pehspnon"),
				"actual_disposable": float(y),
				"pred_disposable": float(p),
				"error": float(p - y),
			}
			w.writerow(row)
	print(f"Saved test predictions: {path}")
	return path


def _bucket_age(age: Optional[float]) -> str:
	if age is None:
		return "Unknown"
	a = int(age)
	if a < 18:
		return "<18"
	if a <= 24:
		return "18-24"
	if a <= 34:
		return "25-34"
	if a <= 44:
		return "35-44"
	if a <= 54:
		return "45-54"
	if a <= 64:
		return "55-64"
	return "65+"


def _group_mean(values: np.ndarray) -> Tuple[float, float]:
	if values.size == 0:
		return float("nan"), float("nan")
	return float(np.mean(values)), float(np.median(values))


def demographic_group_summary(
	test_x: Sequence[Dict[str, Optional[float]]],
	test_y: np.ndarray,
	pred: np.ndarray,
) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []

	def emit(group_name: str, key_func):
		buckets: Dict[str, List[int]] = {}
		for i, x in enumerate(test_x):
			k = key_func(x)
			buckets.setdefault(str(k), []).append(i)

		for k, idxs in sorted(buckets.items(), key=lambda kv: kv[0]):
			yv = test_y[idxs]
			pv = pred[idxs]
			mean_y, med_y = _group_mean(yv)
			mean_p, med_p = _group_mean(pv)
			rows.append(
				{
					"group": group_name,
					"bucket": k,
					"n": int(len(idxs)),
					"actual_mean": mean_y,
					"actual_median": med_y,
					"pred_mean": mean_p,
					"pred_median": med_p,
				}
			)

	emit("age_bin", lambda x: _bucket_age(x.get("prtage")))
	emit("sex", lambda x: int(x["pesex"]) if x.get("pesex") is not None else "Unknown")
	emit("marital", lambda x: int(x["pemaritl"]) if x.get("pemaritl") is not None else "Unknown")
	emit("education", lambda x: int(x["peeduca"]) if x.get("peeduca") is not None else "Unknown")
	emit("race", lambda x: int(x["ptdtrace"]) if x.get("ptdtrace") is not None else "Unknown")
	emit("hispanic", lambda x: int(x["pehspnon"]) if x.get("pehspnon") is not None else "Unknown")

	return rows


def save_plots(
	out_dir: str,
	test_x: Sequence[Dict[str, Optional[float]]],
	test_y: np.ndarray,
	pred: np.ndarray,
	*,
	max_points: int = 8000,
) -> None:
	plt = _try_import_matplotlib()
	if plt is None:
		print("matplotlib not available; skipping plots.", file=sys.stderr)
		return

	os.makedirs(out_dir, exist_ok=True)

	n = int(test_y.shape[0])
	idx = np.arange(n)
	if n > max_points:
		rng = np.random.default_rng(42)
		idx = rng.choice(idx, size=max_points, replace=False)

	y_s = test_y[idx]
	p_s = pred[idx]
	salary = np.array([float(test_x[i].get("annual_salary") or 0.0) for i in idx], dtype=float)

	fig = plt.figure(figsize=(12, 10))
	gs = fig.add_gridspec(2, 2)

	ax1 = fig.add_subplot(gs[0, 0])
	ax1.scatter(y_s, p_s, s=6, alpha=0.25)
	lo = float(min(np.min(y_s), np.min(p_s)))
	hi = float(max(np.max(y_s), np.max(p_s)))
	ax1.plot([lo, hi], [lo, hi], color="black", linewidth=1)
	ax1.set_title("Predicted vs Actual (holdout)")
	ax1.set_xlabel("Actual disposable income")
	ax1.set_ylabel("Predicted disposable income")

	ax2 = fig.add_subplot(gs[0, 1])
	res = pred - test_y
	res_s = res[idx]
	ax2.hist(res_s, bins=50, alpha=0.85)
	ax2.set_title("Residuals (pred - actual)")
	ax2.set_xlabel("Dollars")
	ax2.set_ylabel("Count")

	ax3 = fig.add_subplot(gs[1, 0])
	ax3.scatter(salary, y_s, s=6, alpha=0.20, label="actual")
	ax3.scatter(salary, p_s, s=6, alpha=0.20, label="pred")
	ax3.set_title("Disposable vs Gross Salary")
	ax3.set_xlabel("Annual salary (gross)")
	ax3.set_ylabel("Disposable income")
	ax3.legend(loc="best")

	ax4 = fig.add_subplot(gs[1, 1])
	age_bins = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
	actual_means = []
	pred_means = []
	for b in age_bins:
		mask = np.array([_bucket_age(x.get("prtage")) == b for x in test_x], dtype=bool)
		actual_means.append(float(np.mean(test_y[mask])) if np.any(mask) else float("nan"))
		pred_means.append(float(np.mean(pred[mask])) if np.any(mask) else float("nan"))
	pos = np.arange(len(age_bins))
	ax4.plot(pos, actual_means, marker="o", label="actual")
	ax4.plot(pos, pred_means, marker="o", label="pred")
	ax4.set_xticks(pos)
	ax4.set_xticklabels(age_bins)
	ax4.set_title("Mean disposable by age group")
	ax4.set_xlabel("Age bin")
	ax4.set_ylabel("Mean disposable")
	ax4.legend(loc="best")

	fig.tight_layout()
	path = os.path.join(out_dir, "report.png")
	fig.savefig(path, dpi=180)
	plt.close(fig)

	summary = demographic_group_summary(test_x, test_y, pred)
	csv_path = os.path.join(out_dir, "demographic_summary.csv")
	with open(csv_path, "w", newline="") as f:
		w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
		w.writeheader()
		w.writerows(summary)

	print(f"Saved plots to: {out_dir}")
	print(f"Saved demographic summary CSV: {csv_path}")

TRAIN_COLUMNS_NEEDED = {
	"pternwa",
	"hefaminc",
	"prtage",
	"gestfips",
	"gereg",
	"pemaritl",
	"prnmchld",
	"prchld",
	"hrnumhou",
	"gtmetsta",
	"gtcbsasz",
	"prdisflg",
	"pedisphy",
	"pedisrem",
	"peafever",
	"peio1cow",
	"peerncov",
	"prsjmj",
	"peernuot",
	"prdtocc1",
	"peschenr",
	"peschft",
	"pesex",
	"peeduca",
	"ptdtrace",
	"pehspnon",
}


def _extract_core_inputs(row: Dict[str, str], income_annual: float):
	age = _to_int(_safe_get(row, "prtage"))
	if age is None or age <= 0 or age > 95:
		return None

	core: Dict[str, Optional[float]] = {
		"annual_salary": float(income_annual),
		"prtage": float(age),
		"pesex": _to_int(_safe_get(row, "pesex")),
		"pemaritl": _to_int(_safe_get(row, "pemaritl")),
		"peeduca": _to_int(_safe_get(row, "peeduca")),
		"ptdtrace": _to_int(_safe_get(row, "ptdtrace")),
		"pehspnon": _to_int(_safe_get(row, "pehspnon")),
	}
	return core


def load_training_samples(csv_path: str, sample_size: int, seed: int) -> List[Tuple[Dict[str, Optional[float]], float]]:
	rng = random.Random(seed)
	samples: List[Tuple[Dict[str, Optional[float]], float]] = []
	seen = 0

	with open(csv_path, "r", newline="") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames is None:
			raise ValueError("CSV has no header")

		for row in reader:
			seen += 1

			income = gross_annual_income(row)
			if income is None or income <= 0:
				continue

			disp = compute_disposable_income(row)
			if disp is None:
				continue

			x = _extract_core_inputs(row, income_annual=disp["gross"])
			if x is None:
				continue

			y = float(disp["disposable"])

			item = (x, y)
			if len(samples) < sample_size:
				samples.append(item)
			else:
				j = rng.randint(0, seen - 1)
				if j < sample_size:
					samples[j] = item

	return samples

def estimate_disposable_income(
	model: RidgeModel,
	*,
	annual_salary: float,
	age: int,
	sex_code,
	marital_code,
	education_code,
	race_code,
	hispanic_code,
) -> float:
	features: Dict[str, Optional[float]] = {
		"annual_salary": float(annual_salary),
		"prtage": float(age),
		"pesex": sex_code,
		"pemaritl": marital_code,
		"peeduca": education_code,
		"ptdtrace": race_code,
		"pehspnon": hispanic_code,
	}
	pred = model.predict_one(features)
	return max(0.0, float(pred))


def _fmt_money(x: float) -> str:
	return f"${x:,.0f}"


def run_demo(model: RidgeModel) -> None:
	profiles = [
		{
			"label": "Single, 22, HS, $30k",
			"annual_salary": 30000,
			"age": 22,
			"sex_code": 1,
			"marital_code": 6,
			"education_code": 39,
			"race_code": 1,
			"hispanic_code": 2,
		},
		{
			"label": "Married, 35, college, $75k",
			"annual_salary": 75000,
			"age": 35,
			"sex_code": 2,
			"marital_code": 1,
			"education_code": 43,
			"race_code": 1,
			"hispanic_code": 2,
		},
		{
			"label": "Single parent proxy, 29, some college, $48k",
			"annual_salary": 48000,
			"age": 29,
			"sex_code": 2,
			"marital_code": 5,
			"education_code": 40,
			"race_code": 2,
			"hispanic_code": 2,
		},
		{
			"label": "Mid-career, 45, college+, $120k",
			"annual_salary": 120000,
			"age": 45,
			"sex_code": 1,
			"marital_code": 4,
			"education_code": 46,
			"race_code": 1,
			"hispanic_code": 2,
		},
		{
			"label": "Older, 67, retired-ish proxy, $55k",
			"annual_salary": 55000,
			"age": 67,
			"sex_code": 2,
			"marital_code": 3,
			"education_code": 38,
			"race_code": 1,
			"hispanic_code": 2,
		},
		{
			"label": "High income, 40, grad, $220k",
			"annual_salary": 220000,
			"age": 40,
			"sex_code": 1,
			"marital_code": 1,
			"education_code": 48,
			"race_code": 1,
			"hispanic_code": 2,
		},
	]

	print("\nDemo predictions (model inputs: salary + core demographics)")
	print("-" * 72)
	for p in profiles:
		pred = estimate_disposable_income(
			model,
			annual_salary=p["annual_salary"],
			age=p["age"],
			sex_code=p.get("sex_code"),
			marital_code=p.get("marital_code"),
			education_code=p.get("education_code"),
			race_code=p.get("race_code"),
			hispanic_code=p.get("hispanic_code"),
		)
		pct = 100.0 * pred / max(1.0, float(p["annual_salary"]))
		print(f"{p['label']:<40} -> disposable ~ {_fmt_money(pred)} ({pct:.1f}% of gross)")


def main(argv: Optional[Sequence[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Train and demo a disposable-income estimator")
	parser.add_argument("--data", default=os.path.join("data", "dec25pub.csv"), help="Path to CPS-style CSV (default: data/dec25pub.csv)")
	parser.add_argument("--sample", type=int, default=50000, help="Reservoir sample size for training (default: 50000)")
	parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
	parser.add_argument("--l2", type=float, default=25.0, help="Ridge L2 strength (default: 25.0)")
	parser.add_argument("--test-min", type=int, default=1000, help="Minimum holdout test points (default: 1000)")
	parser.add_argument("--test-frac", type=float, default=0.2, help="Holdout test fraction (default: 0.2)")
	parser.add_argument("--out", default="outputs", help="Output directory for saved artifacts (default: outputs)")
	parser.add_argument("--no-demo", action="store_true", help="Disable printing demo profiles")
	parser.add_argument("--plots", default=None, help="If set, save charts/graphs into this output directory (e.g., outputs)")
	parser.add_argument("--max-plot-points", type=int, default=8000, help="Max points in scatter plots (default: 8000)")
	args = parser.parse_args(argv)

	if not os.path.exists(args.data):
		print(f"Data file not found: {args.data}", file=sys.stderr)
		return 2

	print(f"Loading training samples from {args.data} (sample={args.sample})...")
	samples = load_training_samples(args.data, sample_size=args.sample, seed=args.seed)
	if len(samples) < 2000:
		print(f"not enough usable rows to train (got {len(samples)}).", file=sys.stderr)
		print("try a larger sample size or confirm dec25pub.csv has the expected columns.", file=sys.stderr)
		return 2

	print(f"Training model on {len(samples)} sampled rows...")
	model, metrics = train_model(samples, seed=args.seed, l2=args.l2, test_min=args.test_min, test_frac=args.test_frac)
	print(
		"Metrics on held-out split: "
		f"MAE={_fmt_money(metrics['mae'])}, RMSE={_fmt_money(metrics['rmse'])}, R2={metrics['r2']:.3f} "
		f"(train={metrics['n_train']}, test={metrics['n_test']})"
	)

	save_model_artifacts(model, args.out)

	test_x = metrics.get("_test_x")
	test_y = metrics.get("_test_y")
	test_pred = metrics.get("_test_pred")
	if isinstance(test_x, list) and isinstance(test_y, np.ndarray) and isinstance(test_pred, np.ndarray):
		save_test_predictions(args.out, test_x, test_y, test_pred)

		tabulate = _try_import_tabulate()
		summary = demographic_group_summary(test_x, test_y, test_pred)
		print("\nDemographic summary (holdout):")
		slice_rows = [r for r in summary if r["group"] in {"age_bin", "sex"}]
		if tabulate is not None:
			print(tabulate(slice_rows, headers="keys", tablefmt="github", floatfmt=",.0f"))
		else:
			for r in slice_rows:
				print(r)

		plot_dir = args.plots if args.plots else None
		if plot_dir:
			save_plots(plot_dir, test_x, test_y, test_pred, max_points=args.max_plot_points)

	if not args.no_demo:
		run_demo(model)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

