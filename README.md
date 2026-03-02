# M3 Modeling Challenge – Workspace

This workspace contains datasets and an implementation for **Problem #1 (Q1)**: estimating disposable income from salary and demographics.

## Q1: Disposable income model

Code lives in [src/m3_q1](src/m3_q1) with runnable entry points:

- Demo (prints a table of estimates for multiple demographic profiles):
	- `./.venv/bin/python q1_demo.py`
- CLI (one-off estimate):
	- `./.venv/bin/python q1_cli.py --salary 85000 --age 37 --region West --filing-status single --state-tax-rate 0.02`

What the model does:

- Approximates **US 2024** federal income tax + employee payroll taxes (FICA).
- Estimates **essential spending** using the provided BLS CES expenditure table by age group and region, scaled to the user’s salary.
- Returns **disposable income = after-tax income − essential spending**.

More detail and equations are in [src/m3_q1/README_Q1.md](src/m3_q1/README_Q1.md).