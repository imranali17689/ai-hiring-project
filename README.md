# AI Hiring Fairness Project

## Research Question

"Can fairness adjustments reduce bias in AI hiring predictions without significantly lowering model accuracy?"

## Team

- Imran
- Parker
- Jack
- Jarred

## Project Overview

This project compares a baseline AI hiring prediction model with a simple fairness-adjusted version using the `Adult.csv` dataset (from the UCI Adult Income dataset).

We train a small classifier to predict whether an applicant’s income is `>50K` or `<=50K`, then measure selection rates for `Male` vs `Female` applicants. Finally, we apply a basic fairness adjustment to reduce disparity between demographic groups and report accuracy and selection-rate changes side by side.
