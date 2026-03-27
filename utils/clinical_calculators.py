"""
Clinical calculators and reference ranges for the Healthcare AI Portal.
All algorithms are based on published clinical guidelines.
"""

import math

# ============================================================
# Lab Value Classifier
# ============================================================

def classify_value(value, low, high, crit_low=None, crit_high=None):
    """Classify a lab value against reference ranges.
    Returns (status, css_class, color)."""
    if crit_low is not None and value < crit_low:
        return "Critical Low", "flag-critical", "#922b21"
    if crit_high is not None and value > crit_high:
        return "Critical High", "flag-critical", "#922b21"
    if value < low:
        return "Low", "flag-low", "#e67e22"
    if value > high:
        return "High", "flag-high", "#e67e22"
    return "Normal", "flag-normal", "#27ae60"


# ============================================================
# CBC Reference Ranges
# ============================================================

CBC_RANGES = {
    "male": {
        "WBC":        {"unit": "x10\u00b3/\u00b5L", "low": 4.5, "high": 11.0, "crit_low": 2.0, "crit_high": 30.0},
        "RBC":        {"unit": "x10\u2076/\u00b5L", "low": 4.5, "high": 5.5,  "crit_low": 2.0, "crit_high": 8.0},
        "Hemoglobin": {"unit": "g/dL",     "low": 13.5, "high": 17.5, "crit_low": 7.0, "crit_high": 20.0},
        "Hematocrit": {"unit": "%",        "low": 38.3, "high": 48.6, "crit_low": 20.0, "crit_high": 60.0},
        "MCV":        {"unit": "fL",       "low": 80.0, "high": 100.0, "crit_low": 60.0, "crit_high": 120.0},
        "MCH":        {"unit": "pg",       "low": 27.0, "high": 33.0, "crit_low": 20.0, "crit_high": 40.0},
        "MCHC":       {"unit": "g/dL",     "low": 32.0, "high": 36.0, "crit_low": 28.0, "crit_high": 38.0},
        "RDW":        {"unit": "%",        "low": 11.5, "high": 14.5, "crit_low": None, "crit_high": 20.0},
        "Platelets":  {"unit": "x10\u00b3/\u00b5L", "low": 150, "high": 400, "crit_low": 50, "crit_high": 1000},
    },
    "female": {
        "WBC":        {"unit": "x10\u00b3/\u00b5L", "low": 4.5, "high": 11.0, "crit_low": 2.0, "crit_high": 30.0},
        "RBC":        {"unit": "x10\u2076/\u00b5L", "low": 4.0, "high": 5.0,  "crit_low": 2.0, "crit_high": 8.0},
        "Hemoglobin": {"unit": "g/dL",     "low": 12.0, "high": 16.0, "crit_low": 7.0, "crit_high": 20.0},
        "Hematocrit": {"unit": "%",        "low": 35.5, "high": 44.9, "crit_low": 20.0, "crit_high": 60.0},
        "MCV":        {"unit": "fL",       "low": 80.0, "high": 100.0, "crit_low": 60.0, "crit_high": 120.0},
        "MCH":        {"unit": "pg",       "low": 27.0, "high": 33.0, "crit_low": 20.0, "crit_high": 40.0},
        "MCHC":       {"unit": "g/dL",     "low": 32.0, "high": 36.0, "crit_low": 28.0, "crit_high": 38.0},
        "RDW":        {"unit": "%",        "low": 11.5, "high": 14.5, "crit_low": None, "crit_high": 20.0},
        "Platelets":  {"unit": "x10\u00b3/\u00b5L", "low": 150, "high": 400, "crit_low": 50, "crit_high": 1000},
    },
}


def interpret_cbc(values, sex="male"):
    """Generate clinical interpretation text for CBC results."""
    findings = []
    refs = CBC_RANGES[sex]

    wbc = values.get("WBC", 0)
    hgb = values.get("Hemoglobin", 0)
    mcv = values.get("MCV", 0)
    plt = values.get("Platelets", 0)
    rdw = values.get("RDW", 0)
    neut = values.get("Neutrophils", 0)

    if wbc < refs["WBC"]["low"]:
        if neut < 40:
            findings.append("Neutropenia detected. Evaluate infection risk and bone marrow function.")
        else:
            findings.append("Leukopenia noted. Monitor for recurrent infections.")
    elif wbc > refs["WBC"]["high"]:
        findings.append("Leukocytosis present. Evaluate for infection, inflammation, or hematologic malignancy.")

    if hgb < refs["Hemoglobin"]["low"]:
        if mcv < 80:
            findings.append("Microcytic anemia pattern. Consider iron deficiency, thalassemia, or chronic disease.")
        elif mcv > 100:
            findings.append("Macrocytic anemia pattern. Consider B12/folate deficiency or myelodysplasia.")
        else:
            findings.append("Normocytic anemia. Consider chronic disease, acute blood loss, or renal insufficiency.")

    if plt < refs["Platelets"]["low"]:
        findings.append("Thrombocytopenia. Evaluate bleeding risk and underlying etiology.")
    elif plt > refs["Platelets"]["high"]:
        findings.append("Thrombocytosis. Evaluate for reactive vs. primary (myeloproliferative) cause.")

    if rdw > refs["RDW"]["high"] and hgb < refs["Hemoglobin"]["low"]:
        findings.append("Elevated RDW with anemia suggests anisocytosis, commonly seen in iron deficiency.")

    if not findings:
        findings.append("All CBC parameters are within normal reference ranges.")

    return findings


# ============================================================
# FINDRISC Diabetes Risk Score
# ============================================================

def calculate_findrisc(age, bmi, waist, sex, activity, fruit_veg, bp_meds, high_glucose, family_hx):
    """Calculate FINDRISC score. Returns (score, risk_category, ten_year_risk)."""
    score = 0

    # Age
    if age < 45: score += 0
    elif age <= 54: score += 2
    elif age <= 64: score += 3
    else: score += 4

    # BMI
    if bmi < 25: score += 0
    elif bmi <= 30: score += 1
    else: score += 3

    # Waist circumference
    if sex == "male":
        if waist < 94: score += 0
        elif waist <= 102: score += 3
        else: score += 4
    else:
        if waist < 80: score += 0
        elif waist <= 88: score += 3
        else: score += 4

    # Physical activity
    if activity == "low": score += 2

    # Fruit/vegetable intake
    if fruit_veg == "no": score += 1

    # BP medication
    if bp_meds == "yes": score += 2

    # History of high blood glucose
    if high_glucose == "yes": score += 5

    # Family history
    if family_hx == "one_parent": score += 3
    elif family_hx == "both_parents": score += 5

    # Categorize
    if score < 7:
        return score, "Low", "~1%"
    elif score <= 11:
        return score, "Slightly Elevated", "~4%"
    elif score <= 14:
        return score, "Moderate", "~17%"
    elif score <= 20:
        return score, "High", "~33%"
    else:
        return score, "Very High", "~50%"


def classify_hba1c(hba1c):
    """Classify HbA1c per ADA criteria."""
    if hba1c < 5.7:
        return "Normal", "#27ae60"
    elif hba1c < 6.5:
        return "Prediabetes", "#f39c12"
    else:
        return "Diabetes", "#e74c3c"


def classify_fasting_glucose(glucose):
    """Classify fasting glucose per ADA criteria."""
    if glucose < 100:
        return "Normal", "#27ae60"
    elif glucose < 126:
        return "Prediabetes", "#f39c12"
    else:
        return "Diabetes", "#e74c3c"


# ============================================================
# CKD-EPI 2021 eGFR (Race-Free)
# ============================================================

def ckd_epi_creatinine(scr, age, sex):
    """CKD-EPI 2021 race-free equation from serum creatinine."""
    if sex == "female":
        kappa = 0.7
        alpha = -0.241 if scr <= 0.7 else -1.200
        sex_mult = 1.012
    else:
        kappa = 0.9
        alpha = -0.302 if scr <= 0.9 else -1.200
        sex_mult = 1.0

    egfr = 142 * (scr / kappa) ** alpha * (0.9938 ** age) * sex_mult
    return round(egfr, 1)


def ckd_epi_cystatin(cysc, age, sex):
    """CKD-EPI 2021 equation from cystatin C."""
    alpha = -0.499 if cysc <= 0.8 else -1.328
    sex_mult = 0.932 if sex == "female" else 1.0

    egfr = 133 * (cysc / 0.8) ** alpha * (0.9961 ** age) * sex_mult
    return round(egfr, 1)


def stage_ckd(egfr):
    """KDIGO CKD staging from eGFR."""
    if egfr >= 90:
        return "G1", "Normal or High", "#27ae60"
    elif egfr >= 60:
        return "G2", "Mildly Decreased", "#7dcea0"
    elif egfr >= 45:
        return "G3a", "Mild-Moderate Decrease", "#f4d03f"
    elif egfr >= 30:
        return "G3b", "Moderate-Severe Decrease", "#f39c12"
    elif egfr >= 15:
        return "G4", "Severely Decreased", "#e74c3c"
    else:
        return "G5", "Kidney Failure", "#922b21"


def stage_albuminuria(uacr):
    """KDIGO albuminuria staging."""
    if uacr < 30:
        return "A1", "Normal to Mildly Increased", "#27ae60"
    elif uacr <= 300:
        return "A2", "Moderately Increased", "#f39c12"
    else:
        return "A3", "Severely Increased", "#e74c3c"


# ============================================================
# Lipid Panel Classification (ATP III / ACC/AHA)
# ============================================================

LIPID_CLASSES = {
    "Total Cholesterol": [
        (200, "Desirable", "#27ae60"),
        (240, "Borderline High", "#f39c12"),
        (9999, "High", "#e74c3c"),
    ],
    "LDL": [
        (100, "Optimal", "#27ae60"),
        (130, "Near Optimal", "#7dcea0"),
        (160, "Borderline High", "#f39c12"),
        (190, "High", "#e67e22"),
        (9999, "Very High", "#e74c3c"),
    ],
    "HDL": "special",  # reversed — higher is better
    "Triglycerides": [
        (150, "Normal", "#27ae60"),
        (200, "Borderline High", "#f39c12"),
        (500, "High", "#e67e22"),
        (9999, "Very High", "#e74c3c"),
    ],
}


def classify_lipid(name, value):
    """Classify a lipid value. Returns (label, color)."""
    if name == "HDL":
        if value >= 60:
            return "Optimal (Protective)", "#27ae60"
        elif value >= 40:
            return "Normal", "#7dcea0"
        else:
            return "Low (Risk Factor)", "#e74c3c"

    thresholds = LIPID_CLASSES.get(name, [])
    for cutoff, label, color in thresholds:
        if value < cutoff:
            return label, color
    return "Unknown", "#888"


# ============================================================
# Pooled Cohort Equations — 10-Year ASCVD Risk
# ============================================================

# Coefficients: (ln_age, ln_tc, ln_hdl, ln_treated_sbp, ln_untreated_sbp,
#                smoker, diabetes, mean_coeff_sum, baseline_survival)

_PCE_COEFFS = {
    "white_female": {
        "ln_age": -29.799,
        "ln_age_sq": 4.884,
        "ln_tc": 13.540,
        "ln_age_x_ln_tc": -3.114,
        "ln_hdl": -13.578,
        "ln_age_x_ln_hdl": 3.149,
        "ln_sbp_treated": 2.019,
        "ln_sbp_untreated": 1.957,
        "smoker": 7.574,
        "ln_age_x_smoker": -1.665,
        "diabetes": 0.661,
        "mean_sum": -29.18,
        "baseline_survival": 0.9665,
    },
    "white_male": {
        "ln_age": 12.344,
        "ln_age_sq": 0.0,
        "ln_tc": 11.853,
        "ln_age_x_ln_tc": -2.664,
        "ln_hdl": -7.990,
        "ln_age_x_ln_hdl": 1.769,
        "ln_sbp_treated": 1.797,
        "ln_sbp_untreated": 1.764,
        "smoker": 7.837,
        "ln_age_x_smoker": -1.795,
        "diabetes": 0.658,
        "mean_sum": 61.18,
        "baseline_survival": 0.9144,
    },
    "aa_female": {
        "ln_age": 17.114,
        "ln_age_sq": 0.0,
        "ln_tc": 0.940,
        "ln_age_x_ln_tc": 0.0,
        "ln_hdl": -18.920,
        "ln_age_x_ln_hdl": 4.475,
        "ln_sbp_treated": 29.291,
        "ln_sbp_untreated": 27.820,
        "smoker": 0.691,
        "ln_age_x_smoker": 0.0,
        "diabetes": 0.874,
        "mean_sum": 86.61,
        "baseline_survival": 0.9533,
    },
    "aa_male": {
        "ln_age": 2.469,
        "ln_age_sq": 0.0,
        "ln_tc": 0.302,
        "ln_age_x_ln_tc": 0.0,
        "ln_hdl": -0.307,
        "ln_age_x_ln_hdl": 0.0,
        "ln_sbp_treated": 1.916,
        "ln_sbp_untreated": 1.809,
        "smoker": 0.549,
        "ln_age_x_smoker": 0.0,
        "diabetes": 0.645,
        "mean_sum": 19.54,
        "baseline_survival": 0.8954,
    },
}


def calculate_ascvd_risk(age, sex, race, total_chol, hdl, sbp, bp_treated,
                          smoker, diabetes):
    """Calculate 10-year ASCVD risk using Pooled Cohort Equations.
    Valid for ages 40-79. Returns (risk_percent, category, color)."""
    if age < 40 or age > 79:
        return None, "Outside valid range (40-79)", "#888"

    # Select coefficient set
    race_key = "aa" if race == "African American" else "white"
    sex_key = "female" if sex == "female" else "male"
    key = f"{race_key}_{sex_key}"
    c = _PCE_COEFFS[key]

    ln_age = math.log(age)
    ln_tc = math.log(total_chol)
    ln_hdl = math.log(hdl)
    ln_sbp = math.log(sbp)

    ind_sum = (
        c["ln_age"] * ln_age +
        c["ln_age_sq"] * ln_age ** 2 +
        c["ln_tc"] * ln_tc +
        c["ln_age_x_ln_tc"] * ln_age * ln_tc +
        c["ln_hdl"] * ln_hdl +
        c["ln_age_x_ln_hdl"] * ln_age * ln_hdl +
        (c["ln_sbp_treated"] if bp_treated else c["ln_sbp_untreated"]) * ln_sbp +
        c["smoker"] * (1 if smoker else 0) +
        c["ln_age_x_smoker"] * ln_age * (1 if smoker else 0) +
        c["diabetes"] * (1 if diabetes else 0)
    )

    risk = 1 - c["baseline_survival"] ** math.exp(ind_sum - c["mean_sum"])
    risk_pct = round(max(0, min(risk * 100, 100)), 1)

    if risk_pct < 5:
        return risk_pct, "Low", "#27ae60"
    elif risk_pct < 7.5:
        return risk_pct, "Borderline", "#7dcea0"
    elif risk_pct < 20:
        return risk_pct, "Intermediate", "#f39c12"
    else:
        return risk_pct, "High", "#e74c3c"


# ============================================================
# Demo Lab Report Data
# ============================================================

DEMO_LAB_REPORT = [
    {"analyte": "WBC",              "value": 11.8,  "unit": "x10\u00b3/\u00b5L", "ref_low": 4.5,  "ref_high": 11.0},
    {"analyte": "RBC",              "value": 4.2,   "unit": "x10\u2076/\u00b5L", "ref_low": 4.0,  "ref_high": 5.0},
    {"analyte": "Hemoglobin",       "value": 11.2,  "unit": "g/dL",     "ref_low": 12.0, "ref_high": 16.0},
    {"analyte": "Hematocrit",       "value": 34.5,  "unit": "%",        "ref_low": 35.5, "ref_high": 44.9},
    {"analyte": "Platelets",        "value": 285,   "unit": "x10\u00b3/\u00b5L", "ref_low": 150,  "ref_high": 400},
    {"analyte": "Glucose",          "value": 132,   "unit": "mg/dL",    "ref_low": 70,   "ref_high": 100},
    {"analyte": "BUN",              "value": 22,    "unit": "mg/dL",    "ref_low": 7,    "ref_high": 20},
    {"analyte": "Creatinine",       "value": 1.1,   "unit": "mg/dL",    "ref_low": 0.6,  "ref_high": 1.2},
    {"analyte": "Sodium",           "value": 140,   "unit": "mEq/L",    "ref_low": 136,  "ref_high": 145},
    {"analyte": "Potassium",        "value": 5.3,   "unit": "mEq/L",    "ref_low": 3.5,  "ref_high": 5.0},
    {"analyte": "Calcium",          "value": 9.2,   "unit": "mg/dL",    "ref_low": 8.5,  "ref_high": 10.5},
    {"analyte": "Total Protein",    "value": 7.0,   "unit": "g/dL",     "ref_low": 6.0,  "ref_high": 8.3},
    {"analyte": "ALT",              "value": 45,    "unit": "U/L",      "ref_low": 7,    "ref_high": 35},
    {"analyte": "AST",              "value": 32,    "unit": "U/L",      "ref_low": 8,    "ref_high": 33},
    {"analyte": "Total Cholesterol","value": 248,   "unit": "mg/dL",    "ref_low": 0,    "ref_high": 200},
    {"analyte": "HbA1c",            "value": 6.1,   "unit": "%",        "ref_low": 0,    "ref_high": 5.7},
    {"analyte": "TSH",              "value": 2.5,   "unit": "mIU/L",    "ref_low": 0.4,  "ref_high": 4.0},
]
