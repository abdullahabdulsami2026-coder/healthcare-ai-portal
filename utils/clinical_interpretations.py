"""
Clinical interpretation text generator for the Healthcare AI Portal.
Provides plain-language explanations of test results for patients and clinicians.
"""


# ============================================================
# Heart / ECG Interpretations
# ============================================================

ECG_CLASS_EXPLANATIONS = {
    "Normal ECG": (
        "Your ECG shows a **normal sinus rhythm**, meaning your heart's electrical activity "
        "is functioning as expected. The heart rate, rhythm, and waveform morphology are all "
        "within normal limits. No signs of arrhythmia, conduction abnormalities, or ischemic changes."
    ),
    "Myocardial Infarction": (
        "The ECG pattern is suggestive of a **myocardial infarction (heart attack)**. This indicates "
        "that part of the heart muscle may have been damaged due to reduced blood flow. Key findings "
        "may include ST-segment elevation, pathological Q waves, or T-wave inversions. "
        "**Immediate clinical evaluation is strongly recommended.**"
    ),
    "ST/T Change": (
        "The ECG shows **ST-segment or T-wave changes**, which can indicate a variety of conditions "
        "including myocardial ischemia (reduced blood flow to the heart), electrolyte imbalances, "
        "medication effects, or left ventricular strain. These changes should be correlated with "
        "clinical symptoms and prior ECGs for proper interpretation."
    ),
    "Hypertrophy": (
        "The ECG pattern suggests **ventricular hypertrophy**, meaning the heart muscle appears "
        "thickened. This is commonly associated with chronic high blood pressure (hypertension), "
        "valvular heart disease, or athletic conditioning. High-voltage QRS complexes are the "
        "hallmark finding. An echocardiogram may be recommended for further evaluation."
    ),
    "Conduction Disturbance": (
        "The ECG indicates a **conduction disturbance**, meaning the electrical signals in the heart "
        "are not traveling along their normal pathways. This may include bundle branch blocks, "
        "AV blocks, or fascicular blocks. Depending on the type and severity, this may be benign "
        "or may require further cardiac evaluation and monitoring."
    ),
}

ECG_METRIC_EXPLANATIONS = {
    "heart_rate": (
        "**Heart Rate (BPM):** The number of times your heart beats per minute. "
        "Normal resting heart rate is 60-100 BPM. Below 60 is bradycardia (slow), "
        "above 100 is tachycardia (fast). Athletes often have lower resting rates."
    ),
    "sdnn": (
        "**SDNN (ms):** Standard deviation of all R-R intervals. This is a key measure of "
        "overall heart rate variability (HRV). Higher SDNN indicates better autonomic nervous "
        "system function. Values below 50ms suggest reduced variability, which may be associated "
        "with increased cardiac risk."
    ),
    "rmssd": (
        "**RMSSD (ms):** Root mean square of successive R-R interval differences. This reflects "
        "parasympathetic (vagal) activity. Higher values indicate good vagal tone and cardiac "
        "resilience. Low RMSSD may suggest autonomic dysfunction or increased stress."
    ),
}


# ============================================================
# Health Risk Assessment Interpretations
# ============================================================

def interpret_heart_risk(predicted, risk_score, age, sex, cp, trestbps, chol, thalach, oldpeak, ca):
    """Generate comprehensive heart risk interpretation."""
    sections = []

    # Overall risk
    if predicted == "High Risk":
        sections.append(
            "### Overall Assessment: Elevated Risk\n"
            f"Your heart disease risk score is **{risk_score:.0f}%**, placing you in the **high-risk** category. "
            "This means the combination of your clinical factors suggests an increased likelihood of "
            "heart disease. This is not a diagnosis, but an indication that further evaluation and "
            "preventive measures may be warranted."
        )
    else:
        sections.append(
            "### Overall Assessment: Lower Risk\n"
            f"Your heart disease risk score is **{risk_score:.0f}%**, placing you in the **lower-risk** category. "
            "While this is reassuring, it does not eliminate the possibility of heart disease. "
            "Maintaining a healthy lifestyle and regular check-ups remain important."
        )

    # Factor-by-factor breakdown
    factors = []

    if trestbps > 140:
        factors.append(
            f"- **Blood Pressure ({trestbps} mmHg):** Your resting BP is elevated (>140 mmHg), "
            "classified as stage 2 hypertension. High blood pressure is a major risk factor for "
            "heart disease, stroke, and kidney damage. Lifestyle changes and medication may be needed."
        )
    elif trestbps > 130:
        factors.append(
            f"- **Blood Pressure ({trestbps} mmHg):** Your resting BP is mildly elevated (130-140 mmHg), "
            "classified as stage 1 hypertension. Dietary changes, exercise, and monitoring are recommended."
        )

    if chol > 240:
        factors.append(
            f"- **Cholesterol ({chol} mg/dL):** Your total cholesterol is high (>240 mg/dL). "
            "Elevated cholesterol contributes to plaque buildup in arteries (atherosclerosis), "
            "increasing heart attack and stroke risk. A detailed lipid panel is recommended."
        )
    elif chol > 200:
        factors.append(
            f"- **Cholesterol ({chol} mg/dL):** Your total cholesterol is borderline high (200-240 mg/dL). "
            "Dietary modifications and regular monitoring are advised."
        )

    if thalach < 120:
        factors.append(
            f"- **Max Heart Rate ({thalach} BPM):** Your maximum achieved heart rate during exercise "
            "is below expected levels. This may indicate reduced cardiovascular fitness or "
            "chronotropic incompetence, which is associated with increased cardiac risk."
        )

    if oldpeak > 2:
        factors.append(
            f"- **ST Depression ({oldpeak}):** Significant ST depression during exercise suggests "
            "exercise-induced ischemia (reduced blood flow to the heart during exertion). "
            "This finding warrants further cardiac evaluation, potentially including stress imaging."
        )

    if ca > 0:
        factors.append(
            f"- **Major Vessels ({ca}):** Fluoroscopy shows {ca} major coronary vessel(s) with "
            "significant narrowing. This is a strong predictor of coronary artery disease."
        )

    if cp == "Asymptomatic":
        factors.append(
            "- **Chest Pain Type (Asymptomatic):** Absence of typical chest pain does not rule out "
            "heart disease. Silent ischemia is common, especially in diabetic patients."
        )

    if factors:
        sections.append("### Key Risk Factors\n" + "\n".join(factors))
    else:
        sections.append(
            "### Key Risk Factors\n"
            "No individual risk factors were flagged as significantly elevated. "
            "Continue maintaining healthy habits."
        )

    # Recommendations
    sections.append(
        "### What This Means\n"
        "- **Risk Score** represents the statistical likelihood of heart disease based on clinical features\n"
        "- **Radar Chart** shows how each vital compares relative to its clinical range — "
        "larger extensions indicate values further from optimal\n"
        "- **Gauge** visualizes overall risk: green (0-30%) is low, yellow (30-60%) is moderate, red (60-100%) is high\n"
    )

    return "\n\n".join(sections)


# ============================================================
# Chest X-Ray Interpretations
# ============================================================

def interpret_xray(predicted, confidence):
    """Generate X-ray result interpretation."""
    if predicted == "Pneumonia":
        return (
            f"### Finding: Pneumonia Detected (Confidence: {confidence}%)\n\n"
            "The AI model has identified patterns in this chest X-ray that are consistent with "
            "**pneumonia** — an infection that inflames the air sacs in one or both lungs. "
            "Common radiographic findings include:\n\n"
            "- **Consolidation:** Dense white areas indicating fluid-filled alveoli\n"
            "- **Air bronchograms:** Visible air-filled bronchi surrounded by opacified lung tissue\n"
            "- **Interstitial patterns:** Reticular or ground-glass opacities\n\n"
            "**What to do next:** This finding should be correlated with clinical symptoms "
            "(cough, fever, shortness of breath), lab work (CBC, inflammatory markers), "
            "and possibly sputum cultures for definitive diagnosis."
        )
    else:
        return (
            f"### Finding: Normal Appearance (Confidence: {confidence}%)\n\n"
            "The AI model did not detect significant abnormalities in this chest X-ray. "
            "The lung fields appear clear without evidence of consolidation, effusion, or "
            "mass lesions. The cardiac silhouette and mediastinal contours appear within normal limits.\n\n"
            "**Note:** A normal AI reading does not exclude all pathology. Subtle findings "
            "may require expert radiologist review."
        )


# ============================================================
# CBC Result Explanations
# ============================================================

CBC_EXPLANATIONS = {
    "WBC": (
        "**White Blood Cells (WBC)** are your immune system's defense force. They fight infections "
        "and foreign invaders. High WBC (leukocytosis) can indicate infection, inflammation, stress, "
        "or blood disorders. Low WBC (leukopenia) may suggest bone marrow problems, autoimmune "
        "conditions, or medication side effects."
    ),
    "RBC": (
        "**Red Blood Cells (RBC)** carry oxygen from your lungs to all body tissues and return "
        "carbon dioxide to be exhaled. Low RBC count indicates anemia (fatigue, weakness, pallor). "
        "High RBC may indicate dehydration, lung disease, or polycythemia."
    ),
    "Hemoglobin": (
        "**Hemoglobin (Hgb)** is the protein in red blood cells that carries oxygen. It's the "
        "primary marker for anemia. Low hemoglobin means your blood carries less oxygen, causing "
        "fatigue, shortness of breath, and dizziness. High hemoglobin may occur with dehydration, "
        "lung disease, or living at high altitude."
    ),
    "Hematocrit": (
        "**Hematocrit (Hct)** is the percentage of your blood volume made up of red blood cells. "
        "It moves in parallel with hemoglobin. Low hematocrit confirms anemia. High hematocrit "
        "may indicate dehydration or polycythemia vera."
    ),
    "MCV": (
        "**Mean Corpuscular Volume (MCV)** measures the average size of your red blood cells. "
        "Low MCV (microcytic) suggests iron deficiency or thalassemia. High MCV (macrocytic) "
        "suggests B12 or folate deficiency, liver disease, or certain medications. Normal MCV "
        "with anemia points to chronic disease or acute blood loss."
    ),
    "MCH": (
        "**Mean Corpuscular Hemoglobin (MCH)** is the average amount of hemoglobin per red blood cell. "
        "Low MCH typically accompanies microcytic anemia (iron deficiency). High MCH accompanies "
        "macrocytic anemia (B12/folate deficiency)."
    ),
    "MCHC": (
        "**Mean Corpuscular Hemoglobin Concentration (MCHC)** measures hemoglobin concentration "
        "in red blood cells. Low MCHC (hypochromic) is seen in iron deficiency. High MCHC may "
        "indicate spherocytosis or severe dehydration."
    ),
    "RDW": (
        "**Red Cell Distribution Width (RDW)** measures variation in red blood cell size (anisocytosis). "
        "Elevated RDW with low hemoglobin strongly suggests iron deficiency anemia. Normal RDW with "
        "anemia may point to chronic disease or thalassemia trait."
    ),
    "Platelets": (
        "**Platelets** are cell fragments that help your blood clot. Low platelets (thrombocytopenia) "
        "increase bleeding risk — watch for easy bruising, petechiae, or prolonged bleeding. "
        "High platelets (thrombocytosis) may be reactive (infection/inflammation) or primary "
        "(myeloproliferative disorder), increasing clotting risk."
    ),
    "Neutrophils": (
        "**Neutrophils** are the most abundant white blood cells and the first responders to bacterial "
        "infections. High neutrophils suggest bacterial infection, inflammation, or stress response. "
        "Low neutrophils (neutropenia) increase risk of serious infections."
    ),
    "Lymphocytes": (
        "**Lymphocytes** handle viral infections and immune memory (T-cells and B-cells). "
        "High lymphocytes suggest viral infection, chronic inflammation, or lymphoproliferative disorders. "
        "Low lymphocytes may indicate HIV, immunosuppression, or acute stress."
    ),
    "Monocytes": (
        "**Monocytes** are part of the innate immune system and become macrophages in tissues. "
        "Elevated monocytes are seen in chronic infections (tuberculosis), autoimmune diseases, "
        "and some blood cancers (chronic myelomonocytic leukemia)."
    ),
    "Eosinophils": (
        "**Eosinophils** are involved in allergic reactions and parasitic infections. "
        "Elevated eosinophils suggest allergies, asthma, parasitic infection, or eosinophilic disorders. "
        "They are rarely low, as they normally make up a small percentage of WBCs."
    ),
    "Basophils": (
        "**Basophils** are the rarest white blood cells, involved in allergic and inflammatory reactions. "
        "They release histamine and heparin. Elevated basophils are rare but may be seen in "
        "myeloproliferative disorders or severe allergic reactions."
    ),
}


# ============================================================
# Diabetes Screening Interpretations
# ============================================================

def interpret_diabetes_results(hba1c, glucose, findrisc_score, findrisc_cat, findrisc_risk):
    """Generate comprehensive diabetes screening interpretation."""
    sections = []

    # HbA1c explanation
    if hba1c < 5.7:
        sections.append(
            f"### HbA1c: {hba1c}% — Normal\n"
            "Your HbA1c reflects your average blood sugar over the past 2-3 months. "
            "A level below 5.7% indicates **normal glucose metabolism**. Your body is effectively "
            "managing blood sugar levels."
        )
    elif hba1c < 6.5:
        sections.append(
            f"### HbA1c: {hba1c}% — Prediabetes Range\n"
            "Your HbA1c is between 5.7% and 6.4%, indicating **prediabetes**. This means your "
            "blood sugar is higher than normal but not yet at the diabetes threshold. "
            "Without intervention, approximately 15-30% of people with prediabetes develop type 2 "
            "diabetes within 5 years. **Lifestyle changes (diet, exercise, weight loss) can reverse this.**"
        )
    else:
        sections.append(
            f"### HbA1c: {hba1c}% — Diabetes Range\n"
            "Your HbA1c is 6.5% or higher, which meets the **diagnostic threshold for diabetes** "
            "per ADA criteria. This indicates sustained elevated blood sugar that may cause damage "
            "to blood vessels, nerves, kidneys, and eyes over time. Confirmatory testing and "
            "clinical evaluation are recommended."
        )

    # Fasting glucose
    if glucose < 100:
        sections.append(
            f"### Fasting Glucose: {glucose} mg/dL — Normal\n"
            "A fasting glucose below 100 mg/dL indicates your body is managing sugar properly after "
            "an overnight fast."
        )
    elif glucose < 126:
        sections.append(
            f"### Fasting Glucose: {glucose} mg/dL — Impaired Fasting Glucose\n"
            "A fasting glucose between 100-125 mg/dL indicates **impaired fasting glucose (prediabetes)**. "
            "Your pancreas may be producing enough insulin, but your cells may be becoming resistant to it."
        )
    else:
        sections.append(
            f"### Fasting Glucose: {glucose} mg/dL — Diabetes Range\n"
            "A fasting glucose of 126 mg/dL or higher meets the **diagnostic criteria for diabetes**. "
            "This should be confirmed with a repeat test on a separate day."
        )

    # FINDRISC
    sections.append(
        f"### FINDRISC Score: {findrisc_score} — {findrisc_cat} Risk\n"
        f"Your 10-year risk of developing type 2 diabetes is approximately **{findrisc_risk}**. "
        "The FINDRISC (Finnish Diabetes Risk Score) is a validated questionnaire that predicts "
        "diabetes risk based on age, BMI, waist circumference, physical activity, diet, "
        "blood pressure medication, glucose history, and family history.\n\n"
    )

    if findrisc_score >= 15:
        sections.append(
            "**Recommendation:** Your risk is elevated. Consider:\n"
            "- Annual HbA1c and fasting glucose monitoring\n"
            "- Structured weight management program\n"
            "- 150+ minutes/week of moderate physical activity\n"
            "- Dietary consultation (Mediterranean or DASH diet)\n"
            "- Diabetes prevention program enrollment"
        )
    elif findrisc_score >= 12:
        sections.append(
            "**Recommendation:** Moderate risk warrants attention:\n"
            "- Biennial screening with HbA1c\n"
            "- Maintain healthy weight (BMI < 25)\n"
            "- Regular exercise and balanced diet"
        )

    return "\n\n".join(sections)


# ============================================================
# Lipid Panel Interpretations
# ============================================================

def interpret_lipid_results(tc, ldl, hdl, tg, ascvd_risk, ascvd_cat, tc_hdl_ratio, ldl_hdl_ratio):
    """Generate comprehensive lipid panel interpretation."""
    sections = []

    sections.append("### Understanding Your Lipid Panel\n")

    # Total cholesterol
    if tc < 200:
        sections.append(f"**Total Cholesterol ({tc} mg/dL) — Desirable.** Your total cholesterol is within the healthy range.")
    elif tc < 240:
        sections.append(f"**Total Cholesterol ({tc} mg/dL) — Borderline High.** You are approaching the high-risk threshold. Dietary changes may help.")
    else:
        sections.append(f"**Total Cholesterol ({tc} mg/dL) — High.** Elevated total cholesterol increases atherosclerosis risk. Further evaluation and treatment may be needed.")

    # LDL
    if ldl < 100:
        sections.append(f"\n**LDL Cholesterol ({ldl} mg/dL) — Optimal.** LDL is the 'bad' cholesterol that builds up in artery walls. Your level is ideal.")
    elif ldl < 130:
        sections.append(f"\n**LDL Cholesterol ({ldl} mg/dL) — Near Optimal.** Slightly above optimal but acceptable for most people without additional risk factors.")
    elif ldl < 160:
        sections.append(f"\n**LDL Cholesterol ({ldl} mg/dL) — Borderline High.** Consider lifestyle modifications: reduce saturated fat, increase fiber, exercise regularly.")
    elif ldl < 190:
        sections.append(f"\n**LDL Cholesterol ({ldl} mg/dL) — High.** Statin therapy should be discussed with your healthcare provider, especially with other risk factors.")
    else:
        sections.append(f"\n**LDL Cholesterol ({ldl} mg/dL) — Very High.** LDL >= 190 may indicate familial hypercholesterolemia. Aggressive lipid-lowering therapy is typically recommended.")

    # HDL
    if hdl >= 60:
        sections.append(f"\n**HDL Cholesterol ({hdl} mg/dL) — Protective.** HDL is the 'good' cholesterol that removes LDL from arteries. Your high level is a protective factor.")
    elif hdl >= 40:
        sections.append(f"\n**HDL Cholesterol ({hdl} mg/dL) — Normal.** Adequate but not at the protective threshold (>=60). Exercise and healthy fats can raise HDL.")
    else:
        sections.append(f"\n**HDL Cholesterol ({hdl} mg/dL) — Low (Risk Factor).** Low HDL is an independent cardiovascular risk factor. Aerobic exercise, omega-3 fatty acids, and moderate alcohol intake may help.")

    # Triglycerides
    if tg < 150:
        sections.append(f"\n**Triglycerides ({tg} mg/dL) — Normal.** Triglycerides are fats from food that circulate in blood. Your level is healthy.")
    elif tg < 200:
        sections.append(f"\n**Triglycerides ({tg} mg/dL) — Borderline High.** Reduce refined carbohydrates, sugar, and alcohol. Increase omega-3 intake.")
    elif tg < 500:
        sections.append(f"\n**Triglycerides ({tg} mg/dL) — High.** Significantly elevated. May require medication in addition to lifestyle changes. Check for secondary causes (diabetes, hypothyroidism).")
    else:
        sections.append(f"\n**Triglycerides ({tg} mg/dL) — Very High (Pancreatitis Risk).** TG >= 500 mg/dL carries risk of acute pancreatitis. Urgent treatment needed.")

    # Ratios
    sections.append(
        f"\n### Lipid Ratios\n"
        f"- **TC/HDL Ratio: {tc_hdl_ratio}** — {'Good (<5.0)' if tc_hdl_ratio < 5 else 'Elevated (>=5.0, increased CV risk)'}\n"
        f"- **LDL/HDL Ratio: {ldl_hdl_ratio}** — {'Good (<3.0)' if ldl_hdl_ratio < 3 else 'Elevated (>=3.0)'}\n\n"
        "These ratios provide additional insight beyond individual values. Lower ratios indicate "
        "a healthier balance between harmful and protective cholesterol."
    )

    # ASCVD
    if ascvd_risk is not None:
        sections.append(
            f"\n### 10-Year ASCVD Risk: {ascvd_risk}% — {ascvd_cat}\n"
            "This is your estimated probability of having a heart attack or stroke in the next 10 years, "
            "calculated using the ACC/AHA Pooled Cohort Equations. "
        )
        if ascvd_risk >= 20:
            sections.append("**High risk (>=20%).** ACC/AHA guidelines recommend high-intensity statin therapy and aggressive risk factor management.")
        elif ascvd_risk >= 7.5:
            sections.append("**Intermediate risk (7.5-20%).** Moderate-to-high-intensity statin therapy should be considered. Risk-enhancing factors (family history, CRP, coronary calcium score) may inform treatment decisions.")
        elif ascvd_risk >= 5:
            sections.append("**Borderline risk (5-7.5%).** Lifestyle optimization is the primary intervention. Statin therapy may be considered if risk-enhancing factors are present.")
        else:
            sections.append("**Low risk (<5%).** Focus on maintaining healthy lifestyle habits. Reassess in 5 years.")

    return "\n".join(sections)


# ============================================================
# Kidney Function Interpretations
# ============================================================

def interpret_kidney_results(egfr_cr, egfr_cys, ckd_stage, ckd_desc, alb_stage, alb_desc, uacr, bun, creatinine):
    """Generate comprehensive kidney function interpretation."""
    sections = []

    bun_cr_ratio = round(bun / creatinine, 1) if creatinine > 0 else 0

    # eGFR explanation
    sections.append(
        f"### eGFR: {egfr_cr} mL/min/1.73m\u00b2 — Stage {ckd_stage} ({ckd_desc})\n\n"
        "The **estimated Glomerular Filtration Rate (eGFR)** measures how well your kidneys "
        "filter waste from your blood. It is the single best indicator of kidney function.\n"
    )

    if egfr_cr >= 90:
        sections.append("Your eGFR is **normal**. Your kidneys are filtering blood effectively. If albuminuria is also normal, kidney function is healthy.")
    elif egfr_cr >= 60:
        sections.append("Your eGFR is **mildly decreased**. This may be age-related in older adults. If persistent and accompanied by albuminuria, it may indicate early chronic kidney disease.")
    elif egfr_cr >= 45:
        sections.append("Your eGFR indicates **mild-to-moderate kidney impairment**. Monitoring kidney function every 3-6 months is recommended. Nephrotoxic medications should be reviewed.")
    elif egfr_cr >= 30:
        sections.append("Your eGFR indicates **moderate-to-severe kidney impairment**. Nephrology referral is recommended. Medication dosing may need adjustment. Monitor for complications (anemia, bone disease, electrolyte imbalances).")
    elif egfr_cr >= 15:
        sections.append("Your eGFR indicates **severe kidney impairment**. Active nephrology management is essential. Begin planning for potential renal replacement therapy (dialysis or transplant).")
    else:
        sections.append("Your eGFR indicates **kidney failure**. Renal replacement therapy (dialysis or kidney transplant) should be actively discussed.")

    # Cystatin C comparison
    if egfr_cys is not None:
        diff = abs(egfr_cr - egfr_cys)
        sections.append(
            f"\n### Cystatin C-Based eGFR: {egfr_cys} mL/min/1.73m\u00b2\n"
            f"The creatinine-based and cystatin C-based eGFR differ by {diff:.0f} mL/min. "
        )
        if diff > 15:
            sections.append("This significant discrepancy may be due to factors affecting creatinine (muscle mass, diet, medications). The cystatin C-based estimate may be more accurate in these cases.")
        else:
            sections.append("The two estimates are concordant, increasing confidence in the kidney function assessment.")

    # Albuminuria
    sections.append(
        f"\n### Albuminuria: UACR {uacr} mg/g — Stage {alb_stage} ({alb_desc})\n\n"
        "**Urine Albumin-to-Creatinine Ratio (UACR)** detects protein leaking into urine, "
        "which is an early sign of kidney damage — often before eGFR drops.\n"
    )
    if uacr < 30:
        sections.append("Your UACR is **normal**. No significant protein is leaking into your urine.")
    elif uacr <= 300:
        sections.append("**Moderately increased albuminuria (microalbuminuria)** detected. This is an early marker of kidney damage and also an independent cardiovascular risk factor. ACE inhibitor or ARB therapy may be beneficial.")
    else:
        sections.append("**Severely increased albuminuria (macroalbuminuria)** detected. This indicates significant kidney damage. Aggressive treatment of the underlying cause (diabetes, hypertension) and nephrology referral are recommended.")

    # BUN/Creatinine ratio
    sections.append(
        f"\n### BUN/Creatinine Ratio: {bun_cr_ratio}\n"
        "Normal ratio is 10:1 to 20:1. "
    )
    if bun_cr_ratio > 20:
        sections.append("Your elevated ratio may indicate pre-renal causes such as dehydration, heart failure, or gastrointestinal bleeding. Adequate hydration and clinical correlation are important.")
    elif bun_cr_ratio < 10:
        sections.append("A low ratio may be seen in liver disease, malnutrition, or conditions that reduce BUN production.")
    else:
        sections.append("Your ratio is within the normal range.")

    return "\n\n".join(sections)


# ============================================================
# Lab Report General Explanations
# ============================================================

LAB_GENERAL_EXPLANATIONS = {
    "WBC": "Immune cells that fight infection. High = possible infection/inflammation. Low = immune suppression risk.",
    "RBC": "Oxygen-carrying red cells. Low = anemia (fatigue, weakness). High = dehydration or blood disorder.",
    "Hemoglobin": "Oxygen-carrying protein in RBCs. The primary anemia marker. Low = reduced oxygen delivery.",
    "Hematocrit": "Percentage of blood volume that is red blood cells. Moves in parallel with hemoglobin.",
    "Platelets": "Clotting cell fragments. Low = bleeding risk. High = clotting risk.",
    "Glucose": "Blood sugar level. High = possible diabetes or prediabetes. Low = hypoglycemia risk.",
    "BUN": "Blood urea nitrogen — kidney waste product. High = possible kidney dysfunction or dehydration.",
    "Creatinine": "Muscle metabolism waste filtered by kidneys. High = impaired kidney function.",
    "Sodium": "Key electrolyte for fluid balance and nerve function. Abnormal levels affect brain and muscle function.",
    "Potassium": "Critical for heart rhythm. High = cardiac arrhythmia risk. Low = muscle weakness, cardiac risk.",
    "Calcium": "Essential for bones, muscles, and nerve signaling. Abnormal levels can cause neurological and cardiac symptoms.",
    "Total Protein": "Sum of albumin and globulin. Reflects liver function and nutritional/immune status.",
    "ALT": "Liver enzyme (alanine aminotransferase). Elevated = possible liver inflammation or damage.",
    "AST": "Liver/muscle enzyme (aspartate aminotransferase). Elevated with ALT = liver damage. Alone = muscle injury.",
    "Total Cholesterol": "Sum of all cholesterol types. High levels increase cardiovascular disease risk.",
    "HbA1c": "Glycated hemoglobin — 2-3 month average blood sugar. >5.7% = prediabetes, >6.5% = diabetes.",
    "TSH": "Thyroid-stimulating hormone. High = underactive thyroid (hypothyroidism). Low = overactive (hyperthyroidism).",
}
