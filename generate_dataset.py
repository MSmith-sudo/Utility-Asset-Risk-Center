import os
import random
import pandas as pd


os.makedirs("data", exist_ok=True)

materials = ["PVC", "Cast Iron", "Steel", "Copper", "Concrete"]
soil_types = ["Clay", "Sand", "Rock", "Loam"]
pipe_locations = ["Residential", "Commercial", "Industrial", "Downtown", "Rural"]

rows = []

for pipe_id in range(1, 5001):
    age = random.randint(1, 100)
    material = random.choice(materials)
    diameter = random.randint(4, 48)
    pressure = random.randint(20, 150)
    soil_type = random.choice(soil_types)
    previous_breaks = random.randint(0, 10)
    location_type = random.choice(pipe_locations)
    leak_reports = random.randint(0, 15)

    risk_score = 0
    risk_score += age * 0.04
    risk_score += previous_breaks * 1.8
    risk_score += pressure * 0.025
    risk_score += leak_reports * 0.7

    if material == "Cast Iron":
        risk_score += 4
    elif material == "Steel":
        risk_score += 2.5
    elif material == "Concrete":
        risk_score += 1.5

    if soil_type == "Clay":
        risk_score += 2
    elif soil_type == "Rock":
        risk_score += 1

    if location_type == "Downtown":
        risk_score += 1.5
    elif location_type == "Industrial":
        risk_score += 1

    if risk_score >= 18:
        failure_risk = 2
    elif risk_score >= 10:
        failure_risk = 1
    else:
        failure_risk = 0

    rows.append({
        "pipe_id": pipe_id,
        "age": age,
        "material": material,
        "diameter": diameter,
        "pressure": pressure,
        "soil_type": soil_type,
        "previous_breaks": previous_breaks,
        "location_type": location_type,
        "leak_reports": leak_reports,
        "failure_risk": failure_risk
    })

df = pd.DataFrame(rows)
df.to_csv("data/infrastructure_failures.csv", index=False)

print("Dataset created successfully.")
print(df.head())