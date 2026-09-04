#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json

##pd permettra de joindre les deux axes sur le même 

import matplotlib.pyplot as plt
import numpy as np
import base64

##import seaborn as sns

plt.style.use('seaborn-v0_8-muted') 
plt.rcParams['figure.figsize'] = (10, 6)


# In[2]:


with open("donnee1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for i in data["data"]:
    acc = i["acc"]
    ts = acc.get("Timestamp")
    for m in acc.get("ArrayAcc", []):
        rows.append({
            "Timestamp": ts,
            "x": m.get("x"),
            "y": m.get("y"),
            "z": m.get("z"),
        })

df = pd.DataFrame(rows)

# Export Excel
df.to_excel("donnee1.xlsx", index=False)


# In[3]:


#1.1 Lecture par Pandas des données
df_brut_IMU = pd.read_excel("donnee1.xlsx")
print("Lecture terminé")


# In[4]:


#3.1 Affichage de la courbe de libération avec incertitudes

plt.figure(figsize=(10, 5))

# === IMPLANT A ===
df_x_IMU = df_brut_IMU[["Timestamp", "x"]].dropna()
plt.errorbar(
    df_x_IMU["Timestamp"], df_x_IMU["x"],
    fmt="-o", color="#004176", linewidth=2.5, capsize=4, elinewidth=1.5,
    label="X direction"
)

# === IMPLANT B ===
df_y_IMU = df_brut_IMU[["Timestamp", "y"]].dropna()
plt.errorbar(
    df_y_IMU["Timestamp"], df_y_IMU["y"],
    fmt="-X", color="#D55E00", linewidth=2.5, capsize=4, elinewidth=1.5,
    label="Y direction"
)

# === IMPLANT C ===
df_z_IMU = df_brut_IMU[["Timestamp", "z"]].dropna()
plt.errorbar(
    df_z_IMU["Timestamp"], df_z_IMU["z"],
    fmt="-s", color="#0072B2", linewidth=2, capsize=4, elinewidth=1.5,
    label="Z direction"
)

#---------------------------------------------

plt.title("Sensor Displacement", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Directions", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10, loc="best", frameon=True, shadow=True)
plt.show()


# In[ ]:




