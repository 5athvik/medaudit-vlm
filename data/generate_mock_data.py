import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

os.makedirs("data/raw", exist_ok=True)

DRUGS = [
    ("Metformin", "500mg", "Twice daily"),
    ("Atorvastatin", "20mg", "At bedtime"),
    ("Amoxicillin", "250mg", "Every 8 hours"),
    ("Lisinopril", "10mg", "Once daily morning"),
    ("Levothyroxine", "50mcg", "Before breakfast"),
]

def make_sample(idx: int):
    img = Image.new("RGB", (600, 300), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    
    # Pick 1-2 drugs
    chosen = random.sample(DRUGS, k=random.choice([1, 2]))
    
    y = 30
    draw.text((30, y), f"Rx Record: MED-2026-00{idx}", fill=(30, 30, 30))
    y += 40
    
    label_items = []
    for drug, dose, freq in chosen:
        line_text = f"- {drug} | {dose} | {freq}"
        draw.text((35, y), line_text, fill=(20, 20, 70))
        y += 40
        label_items.append({"drug": drug, "dosage": dose, "frequency": freq})
    
    img_path = f"data/raw/rx_{idx}.png"
    img.save(img_path)
    return {"image": img_path, "ground_truth": label_items}

dataset = [make_sample(i) for i in range(50)]

with open("data/raw/dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} mock clinical samples in data/raw/")