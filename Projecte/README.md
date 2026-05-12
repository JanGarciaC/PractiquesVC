# Detector de Globus — YOLOv8

Script d'entrenament i avaluació d'un model de detecció de globus
basat en **YOLOv8** d'Ultralytics.

---

## Estructura del dataset esperada

```
dataset/
├── train/
│   ├── images/       ← imatges d'entrenament (.jpg / .png)
│   └── labels/       ← anotacions YOLO (.txt)
└── valid/
    ├── images/       ← imatges de validació
    └── labels/       ← anotacions YOLO
```

### Format dels fitxers de labels (YOLO)

Cada línia d'un `.txt` representa un globus:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id` = sempre `0` (única classe: balloon)
- Totes les coordenades estan **normalitzades** entre 0 i 1
- Si la imatge **no té globus**, el fitxer `.txt` és **buit**

Exemple per a una imatge amb 2 globus:
```
0 0.512 0.340 0.198 0.321
0 0.801 0.210 0.145 0.287
```

---

## Instal·lació

```bash
pip install ultralytics matplotlib Pillow
```

---

## Ús

### Entrenament + avaluació (configuració per defecte)

```bash
python train.py --dataset /ruta/al/dataset
```

### Opcions principals

| Argument | Default | Descripció |
|----------|---------|------------|
| `--dataset` | `dataset` | Ruta a la carpeta arrel del dataset |
| `--output`  | `runs`    | On es guarden els pesos i gràfiques |
| `--epochs`  | `50`      | Nombre d'epochs |
| `--imgsz`   | `640`     | Mida de les imatges (píxels) |
| `--batch`   | `16`      | Batch size (`-1` = automàtic) |
| `--model`   | `n`       | Mida del model: `n` nano · `s` small · `m` medium · `l` large · `x` |
| `--inference` | off    | Inferència de mostra sobre imatges de validació |

### Exemples

```bash
# Entrenament ràpid amb model nano (ideal per provar)
python train.py --dataset ./dataset --epochs 30 --model n

# Entrenament amb model mitjà, més epochs, amb mostra d'inferència
python train.py --dataset ./dataset --epochs 100 --model m --inference

# Dataset en una altra ruta, resultats en carpeta custom
python train.py --dataset /home/user/globus_data --output /home/user/resultats
```

---

## Resultats generats

```
runs/
├── data.yaml               ← configuració del dataset
├── train_run/
│   ├── weights/
│   │   ├── best.pt         ← millors pesos (usar per inferència)
│   │   └── last.pt         ← últims pesos
│   ├── results.png         ← gràfiques loss / mètriques per epoch
│   ├── confusion_matrix.png
│   └── PR_curve.png
└── eval_run/               ← resultats d'avaluació final
    └── ...
```

### Mètriques reportades

- **mAP50** — mean Average Precision a IoU=0.50
- **mAP50-95** — mAP promig a IoU de 0.50 a 0.95
- **Precisió** — de les deteccions fetes, quantes són correctes
- **Recall** — de tots els globus reals, quants s'han detectat

---

## Inferència sobre imatges noves

Un cop entrenat, pots fer servir els pesos directament:

```python
from ultralytics import YOLO

model = YOLO("runs/train_run/weights/best.pt")
results = model.predict("nova_imatge.jpg", conf=0.25)
results[0].show()          # mostra la imatge amb bounding boxes
```

---

## Consells

| Situació | Recomanació |
|----------|-------------|
| Dataset petit (<200 imgs) | `--model n` o `--model s`, `--epochs 100` |
| Dataset mitjà (200-1000) | `--model s` o `--model m`, `--epochs 50-100` |
| GPU disponible | El script la detecta automàticament |
| Sense GPU (CPU) | Usa `--model n` i `--batch 8` per velocitat |
| Overfitting (val loss puja) | Redueix epochs, usa `--model n` |
