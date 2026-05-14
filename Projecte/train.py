import os
import sys
import argparse
from pathlib import Path

# Instal·lació automàtica de dependències
def install_dependencies():
    import subprocess
    print("Instal·lant dependències...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "ultralytics", "matplotlib", "Pillow", "-q"])
    print("Dependències instal·lades.\n")


# Generació del fitxer de configuració YAML
def create_yaml(dataset_root: Path, output_dir: Path) -> Path:
    yaml_content = f"""# Configuració del dataset de globus
path: {dataset_root.resolve()}
train: train/images
val:   valid/images
nc: 1          # nombre de classes
names:
  - balloon    # única classe: globus
"""
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Fitxer de configuració creat: {yaml_path}\n")
    return yaml_path


# Entrenament
def train(yaml_path: Path, output_dir: Path, epochs: int, imgsz: int, batch: int, model_size: str):
    import io
    import logging
    import contextlib
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.INFO)

    base_model = f"yolov8{model_size}.pt"
    print(f"Model base: {base_model}")
    print(f"Epochs: {epochs} | Imgsz: {imgsz} | Batch: {batch}")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = YOLO(base_model)

    for line in buf.getvalue().splitlines():
        if "Model summary" in line:
            print(line.strip())
            break
    print()

    print("Entrenant...", flush=True)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(output_dir),
        name="train_run",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        verbose=True)
    print("Entrenament completat!")
    return results


# Avaluació sobre el conjunt de validació
def evaluate(output_dir: Path, yaml_path: Path, imgsz: int):
    import io
    import contextlib
    from ultralytics import YOLO

    best_weights = output_dir / "train_run" / "weights" / "best.pt"
    if not best_weights.exists():
        print("ERROR: No s'han trobat els pesos entrenats a:", best_weights)
        return

    print(f"\n{'='*60}")
    print("AVALUACIÓ SOBRE EL CONJUNT DE VALIDACIÓ")
    print(f"{'='*60}")
    print(f"Pesos: {best_weights}")
    print("Avaluant...", flush=True)

    model = YOLO(str(best_weights))
    metrics = model.val(
        data=str(yaml_path),
        imgsz=imgsz,
        split="val",
        plots=True,
        save_json=True,
        project=str(output_dir),
        name="eval_run",
        exist_ok=True,
        verbose=False)

    print("\n--- Resultats ---")
    print(f"  mAP50       : {metrics.box.map50:.4f}")
    print(f"  mAP50-95    : {metrics.box.map:.4f}")
    print(f"  Precisió    : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print(f"{'='*60}\n")

    return metrics

# Main
def main():
    parser = argparse.ArgumentParser(description="Entrenament detector de globus YOLOv8")
    parser.add_argument("--dataset",   type=str, default="dataset",
                        help="Ruta a la carpeta arrel del dataset (conté train/ i valid/)")
    parser.add_argument("--output",    type=str, default="runs",
                        help="Carpeta on es guarden els resultats")
    parser.add_argument("--epochs",    type=int, default=50,
                        help="Nombre d'epochs d'entrenament")
    parser.add_argument("--imgsz",     type=int, default=640,
                        help="Mida de les imatges durant entrenament/validació")
    parser.add_argument("--batch",     type=int, default=16,
                        help="Batch size (-1 = automàtic)")
    parser.add_argument("--model",     type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="Mida del model YOLOv8: n(ano), s(mall), m(edium), l(arge), x")

    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validació de l'estructura del dataset
    for split in ("train", "valid"):
        for sub in ("images", "labels"):
            d = dataset_root / split / sub
            if not d.exists():
                print(f"ERROR: No s'ha trobat la carpeta esperada: {d}")
                sys.exit(1)

    print("="*60)
    print("  DETECTOR DE GLOBUS — YOLOv8")
    print("="*60)
    print(f"  Dataset : {dataset_root.resolve()}")
    print(f"  Output  : {output_dir.resolve()}")
    print("="*60 + "\n")

    # Estadístiques ràpides del dataset
    for split in ("train", "valid"):
        imgs = list((dataset_root / split / "images").glob("*.[jp][pn]g")) + \
               list((dataset_root / split / "images").glob("*.jpeg"))
        lbls = list((dataset_root / split / "labels").glob("*.txt"))
        empty = sum(1 for l in lbls if l.stat().st_size == 0)
        print(f"  {split:5s}: {len(imgs)} imatges | {len(lbls)} labels "
              f"| {empty} imatges sense globus")
    print()

    yaml_path = create_yaml(dataset_root, output_dir)
    train(yaml_path, output_dir, args.epochs, args.imgsz, args.batch, args.model)
    evaluate(output_dir, yaml_path, args.imgsz)

    print("\nFinalitzat! Revisa els resultats a:", output_dir.resolve())


if __name__ == "__main__":
    install_dependencies()
    main()
