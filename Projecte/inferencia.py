import sys
import time
import argparse
from pathlib import Path


def install(pkgs):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs, "-q"])

try:
    from ultralytics import YOLO
except ImportError:
    install(["ultralytics"]); from ultralytics import YOLO

try:
    import cv2
except ImportError:
    install(["opencv-python"]); import cv2

try:
    import numpy as np
except ImportError:
    install(["numpy"]); import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    install(["matplotlib"])
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# Cerca automàtica del millor model disponible
def trobar_model(model_arg: str) -> Path:
    if model_arg:
        p = Path(model_arg)
        if not p.exists():
            print(f"ERROR: No s'ha trobat el model: {p}")
            sys.exit(1)
        return p

    candidates = [
        Path("runs/train_run/weights/best.pt"),
        *sorted(Path(".").rglob("best.pt")),
    ]
    for c in candidates:
        if c.exists():
            print(f"  Model trobat automàticament: {c}")
            return c

    print("ERROR: No s'ha trobat cap model entrenat.")
    print("  Especifica'l amb: --model ruta/al/best.pt")
    print("  O entrena primer amb: python train.py --dataset ./dataset")
    sys.exit(1)


# Color dominant d'un patch (per pintar la bbox del color del globus)
def color_dominant(img, x1, y1, x2, y2, margin=6):
    h_img, w_img = img.shape[:2]
    # Aplica marge interior
    px1 = min(x1 + margin, w_img - 1)
    py1 = min(y1 + margin, h_img - 1)
    px2 = max(x2 - margin, px1 + 1)
    py2 = max(y2 - margin, py1 + 1)

    patch = img[py1:py2, px1:px2]
    if patch.size == 0:
        return (255, 255, 255)

    # Filtra píxels foscos (ombres) i molt clars (fons blanc)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mascara = (
        (hsv[:, :, 1] > 40) &   # saturació mínima (descarta grisos/blancs)
        (hsv[:, :, 2] > 50) &   # valor mínim (descarta negres)
        (hsv[:, :, 2] < 240)    # valor màxim (descarta blancs purs)
    )
    pixels = patch[mascara]

    if len(pixels) < 10:
        # Patch sense prou color: usa la mitjana de tot el patch
        pixels = patch.reshape(-1, 3)

    color_bgr = pixels.mean(axis=0).astype(int)
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))


def saturar_color(bgr, factor=1.5):
    pixel = np.array([[list(bgr)]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV).astype(float)
    hsv[0, 0, 1] = min(255, hsv[0, 0, 1] * factor)  # boost saturació
    hsv[0, 0, 2] = min(255, hsv[0, 0, 2] * 1.2)     # lleuger boost lluminositat
    bgr_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return (int(bgr_out[0, 0, 0]), int(bgr_out[0, 0, 1]), int(bgr_out[0, 0, 2]))


# Dibuixa bounding boxes del color del globus
def dibuixar_boxes(frame, boxes_data, gruix=3):
    for (x1, y1, x2, y2) in boxes_data:
        color_bgr = color_dominant(frame, x1, y1, x2, y2)
        color_bgr = saturar_color(color_bgr)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, gruix)
    return frame


# HUD: dibuixa stats sobre el frame (FPS, globus)
def dibuixar_hud(frame, n_globus, fps, mode=""):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 80), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color_globus = (50, 220, 120) if n_globus > 0 else (180, 180, 180)
    cv2.putText(frame, f"Globus: {n_globus}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_globus, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    if mode:
        cv2.putText(frame, mode, (w - 160, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 180, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q/ESC: sortir  P: pausa  G: captura", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    return frame


# MODE VIDEO / WEBCAM
def processar_stream(model, font, conf, iou, output_dir, guardar, es_webcam):
    cap = cv2.VideoCapture(font)
    if not cap.isOpened():
        msg = f"camara {font}" if isinstance(font, int) else f"video '{font}'"
        print(f"ERROR: No s'ha pogut obrir la {msg}.")
        if isinstance(font, int):
            print("  Prova amb --camera 0 o --camera 1.")
        sys.exit(1)

    fps_src   = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not es_webcam else -1
    w_src     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mode_txt  = "WEBCAM" if es_webcam else "VIDEO"

    print(f"\n  Resolucio  : {w_src}x{h_src}")
    if not es_webcam:
        print(f"  FPS font   : {fps_src:.1f}")
        print(f"  Frames     : {total_frm}")
    print(f"\n  Iniciant finestra — {mode_txt}...")
    print("  Controls: Q/ESC = sortir | P = pausa | G = captura\n")

    writer = None
    if guardar and not es_webcam:
        nom_sortida = output_dir / f"detectat_{Path(str(font)).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(nom_sortida), fourcc, fps_src, (w_src, h_src))
        print(f"  Guardant video a: {nom_sortida}")

    n_frame    = 0
    n_captures = 0
    t_inici    = time.time()
    fps_calc   = 0.0
    pausat     = False
    stats_globus = []
    frame_mostrat = None

    while True:
        if not pausat:
            ret, frame = cap.read()
            if not ret:
                if not es_webcam:
                    print("\n  Fi del video.")
                break

            n_frame += 1

            t0 = time.time()
            resultats = model.predict(frame, conf=conf, iou=iou, verbose=False)
            fps_calc = 1.0 / (time.time() - t0 + 1e-9)

            r = resultats[0]
            n_globus = len(r.boxes)
            stats_globus.append(n_globus)

            # Extrau les caixes en píxels enters
            boxes_px = []
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                boxes_px.append((x1, y1, x2, y2))

            # Dibuixa les bbox del color del globus, sense text
            frame_anotat = frame.copy()
            dibuixar_boxes(frame_anotat, boxes_px)
            estat = "PAUSAT" if pausat else mode_txt
            frame_anotat = dibuixar_hud(frame_anotat, n_globus, fps_calc, estat)

            if writer:
                writer.write(frame_anotat)

            frame_mostrat = frame_anotat

        if frame_mostrat is not None:
            cv2.imshow(f"Detector de Globus — {mode_txt}", frame_mostrat)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("\n  Sortint...")
            break
        elif key in (ord("p"), ord("P")) and not es_webcam:
            pausat = not pausat
            print(f"  {'PAUSAT' if pausat else 'REPRENENT'}...")
        elif key in (ord("g"), ord("G")) and frame_mostrat is not None:
            n_captures += 1
            nom = output_dir / f"captura_{n_captures:04d}.jpg"
            cv2.imwrite(str(nom), frame_mostrat)
            print(f"  Captura guardada: {nom}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_inici
    fps_mig = n_frame / elapsed if elapsed > 0 else 0
    print(f"\n{'='*50}")
    print(f"  RESUM {mode_txt}")
    print(f"  Frames processats  : {n_frame}")
    print(f"  FPS mitja real     : {fps_mig:.1f}")
    print(f"  Captures guardades : {n_captures}")
    if stats_globus:
        print(f"  Max. globus/frame  : {max(stats_globus)}")
        print(f"  Mitjana globus     : {sum(stats_globus)/len(stats_globus):.2f}")
    print(f"{'='*50}\n")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Detector de globus — imatges, video i webcam",
        formatter_class=argparse.RawTextHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--video",   type=str,
                       help="Ruta a un fitxer de video (mp4, avi, ...)")
    group.add_argument("--webcam",  action="store_true",
                       help="Usa la webcam en temps real")

    parser.add_argument("--camera", type=int, default=0,
                        help="Index de la camara (default: 0)")
    parser.add_argument("--model",  type=str, default=None,
                        help="Ruta al best.pt (cerca automatica si no s'indica)")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="Llindar de confianca (default: 0.25)")
    parser.add_argument("--iou",    type=float, default=0.45,
                        help="Llindar IoU per NMS (default: 0.45)")
    parser.add_argument("--guardar", action="store_true",
                        help="Guarda imatges/video amb les deteccions")
    parser.add_argument("--output",  type=str, default="inferencia_output",
                        help="Carpeta de sortida (default: inferencia_output)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  DETECTOR DE GLOBUS — Inferencia")
    print("=" * 50)

    model_path = trobar_model(args.model)
    model = YOLO(str(model_path))

    import torch
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"  Dispositiu : {device}")
    print(f"  Model      : {model_path}")
    print(f"  Confianca  : {args.conf}")
    print(f"  IoU (NMS)  : {args.iou}")
    print("=" * 50)

    # Webcam
    if args.webcam:
        print(f"\n  Mode: WEBCAM (camara {args.camera})")
        processar_stream(model, args.camera, args.conf, args.iou,
                         output_dir, args.guardar, es_webcam=True)

    # Video
    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: No s'ha trobat el video: {video_path}")
            sys.exit(1)
        print(f"\n  Mode: VIDEO -> {video_path.name}")
        processar_stream(model, str(video_path), args.conf, args.iou,
                         output_dir, args.guardar, es_webcam=False)

   

if __name__ == "__main__":
    main()
