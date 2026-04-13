import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import DartHandPoseDataset
from utils.forwardpass import get_EvalFunction
from utils.utils import model_builder


@torch.no_grad()
def run_eval(
    model,
    loader,
    eval_function,
    setting,
    device,
    image_output_dir,
):
    model.eval()

    all_rows = []
    total_error = 0.0
    total_points = 0

    for batch_idx, data in enumerate(loader):
        (
            inputs,
            _gt2Dcrop,
            gt2Dorignal,
            _gt3Dorignal,
            com,
            _M_inv,
            cubesize,
            *_rest,
        ) = data

        inputs = inputs.to(device)
        gt2Dorignal = gt2Dorignal.to(device)
        com = com.to(device)
        cubesize = cubesize.to(device)

        outputs = model(inputs)
        preds = eval_function(inputs, outputs, cubesize, com, setting)

        preds_np = preds.detach().cpu().numpy()
        gts_np = gt2Dorignal.detach().cpu().numpy()
        imgs_np = inputs.detach().cpu().numpy()

        batch_start = batch_idx * loader.batch_size

        for sample_in_batch in range(preds_np.shape[0]):
            dataset_index = batch_start + sample_in_batch
            mapped_index = loader.dataset.indeces[dataset_index]
            raw_sample = loader.dataset.dataset.samples[mapped_index]
            image_name = Path(raw_sample["image_path"]).stem

            pred_uvd = preds_np[sample_in_batch]
            gt_uvd = gts_np[sample_in_batch]

            sample_err = np.linalg.norm(pred_uvd - gt_uvd, axis=1)
            total_error += float(sample_err.sum())
            total_points += int(sample_err.shape[0])

            all_rows.append(
                {
                    "sample_index": dataset_index,
                    "image_path": raw_sample["image_path"],
                    "pred_uvd": pred_uvd.copy(),
                    "gt_uvd": gt_uvd.copy(),
                    "mean_l2_error": float(sample_err.mean()),
                }
            )

            rendered = render_overlay(imgs_np[sample_in_batch, 0], pred_uvd, gt_uvd)
            out_path = image_output_dir / f"{dataset_index:06d}_{image_name}.png"
            cv2.imwrite(str(out_path), rendered)

    mean_l2 = total_error / max(total_points, 1)
    return all_rows, mean_l2


def render_overlay(input_image, pred_uvd, gt_uvd):
    img = ((input_image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for joint_idx in range(gt_uvd.shape[0]):
        gx, gy = int(round(gt_uvd[joint_idx, 0])), int(round(gt_uvd[joint_idx, 1]))
        px, py = int(round(pred_uvd[joint_idx, 0])), int(round(pred_uvd[joint_idx, 1]))

        cv2.circle(canvas, (gx, gy), 3, (0, 255, 0), thickness=-1)
        cv2.circle(canvas, (px, py), 3, (0, 0, 255), thickness=-1)
        cv2.line(canvas, (gx, gy), (px, py), (255, 0, 0), thickness=1)

    cv2.putText(
        canvas,
        "GT=green Pred=red",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_predictions(rows, output_csv, output_npz):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["sample_index", "image_path", "mean_l2_error"]
        for j in range(21):
            header.extend([f"pred_u_{j}", f"pred_v_{j}", f"pred_d_{j}"])
        for j in range(21):
            header.extend([f"gt_u_{j}", f"gt_v_{j}", f"gt_d_{j}"])
        writer.writerow(header)

        for row in rows:
            rec = [row["sample_index"], row["image_path"], row["mean_l2_error"]]
            rec.extend(row["pred_uvd"].reshape(-1).tolist())
            rec.extend(row["gt_uvd"].reshape(-1).tolist())
            writer.writerow(rec)

    np.savez_compressed(
        output_npz,
        sample_index=np.array([r["sample_index"] for r in rows], dtype=np.int64),
        image_path=np.array([r["image_path"] for r in rows]),
        pred_uvd=np.stack([r["pred_uvd"] for r in rows], axis=0),
        gt_uvd=np.stack([r["gt_uvd"] for r in rows], axis=0),
        mean_l2_error=np.array([r["mean_l2_error"] for r in rows], dtype=np.float32),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DART dataset, save predictions, and dump per-image overlays."
    )
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .pt checkpoint")
    parser.add_argument("--datasetpath", required=True, type=str, help="Path to DART root")
    parser.add_argument("--output_dir", default="dart_eval_outputs", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cuda_id", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    setting = ckpt["args"]
    setting.dataset = "dart"

    device = torch.device(
        f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    )

    dataset = DartHandPoseDataset(
        train=False,
        basepath=args.datasetpath,
        cropSize=(setting.cropSize, setting.cropSize),
        cropSize3D=[setting.cubic_size, setting.cubic_size, setting.cubic_size],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = model_builder(
        setting.model_name, num_joints=dataset.num_joints, args=setting
    ).to(device)
    model.load_state_dict(ckpt["model"])

    eval_function = get_EvalFunction(setting)

    output_dir = Path(args.output_dir)
    overlays_dir = output_dir / "overlay_images"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows, mean_l2 = run_eval(
        model=model,
        loader=loader,
        eval_function=eval_function,
        setting=setting,
        device=device,
        image_output_dir=overlays_dir,
    )

    csv_path = output_dir / "predictions.csv"
    npz_path = output_dir / "predictions.npz"
    save_predictions(rows, csv_path, npz_path)

    print(f"Saved {len(rows)} predictions")
    print(f"CSV: {csv_path}")
    print(f"NPZ: {npz_path}")
    print(f"Overlay images: {overlays_dir}")
    print(f"Mean per-joint L2 error in UVD space: {mean_l2:.4f}")


if __name__ == "__main__":
    main()
