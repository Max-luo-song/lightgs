from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse

import torch, numpy as np, subprocess
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    save_images as safe_save_images,
)

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def images_to_video(image_path, output_path, fps=10):
    """
    ffmpeg -framerate 10 -start_number 1 -i ./output/qcraft_20250902_163634_Q3703_1035_1055/20251216_lidar+cam0_1_2_3_5_6_7_9_10_11_12_test_with_novelview_2/videos_eval/full_set_40000_depths_layout/%06d.png -vf "scale=ceil(iw/2)*2:ceil(ih/2)*2:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2" -c:v libx264 -pix_fmt yuv420p ./output/qcraft_20250902_163634_Q3703_1035_1055/20251216_lidar+cam0_1_2_3_5_6_7_9_10_11_12_test_with_novelview_2/full_set_40000_depths.mp4
    """
    cmd = [
        "ffmpeg -framerate",
        str(fps),
        "-start_number 1 -i",
        os.path.join(image_path, "%06d.png"),
        '-vf "scale=ceil(iw/2)*2:ceil(ih/2)*2:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2"',
        "-c:v libx264 -pix_fmt yuv420p -y",
        output_path,
    ]
    os.system(" ".join(cmd))


@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True,
):
    trainer.set_eval()

    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:
        logger.info("Evaluating Test Set Pixels...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )

        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "psnr_no_ego",
                    "ssim_no_ego",
                    "lpips_no_ego",
                    "psnr_with_ego",
                    "ssim_with_ego",
                    "lpips_with_ego",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/test/{k}"] = v
            if args.enable_wandb:
                wandb.log(eval_dict)
            test_metrics_file = (
                f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            )
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"

        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_test_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=2,
            verbose=True,
            save_images=False,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/test/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()

    if cfg.render.render_full:
        logger.info("Evaluating Full Set...")

        if args.render_video_postfix is None:
            video_output_pth = f"./{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = f"./{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"

        all_metrics = [
            "psnr",
            "ssim",
            "lpips",
            "psnr_no_ego",
            "ssim_no_ego",
            "lpips_no_ego",
            "psnr_with_ego",
            "ssim_with_ego",
            "lpips_with_ego",
            "occupied_psnr",
            "occupied_ssim",
            "masked_psnr",
            "masked_ssim",
            "human_psnr",
            "human_ssim",
            "vehicle_psnr",
            "vehicle_ssim",
        ]
        eval_dict = dict(zip(all_metrics, [[] for _ in range(len(all_metrics))]))

        for i in range(0, dataset.num_img_timesteps):
            print(f"crrent render {i} batch, total length {dataset.num_img_timesteps}")
            vis_timestep = np.arange(0, dataset.num_img_timesteps * dataset.num_cams)[
                i * dataset.num_cams : (i + 1) * dataset.num_cams
            ]
            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=vis_timestep,
                )

                if log_metrics:
                    for k, v in render_results.items():
                        if k in all_metrics:
                            eval_dict[k] += [v]

                vis_frame_dict = safe_save_images(
                    render_results,
                    video_output_pth,
                    layout=dataset.layout,
                    timestamps=i,
                    keys=render_keys,
                    verbose=True,
                )

                if args.enable_wandb:
                    for k, v in vis_frame_dict.items():
                        wandb.log({"image_rendering/full/" + k: wandb.Image(v)})

        torch.cuda.empty_cache()

        for dirs in os.listdir(f"{cfg.log_dir}/videos{post_fix}"):
            if (
                "layout" in dirs
                and len(os.listdir(f"{cfg.log_dir}/videos{post_fix}/{dirs}")) > 0
            ):
                img_dir = f"{cfg.log_dir}/videos{post_fix}/{dirs}"
                save_mp4 = f"{cfg.log_dir}/videos{post_fix}/{dirs}.mp4"
                images_to_video(img_dir, save_mp4, cfg.render.fps)

        fine_eval_dict = dict()
        if log_metrics:
            for k, v in eval_dict.items():
                if k in all_metrics:
                    fine_eval_dict[f"image_metrics/full/{k}"] = float(
                        sum(eval_dict[k]) / len(eval_dict[k])
                        if len(eval_dict[k]) > 0
                        else -1
                    )

        if args.enable_wandb:
            wandb.log(fine_eval_dict)
        full_metrics_file = (
            f"{cfg.log_dir}/metrics{post_fix}/images_full_{current_time}.json"
        )
        with open(full_metrics_file, "w") as f:
            json.dump(fine_eval_dict, f)
        logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )

    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True,
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")

    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False

    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix="_eval",
    )

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    # eval
    parser.add_argument(
        "--resume_from",
        default=None,
        help="path to checkpoint to resume from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--render_video_postfix",
        type=str,
        default=None,
        help="an optional postfix for video",
    )
    parser.add_argument(
        "--save_catted_videos",
        type=bool,
        default=False,
        help="visualize lidar on image",
    )

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    main(args)
