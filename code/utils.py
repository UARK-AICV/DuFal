import yaml
import numpy as np
import SimpleITK as sitk
from easydict import EasyDict
from tabulate import tabulate
from pprint import pprint
import subprocess
import json
import shutil
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
import os
import torch
from collections import defaultdict
import glob

def save_checkpoint(save_dir, epoch, model, optimizer, lr_scheduler, val_psnr, val_ssim, is_last=False):
    """Save a checkpoint with the specified format and return its filename."""
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_psnr': val_psnr,
        'val_ssim': val_ssim
    }
    if lr_scheduler is not None:
        checkpoint['lr_scheduler'] = lr_scheduler.state_dict()
    
    if is_last:
        filename = f'ep_{epoch}.pth'
    else:
        # Format filenames for best metrics
        filename_psnr = f'ep_{epoch}_best_vpsnr_{val_psnr:.6f}_vssim_{val_ssim:.6f}.pth'
        filename_ssim = f'ep_{epoch}_vpsnr_{val_psnr:.6f}_best_vssim_{val_ssim:.6f}.pth'
        # Save for each category
        for fname in [filename_psnr, filename_ssim]:
            torch.save(checkpoint, os.path.join(save_dir, fname))
        return [filename_psnr, filename_ssim]
    
    torch.save(checkpoint, os.path.join(save_dir, filename))
    return filename

def manage_checkpoints(save_dir, epoch, model, optimizer, lr_scheduler, val_psnr, val_ssim, k=2):
    """Manage top k checkpoints for PSNR and SSIM independently, preserving the best metrics."""
    # Lists to track checkpoints
    best_psnr = []
    best_ssim = []
    last_checkpoint = None
    
    # Load existing checkpoints
    checkpoint_files = glob.glob(os.path.join(save_dir, '*.pth'))
    for fname in checkpoint_files:
        try:
            ckpt = torch.load(fname, map_location=torch.device('cpu'), weights_only=False)
            metrics = {
                'epoch': ckpt['epoch'],
                'val_psnr': ckpt['val_psnr'],
                'val_ssim': ckpt['val_ssim'],
                'filename': os.path.basename(fname)
            }
            if fname.endswith(f'ep_{ckpt["epoch"]}.pth'):
                last_checkpoint = metrics
            else:
                if 'best_vpsnr' in fname:
                    best_psnr.append(metrics)
                if 'best_vssim' in fname:
                    best_ssim.append(metrics)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {fname}: {str(e)}")
    
    # Find the current best PSNR and SSIM, excluding last checkpoint
    best_psnr_value = max([ckpt['val_psnr'] for ckpt in best_psnr], default=0.0)
    best_ssim_value = max([ckpt['val_ssim'] for ckpt in best_ssim], default=0.0)
    
    # Save new checkpoint only if it improves on the best metrics
    filenames = []
    new_metrics = {'epoch': epoch, 'val_psnr': val_psnr, 'val_ssim': val_ssim}
    
    if val_psnr > best_psnr_value:
        saved_filenames = save_checkpoint(save_dir, epoch, model, optimizer, lr_scheduler, val_psnr, val_ssim, is_last=False)
        new_metrics['filename'] = saved_filenames[0]  # PSNR filename
        best_psnr.append(new_metrics.copy())
        print(f"New best PSNR: {val_psnr:.6f} at epoch {epoch}")
        filenames.extend(saved_filenames)
    
    if val_ssim > best_ssim_value and val_ssim not in [ckpt['val_ssim'] for ckpt in best_ssim]:
        if not filenames:  # Avoid duplicate save if already saved for PSNR
            saved_filenames = save_checkpoint(save_dir, epoch, model, optimizer, lr_scheduler, val_psnr, val_ssim, is_last=False)
            new_metrics['filename'] = saved_filenames[1]  # SSIM filename
            best_ssim.append(new_metrics.copy())
            print(f"New best SSIM: {val_ssim:.6f} at epoch {epoch}")
            filenames.extend(saved_filenames)
    
    # Sort and keep top k for each metric
    best_psnr = sorted(best_psnr, key=lambda x: x['val_psnr'], reverse=True)[:k]
    best_ssim = sorted(best_ssim, key=lambda x: x['val_ssim'], reverse=True)[:k]
    
    # Ensure all top k checkpoints are preserved
    valid_filenames = {ckpt['filename'] for ckpt in best_psnr + best_ssim}
    if last_checkpoint:
        valid_filenames.add(last_checkpoint['filename'])
    
    # Delete obsolete checkpoints
    for fname in checkpoint_files:
        if os.path.basename(fname) not in valid_filenames:
            try:
                os.remove(fname)
            except Exception as e:
                raise RuntimeError(f"Failed to delete obsolete checkpoint {fname}: {str(e)}")
    
    # Save the last checkpoint (overwrites previous last checkpoint)
    last_filename = save_checkpoint(save_dir, epoch, model, optimizer, lr_scheduler, val_psnr, val_ssim, is_last=True)
    valid_filenames.add(last_filename)
    filenames.append(last_filename)
    
    return valid_filenames

def get_git_info():
    """Retrieve the current Git branch and short commit hash."""
    try:
        # Get the current branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True
        ).strip()

        # Get the short-hand commit hash (first 7 characters)
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()

        time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

        return {"branch": branch, "commit": commit, "time_str": time_str}
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to get Git info: {e}"}
    except FileNotFoundError:
        return {"error": "Git is not installed or not found in PATH"}

def display_model_info(info, save_path=None):
    """
    Display model information in a formatted way and optionally save to a file.
    
    Args:
        info (dict): Dictionary from get_model_info containing model metrics.
        save_path (str, optional): Path to save the output as a text or JSON file.
    """
    # Extract key metrics for tabular display
    table_data = [
        ["Total Parameters", f"{info['total_params']:,}"],
        ["Learnable Parameters", f"{info['learnable_params']:,}"],
        ["Non-Learnable Parameters", f"{info['non_learnable_params']:,}"],
        ["FLOPs (GFLOPs)", f"{info['flops_gflops']:.2f}"],
        ["Memory (MB)", f"{info['memory_mb']:.2f}"]
    ]
    
    # Print table
    print("\n=== Model Information ===")
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
    
    # # Print dictionary using pprint for structured view
    # print("\n=== Detailed Metrics ===")
    # pprint({k: v for k, v in info.items() if k != "summary"}, indent=2)
    
    # # Print summary header (first few lines of torchinfo summary)
    # print("\n=== Model Summary (Preview) ===")
    # summary_lines = info["summary"].split("\n")
    # print("\n".join(summary_lines[:10]))  # Show first 10 lines
    # if len(summary_lines) > 10:
    #     print("... (full summary truncated, see saved file or increase preview limit)")
    
    # Save to file if save_path is provided
    if save_path:
        if save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump(info, f, indent=4)
            print(f"\nSaved model info as JSON to {save_path}")
        else:
            with open(save_path, "w") as f:
                f.write("=== Model Information ===\n")
                f.write(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
                f.write("\n\n=== Detailed Metrics ===\n")
                f.write(json.dumps({k: v for k, v in info.items() if k != "summary"}, indent=2))
                f.write("\n\n=== Model Summary ===\n")
                f.write(info["summary"])
            print(f"\nSaved model info as text to {save_path}")



def load_config(path):
    cfg = OmegaConf.load(path)  # Load YAML directly into OmegaConf
    return cfg

def convert_cuda(item):
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda()
    return item


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def sitk_load(path, uint8=False, spacing_unit='mm'):
    # load as float32
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    origin = np.array(itk_img.GetOrigin(), dtype=np.float32)
    if spacing_unit == 'm':
        spacing *= 1000.
        origin *= 1000
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(np.float32)
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]
        image /= 255.
    return image, spacing, origin


def sitk_save(path, image, spacing=None, origin=None, uint8=False):
    # default: float32 (input)
    image = image.astype(np.float32)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    if origin is not None:
        out.SetOrigin(origin.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)
