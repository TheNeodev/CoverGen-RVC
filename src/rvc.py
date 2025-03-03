import logging
from multiprocessing import cpu_count
from pathlib import Path

import torch
from fairseq import checkpoint_utils
from scipy.io import wavfile

from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from my_utils import load_audio
from vc_infer_pipeline import VC

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rvc_infer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    def __init__(self, device, is_half):
        logger.info(f"Initializing Config with device: {device}, is_half: {is_half}")
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()
        logger.debug(f"Device configuration - x_pad: {self.x_pad}, x_query: {self.x_query}, x_center: {self.x_center}, x_max: {self.x_max}")

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            logger.info(f"Detected GPU: {self.gpu_name}")
            
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.warning("16 series/10 series P40 detected - forcing single precision")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    config_path = BASE_DIR / "src" / "configs" / config_file
                    logger.debug(f"Modifying config file: {config_path}")
                    with open(config_path, "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(config_path, "w") as f:
                        f.write(strr)
                
                pipeline_path = BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py"
                logger.debug(f"Modifying pipeline file: {pipeline_path}")
                with open(pipeline_path, "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(pipeline_path, "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
                
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            logger.info(f"GPU memory: {self.gpu_mem}GB")
            
            if self.gpu_mem <= 4:
                logger.warning("Low GPU memory detected - modifying pipeline configuration")
                pipeline_path = BASE_DIR / "src" / "trainset_preprocess_pipeline_print.py"
                with open(pipeline_path, "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open(pipeline_path, "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            logger.info("No NVIDIA GPU found, using MPS for inference")
            self.device = "mps"
        else:
            logger.info("No supported GPU found, using CPU for inference")
            self.device = "cpu"
            self.is_half = True

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()
            logger.debug(f"Using CPU cores: {self.n_cpu}")

        # Memory configuration
        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            logger.info("Adjusting memory configuration for low VRAM")
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

def load_hubert(device, is_half, model_path):
    logger.info(f"Loading Hubert model from {model_path}")
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path], suffix='',
    )
    hubert = models[0]
    hubert = hubert.to(device)
    
    if is_half:
        logger.debug("Using half precision for Hubert model")
        hubert = hubert.half()
    else:
        logger.debug("Using float precision for Hubert model")
        hubert = hubert.float()
    
    hubert.eval()
    logger.info("Hubert model loaded successfully")
    return hubert

def get_vc(device, is_half, config, model_path):
    logger.info(f"Loading voice model from {model_path}")
    try:
        cpt = torch.load(model_path, map_location='cpu')
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise
    
    if "config" not in cpt or "weight" not in cpt:
        logger.error(f"Invalid model format in {model_path} - missing config or weights")
        raise ValueError(f'Incorrect format for {model_path}. Use a voice model trained using RVC v2 instead.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    logger.info(f"Model parameters - Sample rate: {tgt_sr}, F0: {if_f0}, Version: {version}")

    net_g = None
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    
    if net_g is None:
        logger.error("Failed to initialize model architecture")
        raise RuntimeError("Failed to initialize model")

    logger.debug("Removing enc_q from model")
    del net_g.enc_q
    
    load_result = net_g.load_state_dict(cpt["weight"], strict=False)
    logger.info(f"Model load results: {load_result}")
    
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    
    vc = VC(tgt_sr, config)
    logger.info(f"Voice model loaded successfully. Target sample rate: {tgt_sr}")
    return cpt, version, net_g, tgt_sr, vc

def rvc_infer(
    index_path,
    index_rate,
    input_path,
    output_path,
    pitch_change,
    f0_method,
    cpt,
    version,
    net_g,
    filter_radius,
    tgt_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
    vc,
    hubert_model,
    f0autotune,
    f0_min=50,
    f0_max=1100
):
    logger.info(f"Starting inference pipeline - Input: {input_path}, Output: {output_path}")
    logger.debug(f"Inference parameters - Index rate: {index_rate}, Pitch change: {pitch_change}, "
                f"F0 method: {f0_method}, Filter radius: {filter_radius}, RMS mix rate: {rms_mix_rate}")
    
    try:
        audio = load_audio(input_path, 16000)
        logger.debug(f"Loaded audio with length: {len(audio)} samples")
    except Exception as e:
        logger.error(f"Failed to load audio from {input_path}: {e}")
        raise

    times = [0, 0, 0]
    if_f0 = cpt.get('f0', 1)
    
    try:
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            input_path,
            times,
            pitch_change,
            f0_method,
            index_path,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            0,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0autotune,
            f0_file=None,
            f0_min=f0_min,
            f0_max=f0_max
        )
        logger.info(f"Audio processing completed. Processing times: {times}")
    except Exception as e:
        logger.error(f"Error during audio processing: {e}")
        raise

    try:
        wavfile.write(output_path, tgt_sr, audio_opt)
        logger.info(f"Successfully wrote output to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        raise

    return output_path
