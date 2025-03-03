from pathlib import Path
import requests
from tqdm import tqdm

# Optionally import gdown if available.
try:
    import gdown
except ImportError:
    gdown = None


def download_file(url: str, output_path: Path, use_gdown: bool = False) -> None:
    """
    Download a file from a given URL to the specified output path.
    
    Args:
        url (str): The URL of the file.
        output_path (Path): The full path (including filename) where the file will be saved.
        use_gdown (bool): If True and gdown is available, use it to download the file.
                          Otherwise, use requests with a tqdm progress bar.
    """
    # Ensure the destination directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if use_gdown and gdown:
        print(f"Downloading {url} to {output_path} using gdown...")
        # gdown will handle the download. Note: gdown's progress bar is basic.
        gdown.download(url, str(output_path), quiet=False)
    else:
        print(f"Downloading {url} to {output_path} using requests...")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            # Use tqdm to show a progress bar for the download.
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=output_path.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks.
                        f.write(chunk)
                        pbar.update(len(chunk))


def main():
    MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
    RVC_DOWNLOAD_LINK = 'https://huggingface.co/NeoPy/rvc-base/resolve/main/'
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
    rvc_models_dir = BASE_DIR / 'rvc_models'
    
    mdx_model_names = [
        'Kim_Vocal_2.onnx',
        'UVR_MDXNET_KARA_2.onnx',
        'Reverb_HQ_By_FoxJoy.onnx'
    ]
    
    rvc_model_names = [
        'rmvpe.pt',
        'fcpe.pt',
        'hubert_base.pt'
    ]
    
    # Set to True if you want to use gdown (and if your URL supports it).
    use_gdown = False
    
    for model in mdx_model_names:
        url = f"{MDX_DOWNLOAD_LINK}{model}"
        output_path = mdxnet_models_dir / model
        download_file(url, output_path, use_gdown=use_gdown)
    
    for model in rvc_model_names:
        url = f"{RVC_DOWNLOAD_LINK}{model}"
        output_path = rvc_models_dir / model
        download_file(url, output_path, use_gdown=use_gdown)


if __name__ == '__main__':
    main()
