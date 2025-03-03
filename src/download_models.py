import gdown
from pathlib import Path

# Base download links
MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_DOWNLOAD_LINK = 'https://huggingface.co/NeoPy/rvc-base/resolve/main/'

# Set up directories relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
rvc_models_dir = BASE_DIR / 'rvc_models'

# Create directories if they do not exist
mdxnet_models_dir.mkdir(exist_ok=True)
rvc_models_dir.mkdir(exist_ok=True)

def dl_model(link, model_name, dir_name):
    # Compose the full URL and output path
    url = f'{link}{model_name}'
    output = str(dir_name / model_name)
    # gdown.download automatically displays a progress bar when quiet=False
    gdown.download(url, output, quiet=False)

if __name__ == '__main__':
    mdx_model_names = [
        'Kim_Vocal_2.onnx', 
        'UVR_MDXNET_KARA_2.onnx', 
        'Reverb_HQ_By_FoxJoy.onnx'
    ]
    for model in mdx_model_names:
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    rvc_names = ['rmvpe.pt', 'fcpe.pt', 'hubert_base.pt']
    for model in rvc_names:
        dl_model(RVC_DOWNLOAD_LINK, model, rvc_models_dir)
