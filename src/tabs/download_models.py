import os
import sys
import shutil
import urllib.request
import zipfile
import gdown
from argparse import ArgumentParser

import gradio as gr

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVC_MODELS_DIR = os.path.join(BASE_DIR, 'rvc_models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'song_output')

# ----------------------------
# Helper Functions for Models
# ----------------------------
def get_current_models(models_dir: str) -> list:
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'rmvpe.pt', 'fcpe.pt']
    return [item for item in os.listdir(models_dir) if item not in items_to_remove]



def extract_zip(extraction_folder: str, zip_name: str):
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)
    index_filepath, model_filepath = None, None
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            full_path = os.path.join(root, name)
            if name.endswith('.index') and os.stat(full_path).st_size > 1024 * 100:
                index_filepath = full_path
            if name.endswith('.pth') and os.stat(full_path).st_size > 1024 * 1024 * 40:
                model_filepath = full_path
    if not model_filepath:
        raise gr.Error(f'No .pth model file found in {extraction_folder}. Please check the zip contents.')
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))
    for item in os.listdir(extraction_folder):
        item_path = os.path.join(extraction_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

def download_online_model(url: str, dir_name: str, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model: {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(RVC_MODELS_DIR, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Model directory {dir_name} already exists! Please choose a different name.')
        if 'huggingface.co' in url:
            urllib.request.urlretrieve(url, zip_name)
        elif 'pixeldrain.com' in url:
            zip_name = f'{dir_name}.zip'
            url = f'https://pixeldrain.com/api/file/{zip_name}'
            urllib.request.urlretrieve(url, zip_name)
        elif 'drive.google.com' in url:
            zip_name = f'{dir_name}.zip'
            file_id = url.split('/')[-2]
            output = os.path.join('.', zip_name)
            gdown.download(id=file_id, output=output, quiet=False)
        progress(0.5, desc='[~] Extracting model zip...')
        extract_zip(extraction_folder, zip_name)
        print(f'[+] Model {dir_name} successfully downloaded!')
        return f'[+] Model {dir_name} successfully downloaded!'
    except Exception as e:
        raise gr.Error(str(e))

def upload_local_model(zip_path, dir_name: str, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(RVC_MODELS_DIR, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Model directory {dir_name} already exists! Please choose a different name.')
        zip_name = zip_path.name
        progress(0.5, desc='[~] Extracting uploaded model zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Model {dir_name} successfully uploaded!'
    except Exception as e:
        raise gr.Error(str(e))


# ----------------------------
# UI Component Builders
# ----------------------------
def create_download_models_tab():
    with gr.TabItem("Download Models"):
        with gr.Row():
            url_mod = gr.Text(label="Model URL", placeholder="Enter model URL...")
            mod_name = gr.Text(label="Model Name", placeholder="Enter model name...")
        download_btn = gr.Button("Download Model", variant='primary')
        download_btn.click(download_online_model, inputs=[url_mod, mod_name], outputs=None)

