import json
import os
import sys
import shutil
import urllib.request
import zipfile
import gdown
from argparse import ArgumentParser

import gradio as gr
from main import song_cover_pipeline

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

# Helper functions for models
def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'rmvpe.pt', 'fcpe.pt']
    return [item for item in models_list if item not in items_to_remove]

def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.update(choices=models_l)

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            full_path = os.path.join(root, name)
            if name.endswith('.index') and os.stat(full_path).st_size > 1024 * 100:
                index_filepath = full_path
            if name.endswith('.pth') and os.stat(full_path).st_size > 1024 * 1024 * 40:
                model_filepath = full_path

    if not model_filepath:
        raise gr.Error(f'No .pth model file was found in {extraction_folder}. Please check the zip contents.')

    # Move model and index file to extraction folder root
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # Remove any nested folders that are no longer needed
    for item in os.listdir(extraction_folder):
        item_path = os.path.join(extraction_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model: {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Model directory {dir_name} already exists! Please choose a different name.')

        if 'huggingface.co' in url:
            urllib.request.urlretrieve(url, zip_name)
        elif 'pixeldrain.com' in url:
            zip_name = dir_name + '.zip'
            url = f'https://pixeldrain.com/api/file/{zip_name}'
            urllib.request.urlretrieve(url, zip_name)
        elif 'drive.google.com' in url:
            zip_name = dir_name + '.zip'
            file_id = url.split('/')[-2]
            output = os.path.join('.', f'{dir_name}.zip')
            gdown.download(id=file_id, output=output, quiet=False)

        progress(0.5, desc='[~] Extracting model zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Model {dir_name} has been successfully downloaded!'
    except Exception as e:
        raise gr.Error(str(e))

def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Model directory {dir_name} already exists! Please choose a different name.')
        zip_name = zip_path.name
        progress(0.5, desc='[~] Extracting uploaded model zip...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Model {dir_name} has been successfully uploaded!'
    except Exception as e:
        raise gr.Error(str(e))

def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.update(value=pub_models.loc[event.index[0], 'URL']), gr.update(value=pub_models.loc[event.index[0], 'Model Name'])

def swap_visibility():
    # Switch between YouTube link input and file upload view
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)

def process_file_upload(file):
    return file.name, gr.update(value=file.name)

def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo in ['rmvpe+', 'mangio-crepe', 'hybrid[rmvpe+mangio-crepe]', 'hybrid[mangio-crepe+rmvpe]', 'hybrid[mangio-crepe+fcpe]', 'hybrid[mangio-crepe+rmvpe+fcpe]']:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def show_pitch_slider(pitch_detection_algo):
    if pitch_detection_algo in ['rmvpe+']:
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)
        

def update_f0_method(use_hybrid_methods):
    if use_hybrid_methods:
        return gr.update(choices=['hybrid[rmvpe+fcpe]', 'hybrid[rmvpe+mangio-crepe]', 'hybrid[mangio-crepe+rmvpe]', 'hybrid[mangio-crepe+fcpe]', 'hybrid[mangio-crepe+rmvpe+fcpe]'], value='hybrid[rmvpe+fcpe]')
    else:
        return gr.update(choices=['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe'], value='rmvpe+')

# Define a helper to reset parameters (improves readability)
def reset_defaults():
    # Returns the default values for all parameters in order
    return [
        0,    # pitch
        0.5,  # index_rate
        3,    # filter_radius
        0.25, # rms_mix_rate
        0.33, # protect
        128,  # crepe_hop_length
        0,    # main_gain
        0,    # backup_gain
        0,    # inst_gain
        0.25, # reverb_rm_size
        0.75, # reverb_width
        0.05, # reverb_wet
        0.85, # reverb_dry
        0.5,  # reverb_damping
        0,    # delay_time
        0,    # delay_feedback
        4,    # compressor_ratio
        -16,  # compressor_threshold
        -1,   # low_shelf_gain
        3,    # high_shelf_gain
        -30,  # noise_gate_threshold
        6,    # noise_gate_ratio
        10,   # noise_gate_attack
        100,  # noise_gate_release
        0,    # drive_db
        0,    # chorus_rate_hz
        0,    # chorus_depth
        0,    # chorus_centre_delay_ms
        0,    # chorus_feedback
        0,    # chorus_mix
        0,    # clipping_threshold
        0,    # f0autotune (False interpreted as 0)
        50,   # f0_min
        1100, # f0_max
        None, None, None, None, None  # audio outputs
    ]

def downoad_tabs():
    with gr.TabItem("Download Models"):
        with gr.Row():
            url_mod = gr.Text(label="Url Model")
            mod_name = gr.Text(label="model name")
        download = gr.Button("Download")
        download.click(download_online_model inputs=[url_mod, mod_name] outputs=None)


def build_interface():
    # Using a built-in theme to improve appearance
    with gr.Blocks(title="CoverGen-RVC", theme=gr.themes.Soft(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)) as app:
        voice_models = get_current_models(rvc_models_dir)
        is_webui = gr.Number(value=1, visible=False)
        
        # Use Tabs for high-level grouping
        with gr.Tabs():
            with gr.TabItem("CoverGen"):
                with gr.Row():
                    with gr.Column():
                        rvc_model = gr.Dropdown(voice_models, label='Voice Models',
                                                info='Directory "CoverGen/rvc_models". Click "Update Models List" after adding new models.')
                        ref_btn = gr.Button('Update Models List 🔁', variant='primary')
                        with gr.Row():
                            downoad_tabs()
                        with gr.Row():
                            with gr.Column() as yt_link_col:
                                song_input = gr.Text(label='Input Song', info='Enter a YouTube link or a file path')
                                song_input_file = gr.UploadButton('Upload from device', file_types=['audio'], variant='primary')
                            with gr.Column(visible=False) as file_upload_col:
                                local_file = gr.File(label='Audio File')
                                show_yt_link_button = gr.Button('Switch to YouTube/File Path')
                            
                            # Bind file upload callback
                            song_input_file.upload(process_file_upload, inputs=[song_input_file],
                                                   outputs=[local_file, song_input])
                            show_yt_link_button.click(swap_visibility,
                                                      outputs=[yt_link_col, file_upload_col, song_input, local_file])
                            
                        with gr.Row():
                            pitch = gr.Slider(-24, 24, value=0, step=1, label='Voice Pitch Shift',
                                              info='Negative for a more masculine tone; positive for a feminine tone')
                        f0autotune = gr.Checkbox(label="Auto-tuning", info='Automatically adjust pitch for a harmonious vocal', value=False)
                    
                    # Group transformation settings in an accordion
                    with gr.Accordion('Voice Transformation Settings', open=False):
                        gr.Markdown('<center><h2>Basic Settings</h2></center>')
                        with gr.Row():
                            index_rate = gr.Slider(0, 1, value=0.5, label='Indexing Speed',
                                                   info="Balance between AI voice character and artifact reduction")
                            filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius',
                                                      info='Applies median filtering to reduce noise')
                            rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS Mix Rate',
                                                     info="Controls how much of the original volume is preserved")
                            protect = gr.Slider(0, 0.5, value=0.33, label='Protection Level',
                                                info='Protects plosives and breathing sounds')
                        gr.Markdown('<center><h2>Pitch Extraction Settings</h2></center>')
                        with gr.Row():
                            with gr.Column():
                                use_hybrid_methods = gr.Checkbox(label="Use hybrid methods", value=False)
                                f0_method = gr.Dropdown(['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe'], value='rmvpe+', label='Pitch Extraction Method')
                                use_hybrid_methods.change(update_f0_method, inputs=use_hybrid_methods, outputs=f0_method)
                            crepe_hop_length = gr.Slider(8, 512, value=128, step=8, label='Hop Length',
                                                         info='Smaller values yield better pitch accuracy at the expense of processing time')
                            f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                            f0_min = gr.Slider(1, 16000, value=50, step=1, label="Minimum Pitch (Hz)",
                                               info="Lower bound for pitch detection (softer voice)")
                            f0_max = gr.Slider(1, 16000, value=1100, step=1, label="Maximum Pitch (Hz)",
                                               info="Upper bound for pitch detection (more raspy)")
                            f0_method.change(show_pitch_slider, inputs=f0_method, outputs=[f0_min, f0_max])
                        keep_files = gr.Checkbox(label='Save intermediate files',
                                                 info='Keep all temporary audio files in song_output/id', visible=False)
                
                # Audio mixing & effects
                with gr.Accordion('Audio Mixing Settings', open=False):
                    gr.Markdown('<center><h2>Volume Adjustment (dB)</h2></center>')
                    with gr.Row():
                        main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main Vocal')
                        backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backup Vocal')
                        inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Music')
                    with gr.Accordion('Effects', open=False):
                        with gr.Accordion('Reverb', open=False):
                            with gr.Row():
                                reverb_rm_size = gr.Slider(0, 1, value=0.25, label='Room Size')
                                reverb_width = gr.Slider(0, 1, value=0.75, label='Reverb Width')
                                reverb_wet = gr.Slider(0, 1, value=0.05, label='Wet Level')
                                reverb_dry = gr.Slider(0, 1, value=0.85, label='Dry Level')
                                reverb_damping = gr.Slider(0, 1, value=0.5, label='Damping Level')
                        with gr.Accordion('Echo', open=False):
                            with gr.Row():
                                delay_time = gr.Slider(0, 2, value=0, label='Delay Time')
                                delay_feedback = gr.Slider(0, 1, value=0, label='Feedback Level')
                        with gr.Accordion('Chorus', open=False):
                            with gr.Row():
                                chorus_rate_hz = gr.Slider(0.1, 10, value=0, label='Chorus Rate (Hz)')
                                chorus_depth = gr.Slider(0, 1, value=0, label='Chorus Depth')
                                chorus_centre_delay_ms = gr.Slider(0, 50, value=0, label='Center Delay (ms)')
                                chorus_feedback = gr.Slider(0, 1, value=0, label='Feedback')
                                chorus_mix = gr.Slider(0, 1, value=0, label='Mix')
                    with gr.Accordion('Processing', open=False):
                        with gr.Accordion('Compressor', open=False):
                            with gr.Row():
                                compressor_ratio = gr.Slider(1, 20, value=4, label='Compressor Ratio')
                                compressor_threshold = gr.Slider(-60, 0, value=-16, label='Compressor Threshold')
                        with gr.Accordion('Limiter', open=False):
                            with gr.Row():
                                limiter_threshold = gr.Slider(-12, 0, value=0, label='Limiter Threshold')
                        with gr.Accordion('Filters', open=False):
                            with gr.Row():
                                low_shelf_gain = gr.Slider(-20, 20, value=-1, label='Low Shelf Gain')
                                high_shelf_gain = gr.Slider(-20, 20, value=3, label='High Shelf Gain')
                        with gr.Accordion('Noise Reduction', open=False):
                            with gr.Row():
                                noise_gate_threshold = gr.Slider(-60, 0, value=-30, label='Noise Gate Threshold')
                                noise_gate_ratio = gr.Slider(1, 20, value=6, label='Noise Gate Ratio')
                                noise_gate_attack = gr.Slider(0, 100, value=10, label='Noise Gate Attack (ms)')
                                noise_gate_release = gr.Slider(0, 1000, value=100, label='Noise Gate Release (ms)')
                with gr.Accordion('Other Effects', open=False):
                    with gr.Accordion('Distortion', open=False):
                        drive_db = gr.Slider(-20, 20, value=0, label='Drive (dB)')
                    with gr.Accordion('Clipping', open=False):
                        clipping_threshold = gr.Slider(-20, 0, value=0, label='Clipping Threshold')
                
                # Output and control buttons
                with gr.Row():
                    with gr.Column(scale=2, min_width=100):
                        generate_btn = gr.Button("Generate", variant='primary', scale=1, min_width=100)
                    with gr.Column(scale=5):
                        with gr.Group():
                            ai_cover = gr.Audio(label='AI Cover', show_share_button=False)
                            with gr.Accordion("Intermediate Audio Files", open=False):
                                ai_vocals = gr.Audio(label='Transformed Vocals', show_share_button=False)
                                main_vocals_dereverb = gr.Audio(label='Vocals', show_share_button=False)
                                backup_vocals = gr.Audio(label='Backup Vocals', show_share_button=False)
                                instrumentals = gr.Audio(label='Instrumental', show_share_button=False)
                    with gr.Column(scale=1, min_width=100):
                        output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='Output File Type')
                        clear_btn = gr.Button(value='Reset all parameters', min_width=100)
                
                # Bind actions
                ref_btn.click(update_models_list, None, outputs=rvc_model)
                generate_btn.click(
                    song_cover_pipeline,
                    inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                            inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                            protect, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width,
                            low_shelf_gain, high_shelf_gain, limiter_threshold, compressor_ratio,
                            compressor_threshold, delay_time, delay_feedback, noise_gate_threshold, noise_gate_ratio,
                            noise_gate_attack, noise_gate_release, output_format, drive_db, chorus_rate_hz,
                            chorus_depth, chorus_centre_delay_ms, chorus_feedback, chorus_mix, clipping_threshold,
                            f0autotune, f0_min, f0_max],
                    outputs=[ai_cover, ai_vocals, main_vocals_dereverb, backup_vocals, instrumentals]
                )
                clear_btn.click(lambda: reset_defaults(), 
                                outputs=[pitch, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length,
                                         main_gain, backup_gain, inst_gain, reverb_rm_size, reverb_width,
                                         reverb_wet, reverb_dry, reverb_damping, delay_time, delay_feedback,
                                         compressor_ratio, compressor_threshold, low_shelf_gain, high_shelf_gain,
                                         limiter_threshold, noise_gate_threshold, noise_gate_ratio, noise_gate_attack,
                                         noise_gate_release, drive_db, chorus_rate_hz, chorus_depth,
                                         chorus_centre_delay_ms, chorus_feedback, chorus_mix, clipping_threshold,
                                         f0autotune, f0_min, f0_max, ai_cover, ai_vocals, main_vocals_dereverb,
                                         backup_vocals, instrumentals])
        
        return app

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Create an AI cover of a song in the directory song_output/id.',
        add_help=True
    )
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Allow sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI accessible from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    args = parser.parse_args()

    app = build_interface()
    app.launch(
        share=True,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
