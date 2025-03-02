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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

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
        if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
            index_filepath = os.path.join(root, name)
        if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
            model_filepath = os.path.join(root, name)

if not model_filepath:
    raise gr.Error(f'The .pth model file was not found in the extracted zip file. Please check {extraction_folder}.')

os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
if index_filepath:
    os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

for filepath in os.listdir(extraction_folder):
    if os.path.isdir(os.path.join(extraction_folder, filepath)):
        shutil.rmtree(os.path.join(extraction_folder, filepath))

def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Downloading voice model named {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'The voice model directory {dir_name} already exists! Please choose a different name for your voice model.')

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

        progress(0.5, desc='[~] Extracting zip file...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Model {dir_name} has been successfully downloaded!'
    except Exception as e:
        raise gr.Error(str(e))

def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'The voice model directory {dir_name} already exists! Please choose a different name for your voice model.')

        zip_name = zip_path.name
        progress(0.5, desc='[~] Extracting zip file...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Model {dir_name} has been successfully uploaded!'
    except Exception as e:
        raise gr.Error(str(e))

def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.update(value=pub_models.loc[event.index[0], 'URL']), gr.update(value=pub_models.loc[event.index[0], 'Model Name'])

def swap_visibility():
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
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def update_f0_method(use_hybrid_methods):
    if use_hybrid_methods:
        return gr.update(choices=['hybrid[rmvpe+fcpe]', 'hybrid[rmvpe+mangio-crepe]', 'hybrid[mangio-crepe+rmvpe]', 'hybrid[mangio-crepe+fcpe]', 'hybrid[mangio-crepe+rmvpe+fcpe]'], value='hybrid[rmvpe+fcpe]')
    else:
        return gr.update(choices=['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe'], value='rmvpe+')

if __name__ == '__main__':
    parser = ArgumentParser(description='Create an AI cover of a song in the directory song_output/id.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Allow sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make the WebUI accessible from your local network.")
    parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
    parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)

    with gr.Blocks(title='CoverGen-RVC') as app:

        
        with gr.Tab("CoverGen"):
            with gr.Row():
                with gr.Column():
                    rvc_model = gr.Dropdown(voice_models, label='Voice Models', info='Directory "CoverGen/rvc_models". After adding new models to this directory, click the "Update Models List" button')
                    ref_btn = gr.Button('Update Models List 🔁', variant='primary')
                    with gr.Row():
                        
                        with gr.Column() as yt_link_col:
                            song_input = gr.Text(label='Input Song', info='Link to a song on YouTube or full path to a local file')
                            song_input_file = gr.UploadButton('Upload file from device', file_types=['audio'], variant='primary')

                        with gr.Column(visible=False) as file_upload_col:
                            local_file = gr.File(label='Audio File')
                            show_yt_link_button = gr.Button('Insert YouTube link / file path')
                            song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])
                            show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

                        with gr.Column():
                            pitch = gr.Slider(-24, 24, value=0, step=1, label='Voice Pitch Shift', info='-24 for masculine voice || 24 for feminine voice')
                            f0autotune = gr.Checkbox(label="Auto-tuning", info='Automatically adjusts the pitch for a more harmonious vocal sound', value=False)

                with gr.Accordion('Voice Transformation Settings', open=False):
                    gr.Markdown('<center><h2>Basic Settings</h2></center>')
                    with gr.Row():
                        index_rate = gr.Slider(0, 1, value=0.5, label='Indexing Speed', info="Controls how much of the AI voice's character is preserved in the vocal. Lower values may help reduce artifacts present in the audio")
                        filter_radius = gr.Slider(0, 7, value=3, step=1, label='Filter Radius', info='If >=3: applies median filtering to the pitch extraction results. May reduce breathing noise')
                        rms_mix_rate = gr.Slider(0, 1, value=0.25, label='RMS Mix Rate', info="Controls how accurately the original voice's volume is preserved (0) or fixed volume (1)")
                        protect = gr.Slider(0, 0.5, value=0.33, label='Protection Level', info='Protects plosives and breathing sounds. Increasing this parameter to its maximum of 0.5 ensures full protection')
                    gr.Markdown('<center><h2>Pitch Extraction Settings</h2></center>')
                    with gr.Row():
                        with gr.Column():
                            use_hybrid_methods = gr.Checkbox(label="Use hybrid methods", value=False)
                            f0_method = gr.Dropdown(['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe'], value='rmvpe+', label='Pitch Extraction Method')
                            use_hybrid_methods.change(update_f0_method, inputs=use_hybrid_methods, outputs=f0_method)
                        crepe_hop_length = gr.Slider(8, 512, value=128, step=8, visible=True, label='Hop Length', info='Smaller values lead to longer processing time and a higher risk of voice cracking, but with better pitch accuracy')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                        f0_min = gr.Slider(label="Minimum pitch range:", info="Specify the minimum pitch range for inference (prediction) in Hertz. This parameter sets the lower bound of the pitch range the algorithm will use to determine the fundamental frequency (F0) in the audio signal. (VOICE WILL BE SOFTER)", step=1, minimum=1, value=50, maximum=16000, visible=True)
                        f0_method.change(show_pitch_slider, inputs=f0_method, outputs=f0_min)
                        f0_max = gr.Slider(label="Maximum pitch range:", info="Specify the maximum pitch range for inference (prediction) in Hertz. This parameter sets the upper bound of the pitch range the algorithm will use to determine the fundamental frequency (F0) in the audio signal. (VOICE WILL BE MORE RASPY)", step=1, minimum=1, value=1100, maximum=16000, visible=True)
                        f0_method.change(show_pitch_slider, inputs=f0_method, outputs=f0_max)
                    keep_files = gr.Checkbox(label='Save intermediate files', info='Save all audio files created in the directory song_output/id, such as Extracted Vocals/Instrumental', visible=False)

                with gr.Accordion('Audio Mixing Settings', open=False):
                    gr.Markdown('<center><h2>Volume Adjustment (decibels)</h2></center>')
                    with gr.Row():
                        main_gain = gr.Slider(-20, 20, value=0, step=1, label='Main Vocal')
                        backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Backup Vocal')
                        inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Music')

                    with gr.Accordion('Effects', open=False):
                        with gr.Accordion('Reverb', open=False):
                            with gr.Row():
                                reverb_rm_size = gr.Slider(0, 1, value=0.25, label='Room Size', info='This parameter determines the size of the virtual room in which the reverb will sound. A higher value means a larger room and a longer reverb tail.')
                                reverb_width = gr.Slider(0, 1, value=0.75, label='Reverb Width', info='This parameter determines the width of the reverb sound. A higher value means a wider reverb.')
                                reverb_wet = gr.Slider(0, 1, value=0.05, label='Wet Level', info='This parameter determines the level of reverb. A higher value means the reverb effect will be more prominent and have a longer tail.')
                                reverb_dry = gr.Slider(0, 1, value=0.85, label='Dry Level', info='This parameter determines the level of the original sound without reverb. A lower value means the AI vocal sound will be quieter. If set to 0, the original sound will disappear completely.')
                                reverb_damping = gr.Slider(0, 1, value=0.5, label='Damping Level', info='This parameter controls the absorption of high frequencies in the reverb. A higher value means more absorption and a less "bright" reverb sound.')
                        with gr.Accordion('Echo', open=False):
                            with gr.Row():
                                delay_time = gr.Slider(0, 2, value=0, label='Echo - Delay Time', info='This parameter controls the time interval between the original sound and its echo. A higher value means a longer delay.')
                                delay_feedback = gr.Slider(0, 1, value=0, label='Echo - Feedback Level', info='This parameter controls the amount of echo that is fed back into the effect. A higher value means more repetitions of the echo.')
                        with gr.Accordion('Chorus', open=False):
                            with gr.Row():
                                chorus_rate_hz = gr.Slider(0.1, 10, value=0, label='Chorus Rate (Hz)', info='This parameter controls the oscillation speed of the chorus effect in Hertz. A higher value means faster oscillation.')
                                chorus_depth = gr.Slider(0, 1, value=0, label='Chorus Depth', info='This parameter controls the intensity of the chorus effect. A higher value means a stronger chorus effect.')
                                chorus_centre_delay_ms = gr.Slider(0, 50, value=0, label='Center Delay (ms)', info='This parameter controls the delay of the central signal in the chorus effect in milliseconds. A higher value means a longer delay.')
                                chorus_feedback = gr.Slider(0, 1, value=0, label='Feedback', info='This parameter controls the level of feedback in the chorus effect. A higher value means stronger feedback.')
                                chorus_mix = gr.Slider(0, 1, value=0, label='Mix', info='This parameter controls the balance between the original signal and the chorus effect. A higher value means a stronger chorus effect.')
                    with gr.Accordion('Processing', open=False):
                        with gr.Accordion('Compressor', open=False):
                            with gr.Row():
                                compressor_ratio = gr.Slider(1, 20, value=4, label='Compressor - Ratio', info='This parameter controls the amount of compression applied to the audio. A higher value means more compression, which reduces the dynamic range of the audio, making loud parts quieter and quiet parts louder.')
                                compressor_threshold = gr.Slider(-60, 0, value=-16, label='Compressor - Threshold', info='This parameter sets the threshold above which the compressor activates. The compressor compresses loud sounds to create a more even sound. A lower threshold means more sounds will be compressed.')
                        with gr.Accordion('Limiter', open=False):
                            with gr.Row():
                                limiter_threshold = gr.Slider(-12, 0, value=0, label='Limiter - Threshold', info='This parameter sets the threshold at which the limiter activates. The limiter restricts the audio volume to prevent overload and distortion. If set too low, the sound may become overloaded and distorted')
                        with gr.Accordion('Filters', open=False):
                            with gr.Row():
                                low_shelf_gain = gr.Slider(-20, 20, value=-1, label='Low Shelf Filter', info='This parameter controls the gain (volume) of the low frequencies. A positive value boosts the lows, giving a bassier sound. A negative value reduces the lows, making the sound thinner.')
                                high_shelf_gain = gr.Slider(-20, 20, value=3, label='High Shelf Filter', info='This parameter controls the gain of the high frequencies. A positive value boosts the highs, making the sound brighter. A negative value reduces the highs, making the sound duller.')
                        with gr.Accordion('Noise Reduction', open=False):
                            with gr.Row():
                                noise_gate_threshold = gr.Slider(-60, 0, value=-30, label='Threshold', info='This parameter sets the threshold in decibels below which the signal is considered noise. When the signal drops below this threshold, the noise gate activates and reduces the volume.')
                                noise_gate_ratio = gr.Slider(1, 20, value=6, label='Ratio', info='This parameter sets the level of noise reduction. A higher value means stronger noise reduction.')
                                noise_gate_attack = gr.Slider(0, 100, value=10, label='Attack Time (ms)', info='This parameter controls how quickly the noise gate opens when the sound becomes loud enough. A higher value means a slower attack.')
                                noise_gate_release = gr.Slider(0, 1000, value=100, label='Release Time (ms)', info='This parameter controls how quickly the noise gate closes when the sound becomes quiet. A higher value means a slower release.')
                with gr.Accordion('Other Effects', open=False):
                    with gr.Accordion('Distortion', open=False):
                        drive_db = gr.Slider(-20, 20, value=0, label='Drive (dB)', info='This parameter controls the level of signal distortion in decibels. A higher value means more distortion.')
                    with gr.Accordion('Clipping', open=False):
                        clipping_threshold = gr.Slider(-20, 0, value=0, label='Clipping Threshold', info='This parameter sets the threshold in decibels at which clipping occurs. Clipping is used to prevent overload and distortion. If the threshold is set too low, the sound may become overloaded and distorted.')
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
                        output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='Output File Type', scale=0.5)
                        clear_btn = gr.ClearButton(value='Reset all parameters', components=[keep_files, use_hybrid_methods], min_width=100)

                ref_btn.click(update_models_list, None, outputs=rvc_model)
                is_webui = gr.Number(value=1, visible=False)
                generate_btn.click(song_cover_pipeline,
                                  inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                          inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                          protect, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width,
                                          low_shelf_gain, high_shelf_gain, limiter_threshold, compressor_ratio,
                                          compressor_threshold, delay_time, delay_feedback, noise_gate_threshold,
                                          noise_gate_ratio, noise_gate_attack, noise_gate_release, output_format,
                                          drive_db, chorus_rate_hz, chorus_depth, chorus_centre_delay_ms, chorus_feedback, chorus_mix,
                                          clipping_threshold, f0autotune, f0_min, f0_max],
                                  outputs=[ai_cover, ai_vocals, main_vocals_dereverb, backup_vocals, instrumentals])
                clear_btn.click(lambda: [0, 0.5, 3, 0.25, 0.33, 128,
                                          0, 0, 0, 0.25, 0.75, 0.05, 0.85, 0.5, 0, 0,
                                          4, -16, 0, -1, 3, -30, 6, 10, 100, 0, 0,
                                          0, 0, 0, 0, 0, False, 50, 1100,
                                          None, None, None, None, None],
                                  outputs=[pitch, index_rate, filter_radius, rms_mix_rate, protect,
                                          crepe_hop_length, main_gain, backup_gain, inst_gain, reverb_rm_size, reverb_width,
                                          reverb_wet, reverb_dry, reverb_damping, delay_time, delay_feedback, compressor_ratio,
                                          compressor_threshold, low_shelf_gain, high_shelf_gain, limiter_threshold,
                                          noise_gate_threshold, noise_gate_ratio, noise_gate_attack, noise_gate_release,
                                          drive_db, chorus_rate_hz, chorus_depth, chorus_centre_delay_ms, chorus_feedback,
                                          chorus_mix, clipping_threshold, f0autotune, f0_min, f0_max,
                                          ai_cover, ai_vocals, main_vocals_dereverb, backup_vocals, instrumentals])

    # Andik, go fuck yourself =)

    with gr.Tab('Model Upload'):
            with gr.Tab('Upload via Link'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Model download link', info='This should be a link to a zip file containing the .pth model file and an optional .index file.', scale=3)
                    model_name = gr.Text(label='Model Name', info='Give your model a unique name, distinct from other voice models.', scale=1.5)

                with gr.Row():
                    dl_output_message = gr.Text(label='Output Message', interactive=False, scale=3)
                    download_btn = gr.Button('Download Model', variant='primary', scale=1.5)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

            with gr.Tab('Upload Locally'):
                gr.Markdown('## Upload a locally trained RVC v2 model and index file')
                gr.Markdown('- Locate the model file (in the weights folder) and the optional index file (in the logs/[name] folder)')
                gr.Markdown('- Compress the files into a zip file')
                gr.Markdown('- Upload the zip file and give the voice a unique name')
                gr.Markdown('- Click the "Upload Model" button')

                with gr.Row():
                    with gr.Column(scale=2):
                        zip_file = gr.File(label='Zip File')
                    with gr.Column(scale=1.5):
                        local_model_name = gr.Text(label='Model Name', info='Give your model a unique name, distinct from other voice models.')
                        model_upload_button = gr.Button('Upload Model', variant='primary')

                with gr.Row():
                    local_upload_output_message = gr.Text(label='Output Message', interactive=False)
                    model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

    app.launch(
        share=True,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
