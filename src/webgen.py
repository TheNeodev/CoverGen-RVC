import os
import sys
import shutil
import urllib.request
import zipfile
import gdown
from argparse import ArgumentParser

import gradio as gr
from tabs.covergen import covergen_tab
from tabs.download_models import create_download_models_tab


def build_interface():
    # Custom CSS to ensure rows stack on narrow screens (mobile responsiveness)
    custom_css = """
    @media (max-width: 768px) {
        .gradio-container .row { 
            flex-direction: column !important; 
        }
    }
    """
    with gr.Blocks(
        title="CoverGen-RVC", 
        css=custom_css
    ) as app:
        gr.Label("CoverGen RVC")
        with gr.Tabs():
            covergen_tab()
            create_download_models_tab()
    return app

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generate an AI cover of a song and output to song_output/id.',
        add_help=True
    )
    parser.add_argument("-s", "--share", action="store_true", dest="share_enabled", default=False, help="Allow sharing")
    parser.add_argument("-l", "--listen", action="store_true", default=False, help="Make the WebUI accessible on your local network.")
    parser.add_argument("-lh", '--listen-host', type=str, help='Hostname for the server.')
    parser.add_argument("-lp",  '--listen-port', type=int, help='Port for the server.')
    args = parser.parse_args()

    app = build_interface()
    app.launch(
        share=args.share_enabled,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
