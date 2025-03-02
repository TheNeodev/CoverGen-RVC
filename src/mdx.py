import gc
import hashlib
import os
import queue
import threading
import warnings
import logging

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from tqdm import tqdm
from IPython.display import clear_output

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
stem_naming = {'Vocals': 'Instrumental', 'Other': 'Instruments', 'Instrumental': 'Vocals', 'Drums': 'Drumless', 'Bass': 'Bassless'}


class MDXModel:
    def __init__(self, device, dim_f, dim_t, n_fft, hop=1024, stem_name=None, compensation=1.000):
        logging.info("Initializing MDXModel")
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.dim_c = 4
        self.n_fft = n_fft
        self.hop = hop
        self.stem_name = stem_name
        self.compensation = compensation

        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)

        out_c = self.dim_c
        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)
        logging.info("MDXModel initialized with chunk_size: %d, n_bins: %d", self.chunk_size, self.n_bins)

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 4, self.n_bins, self.dim_t])
        return x[:, :, :self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1, 2, self.chunk_size])


class MDX:
    DEFAULT_SR = 44100
    # Unit: seconds
    DEFAULT_CHUNK_SIZE = 0 * DEFAULT_SR
    DEFAULT_MARGIN_SIZE = 1 * DEFAULT_SR

    DEFAULT_PROCESSOR = 0

    def __init__(self, model_path: str, params: MDXModel, processor=DEFAULT_PROCESSOR):
        logging.info("Initializing MDX session with model: %s", model_path)
        self.device = torch.device(f'cuda:{processor}') if processor >= 0 else torch.device('cpu')
        self.provider = ['CUDAExecutionProvider'] if processor >= 0 else ['CPUExecutionProvider']
        logging.info("Using device: %s, provider: %s", self.device, self.provider)

        self.model = params

        # Load the ONNX model using ONNX Runtime
        self.ort = ort.InferenceSession(model_path, providers=self.provider)
        # Preload the model for faster performance
        self.ort.run(None, {'input': torch.rand(1, 4, params.dim_f, params.dim_t).numpy()})
        self.process = lambda spec: self.ort.run(None, {'input': spec.cpu().numpy()})[0]

        self.prog = None
        logging.info("MDX session initialized.")

    @staticmethod
    def get_hash(model_path):
        logging.info("Calculating hash for model: %s", model_path)
        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logging.warning("Fallback hash computation due to error: %s", e)
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

        logging.info("Model hash: %s", model_hash)
        return model_hash

    @staticmethod
    def segment(wave, combine=True, chunk_size=DEFAULT_CHUNK_SIZE, margin_size=DEFAULT_MARGIN_SIZE):
        logging.info("Segmenting wave: combine=%s, chunk_size=%d, margin_size=%d", combine, chunk_size, margin_size)
        if combine:
            processed_wave = None
            for segment_count, segment in enumerate(wave):
                start = 0 if segment_count == 0 else margin_size
                end = None if segment_count == len(wave) - 1 else -margin_size
                if margin_size == 0:
                    end = None
                if processed_wave is None:
                    processed_wave = segment[:, start:end]
                else:
                    processed_wave = np.concatenate((processed_wave, segment[:, start:end]), axis=-1)
            logging.info("Combined segmentation complete.")
        else:
            processed_wave = []
            sample_count = wave.shape[-1]

            if chunk_size <= 0 or chunk_size > sample_count:
                chunk_size = sample_count

            if margin_size > chunk_size:
                margin_size = chunk_size

            for segment_count, skip in enumerate(range(0, sample_count, chunk_size)):
                margin = 0 if segment_count == 0 else margin_size
                end = min(skip + chunk_size + margin_size, sample_count)
                start = skip - margin

                cut = wave[:, start:end].copy()
                processed_wave.append(cut)

                if end == sample_count:
                    break
            logging.info("Segmentation complete with %d segments.", len(processed_wave))

        return processed_wave

    def pad_wave(self, wave):
        logging.info("Padding wave with shape: %s", wave.shape)
        n_sample = wave.shape[1]
        trim = self.model.n_fft // 2
        gen_size = self.model.chunk_size - 2 * trim
        pad = gen_size - n_sample % gen_size

        wave_p = np.concatenate((np.zeros((2, trim)), wave, np.zeros((2, pad)), np.zeros((2, trim))), 1)

        mix_waves = []
        for i in range(0, n_sample + pad, gen_size):
            waves = np.array(wave_p[:, i:i + self.model.chunk_size])
            mix_waves.append(waves)

        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)
        logging.info("Padding complete. pad=%d, trim=%d, number of segments=%d", pad, trim, mix_waves.shape[0])
        return mix_waves, pad, trim

    def _process_wave(self, mix_waves, trim, pad, q: queue.Queue, _id: int):
        logging.info("Processing wave segment ID: %d", _id)
        mix_waves = mix_waves.split(1)
        with torch.no_grad():
            pw = []
            for mix_wave in mix_waves:
                self.prog.update()
                spec = self.model.stft(mix_wave)
                processed_spec = torch.tensor(self.process(spec))
                processed_wav = self.model.istft(processed_spec.to(self.device))
                processed_wav = processed_wav[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).cpu().numpy()
                pw.append(processed_wav)
        processed_signal = np.concatenate(pw, axis=-1)[:, :-pad]
        q.put({_id: processed_signal})
        logging.info("Finished processing segment ID: %d", _id)
        return processed_signal

    def process_wave(self, wave: np.array, mt_threads=1):
        logging.info("Starting wave processing using %d threads.", mt_threads)
        self.prog = tqdm(total=0)
        chunk = wave.shape[-1] // mt_threads
        waves = self.segment(wave, False, chunk)
        logging.info("Total segments to process: %d", len(waves))

        q = queue.Queue()
        threads = []
        for c, batch in enumerate(waves):
            mix_waves, pad, trim = self.pad_wave(batch)
            self.prog.total = len(mix_waves) * mt_threads
            thread = threading.Thread(target=self._process_wave, args=(mix_waves, trim, pad, q, c))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.prog.close()
        clear_output(wait=True)
        logging.info("Wave processing complete. Collected segments: %d", q.qsize())

        processed_batches = []
        while not q.empty():
            processed_batches.append(q.get())
        processed_batches = [list(wave.values())[0] for wave in
                             sorted(processed_batches, key=lambda d: list(d.keys())[0])]
        assert len(processed_batches) == len(waves), 'Incomplete processed batches, please reduce batch size!'
        return self.segment(processed_batches, True, chunk)


def run_mdx(model_params, output_dir, model_path, filename, exclude_main=False, exclude_inversion=False, suffix=None, invert_suffix=None, denoise=False, keep_orig=True, m_threads=2):
    logging.info("Running MDX on file: %s", filename)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(device)
        vram_gb = device_properties.total_memory / 1024**3
        m_threads = 1 if vram_gb < 8 else 2
        logging.info("CUDA available. VRAM: %.2fGB, using %d threads", vram_gb, m_threads)
    else:
        logging.info("CUDA not available. Using CPU.")

    model_hash = MDX.get_hash(model_path)
    mp = model_params.get(model_hash)
    logging.info("Model parameters found for hash.")
    model = MDXModel(
        device,
        dim_f=mp["mdx_dim_f_set"],
        dim_t=2 ** mp["mdx_dim_t_set"],
        n_fft=mp["mdx_n_fft_scale_set"],
        stem_name=mp["primary_stem"],
        compensation=mp["compensate"]
    )

    mdx_sess = MDX(model_path, model)
    logging.info("Loading audio file: %s", filename)
    wave, sr = librosa.load(filename, mono=False, sr=44100)
    peak = max(np.max(wave), abs(np.min(wave)))
    wave /= peak
    logging.info("Audio loaded and normalized. Peak value: %f", peak)
    if denoise:
        logging.info("Denoise enabled. Processing inverted and non-inverted signals.")
        wave_processed = -(mdx_sess.process_wave(-wave, m_threads)) + (mdx_sess.process_wave(wave, m_threads))
        wave_processed *= 0.5
    else:
        wave_processed = mdx_sess.process_wave(wave, m_threads)
    wave_processed *= peak
    stem_name = model.stem_name if suffix is None else suffix

    main_filepath = None
    if not exclude_main:
        main_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        sf.write(main_filepath, wave_processed.T, sr)
        logging.info("Main file written to: %s", main_filepath)

    invert_filepath = None
    if not exclude_inversion:
        diff_stem_name = stem_naming.get(stem_name) if invert_suffix is None else invert_suffix
        stem_name = f"{stem_name}_diff" if diff_stem_name is None else diff_stem_name
        invert_filepath = os.path.join(output_dir, f"{os.path.basename(os.path.splitext(filename)[0])}_{stem_name}.wav")
        sf.write(invert_filepath, (-wave_processed.T * model.compensation) + wave.T, sr)
        logging.info("Inversion file written to: %s", invert_filepath)

    if not keep_orig:
        os.remove(filename)
        logging.info("Original file removed: %s", filename)

    del mdx_sess, wave_processed, wave
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logging.info("run_mdx completed.")
    return main_filepath, invert_filepath
