# CoverGen-RVC

CoverGen-RVC is an autonomous pipeline designed to create song covers using any RVC v2 trained AI voice from local/youtube audio files.
This tool is ideal for developers aiming to integrate singing functionalities into their AI assistants, chatbots, or VTubers, as well as for individuals who wish to hear their favorite characters perform their favorite songs.


## Features

**Inference WebUI**: Easily generate AI covers by selecting a local audio file and applying a trained RVC model. The intuitive interface ensures a straightforward process, even for those new to voice conversion technologies.

**Model Downloader**: Conveniently download RVC models directly from within the WebUI. Navigate to the "Download Model" tab, paste the download link, and assign a unique name. Ensure the downloaded zip file contains the .pth model file and, optionally, an .index file.


## Getting Started

To begin using **CoverGen-RVC**, follow these steps:

1. Clone the Repository: Clone the CoverGen-RVC repository to your local machine.

```
git clone https://github.com/TheNeodev/CoverGen-RVC.git
```

2. Navigate to the Directory: Move into the cloned repository's directory.
```
cd CoverGen-RVC
```

3. Install Dependencies: Install the necessary dependencies. It's recommended to use a virtual environment.
```
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

4. Run the WebUI: Launch the WebUI to start generating AI covers.
```
python src/webgen.py -h
```

5. Access the WebUI: Open your web browser and navigate to http://localhost:5000 to access the CoverGen-RVC interface.



## Usage

## Generating AI Covers

**1. Select Audio File**: In the "Generate" tab, upload a local audio file you wish to convert or paste youtube url.


**2. Choose Voice Model**: From the "Voice Models" dropdown menu, select the RVC v2 trained AI voice model you want to apply.


**3. Adjust Pitch (Optional)**: Set the pitch adjustment as needed. For instance, use -12 for male voices and 12 for female voices.


**4. Generate Cover**: Click the "Generate" button. The AI-generated cover will be produced in a few minutes, depending on your system's performance.



### Downloading Voice Models

**1. Navigate to "Download Model" Tab**: Access the "Download Model" section within the WebUI.


**2. Provide Download Link**: Paste the download link of the desired RVC model. You can find trained voice models in the AI Hub Discord.


**3. Assign Model Name**: Enter a unique name for the model to distinguish it from others.


**4. Download Model**: Click the "Download" button. Once the message "[NAME] Model successfully downloaded!" appears, the model is ready for use.



## Contributing

We welcome contributions to enhance CoverGen-RVC. To contribute:

**1. Fork the repository.**


**2. Create a new branch**:
```
git checkout -b feature-name
```

**3. Make your changes and commit them:**
```
git commit -m 'Add new feature'
```

**4. Push to the branch:**
```
git push origin feature-name
```

**5. Open a pull request detailing your changes.**



## License

This project is licensed under the MIT License.
See the LICENSE file for more information.

## Acknowledgments

Special thanks to the developers and contributors of RVC v2 and the AI community for their continuous support and innovation.


---

*Note: Ensure that the downloaded zip file for voice models contains the .pth model file and, optionally, an .index file. This structure is crucial for the proper functioning of the model within CoverGen-RVC.*

