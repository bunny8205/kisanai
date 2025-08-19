🌾 KisanAi

A Multilingual, Voice-First Agentic Assistant for Agriculture

📌 Project Description

KrishiMitra AI is a multilingual, voice-first agricultural assistant designed to help Indian farmers with crop recommendations, irrigation and fertilizer guidance, soil health analysis, weather insights, and access to government subsidies/loans.

It integrates speech recognition, translation, machine learning models, and curated datasets into one reliable, explainable, and farmer-friendly assistant. Farmers can interact in local Indian languages via text or speech, and receive stage-wise, actionable advice.

⚙️ Features

✅ Voice + Text Support (Hindi, English, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali)

✅ Crop recommendation using soil, climate, and seasonal conditions

✅ Soil pH prediction with fallback to district averages

✅ Stage-wise irrigation & fertilizer guidance

✅ Peak price & demand month prediction

✅ Government subsidy/loan scheme retrieval with fuzzy search

✅ Fallback strategies to ensure reliability even with noisy/incomplete queries

🛠️ Installation
1. Clone the repository
git clone https://github.com/bunny8205/kisanai.git


cd kisanai

3. Create a virtual environment (recommended)
   
py -m venv venv

source venv/bin/activate   # for Linux/Mac

venv\Scripts\activate      # for Windows

5. Install dependencies
pip install -r requirements.txt

📦 Requirements

The file requirements.txt is already provided.
If needed, it should contain:

annotated-types==0.7.0
anyio==4.10.0
asgiref==3.9.1
audioop-lts==0.2.2
blinker==1.9.0
category_encoders==2.8.1
certifi==2025.8.3
chardet==3.0.4
charset-normalizer==3.4.3
click==8.1.8
colorama==0.4.6
decorator==5.2.1
Flask==3.1.1
flask-cors==6.0.1
future==1.0.0
geocoder==1.38.1
googletrans==4.0.2
googletrans-py==4.0.0
gTTS==2.5.4
h11==0.16.0
h2==4.2.0
hpack==4.1.0
hstspreload==2025.1.1
httpcore==1.0.9
httpx==0.28.1
hyperframe==6.1.0
idna==2.10
imbalanced-learn==0.13.0
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.5.1
lightgbm==4.6.0
MarkupSafe==3.0.2
numpy==2.3.2
ollama==0.5.3
packaging==25.0
pandas==2.3.1
patsy==1.0.1
pyarrow==21.0.0
pydantic==2.11.7
pydantic_core==2.33.2
pydub==0.25.1
python-dateutil==2.9.0.post0
pytz==2025.2
ratelim==0.1.6
requests==2.32.4
rfc3986==1.5.0
scikit-learn==1.6.1
scipy==1.16.1
six==1.17.0
sklearn-compat==0.1.3
sniffio==1.3.1
SpeechRecognition==3.14.3
standard-aifc==3.13.0
standard-chunk==3.13.0
statsmodels==0.14.5
threadpoolctl==3.6.0
typing-inspection==0.4.1
typing_extensions==4.14.1
tzdata==2025.2
urllib3==2.5.0
Werkzeug==3.1.3
xgboost==3.0.4

📂 Project Structure

All datasets and models are in the root directory



🧩 Installing Ollama

This project uses Ollama to run local LLMs (e.g., Llama 2) for natural language understanding.
Ollama is a system-level dependency and must be installed separately (not via pip).

1. Install Ollama

Linux / macOS

curl -fsSL https://ollama.com/install.sh | sh


Windows
Download and install from 👉 https://ollama.com/download

2. Start the Ollama Server

Run the Ollama background service:

ollama serve

3. Pull the Required Model

This project uses Llama 2 by default.
Download it once (approx. 4–7 GB depending on variant):

ollama pull llama2

ollama run llama2

▶️ Running the Application

Start the Flask server:

python ai8.py


By default, it runs at:

http://127.0.0.1:5000

🚀 Usage Examples

Text Query (English):

Which crop should I grow in Pune this month?


Text Query (Hindi):

इस महीने धान को कौन सा खाद डालना है?


Voice Query:
Farmer speaks in Hindi → Recognized → Translated → Answer returned in text + speech.

Sample Output:

“In Pune during August, with soil pH ~6.5 and rainfall ~520mm, you can grow rice, maize, or soybean. During flowering stage, drip irrigation is recommended every 7–10 days.”

🚧 Limitations

Depends on public datasets (some incomplete/outdated).

Weather API requires internet access.

No mobile/IVR interface yet (currently Flask backend only).

No pest/disease image detection (future module).

🔮 Future Enhancements

Offline-first support with on-device speech & translation models.

IVR/USSD/mobile app for farmers with no smartphones.

Pest/disease detection via computer vision.

IoT sensor integration for irrigation automation.

Direct integration with government scheme portals.

📜 License

This project uses public datasets and is intended for research & hackathon purposes.

