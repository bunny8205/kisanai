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

2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

📦 Requirements

The file requirements.txt is already provided.
If needed, it should contain:

flask
speechrecognition
gtts
googletrans==4.0.0-rc1
ollama
numpy
pandas
joblib
requests
geocoder
pytz
pydub

📂 Project Structure

Since all datasets and models are in the root directory, your folder should look like this:

kisanai/
│── ai8.py
│── requirements.txt
│── README.md
│── cleaned_dataset.csv
│── crop_steps.csv
│── loansubsidy.csv
│── improved_crop_category_model (1).pkl
│── crop_category_encoder (1).pkl
│── soil_ph_model (1).pkl
│── irrigation_requirement_model.joblib
│── stage_prediction_model.joblib
│── irrigation_type_model.joblib
│── fertilizer_type_grouped_model.joblib
│── fertilizer_median_dose_table.parquet


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
Please cite data sources if reusing.
