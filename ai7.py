import asyncio
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator
import os
from flask import Flask, render_template, request, jsonify, send_file
import ollama
import json
from typing import Dict, Any, List
import re
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import joblib
import requests
import geocoder
import logging
import random
import tempfile
from pydub import AudioSegment

app = Flask(__name__)

# Initialize translator globally
translator = Translator()

# System prompt constant
SYSTEM_PROMPT = """
You are an agricultural assistant that helps farmers with crop recommendations, 
soil analysis, and farming advice. You have access to datasets about soil pH, 
crop suitability, and weather patterns across India.

You can perform the following tasks:
1. Retrieve soil pH for a location
2. Recommend crops based on location and time
3. Identify soil type for a location
4. Provide rainfall and humidity information
5. Provide farming advice based on conditions
6. Provide irrigation information for specific crops
7. Provide fertilizer recommendations for specific crops
8. Provide peak demand month and price information for specific crops

Always extract and use the following from user queries:
- Location (district and state)
- Time period (month/year) if relevant
- Specific parameters if mentioned (pH, rainfall, moisture, humidity, temperature, etc.)
- Crop name when asking about irrigation or fertilizers or peak prices

Important: When user says "this month" or "current month", extract it as "this month".
When user says "this year" or doesn't specify year, don't include year in response.

If location is not specified, ask for clarification.
"""

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        self.supported_languages = {
            'hindi': 'hi-IN',
            'english': 'en-IN',
            'tamil': 'ta-IN',
            'telugu': 'te-IN',
            'kannada': 'kn-IN',
            'malayalam': 'ml-IN',
            'marathi': 'mr-IN',
            'bengali': 'bn-IN'
        }

    async def translate_to_english(self, text: str, src_lang: str) -> str:
        """Translate text from source language to English"""
        if src_lang == 'en':
            return text
        try:
            translation = await self.translator.translate(text, src=src_lang, dest='en')
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails

    async def translate_from_english(self, text: str, dest_lang: str) -> str:
        """Translate text from English to target language"""
        if dest_lang == 'en':
            return text
        try:
            translation = await self.translator.translate(text, src='en', dest=dest_lang)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails

    def text_to_speech(self, text: str, language: str) -> str:
        """Convert text to speech using explicit Indian language codes"""
        try:
            lang_code = self.supported_languages.get(language.lower(), 'en-IN')
            tts = gTTS(text=text, lang=lang_code, slow=False)
            temp_file = os.path.join(tempfile.gettempdir(), f"response_{random.randint(0, 10000)}.mp3")
            tts.save(temp_file)
            return temp_file
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None

class AgriculturalAgent:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize voice handler
        self.voice_handler = VoiceHandler()

        # Initialize models and data
        try:
            self.augmented_df = pd.read_csv('cleaned_dataset.csv')
            self.model = joblib.load('improved_crop_category_model (1).pkl')
            self.label_encoder = joblib.load('crop_category_encoder (1).pkl')
            self.model_ph = joblib.load('soil_ph_model (1).pkl')
            self.crop_steps_df = pd.read_csv('crop_steps.csv')
        except FileNotFoundError as e:
            raise SystemExit(f"Error loading required files: {e}")

        # Crop categories to specific crops mapping
        self.crop_categories = {
            'cereals': [
                'wheat', 'rice', 'maize', 'barley', 'sorghum', 'pearl millet', 'ragi', 'jowar'
            ],
            'pulses': [
                'chickpea', 'urad dal', 'mung bean', 'lentil', 'pigeon pea', 'rajma',
                'black-eyed pea', 'adzuki bean', 'green bean', 'cowpea', 'field bean',
                'horse gram', 'lathyrus'
            ],
            'oilseeds': [
                'mustard', 'sunflower', 'groundnut', 'sesame', 'castor', 'linseed', 'soybean', 'peanut'
            ],
            'vegetables': [
                'tomato', 'potato', 'onion', 'brinjal', 'cabbage', 'cauliflower',
                'pumpkin', 'okra', 'bottle gourd', 'cucumber', 'chili pepper', 'spinach',
                'amaranth', 'carrot', 'bitter gourd', 'broccoli', 'lettuce', 'radish'
            ],
            'fruits': [
                'mango', 'banana', 'orange', 'kinnow', 'lemon', 'lime', 'guava',
                'watermelon', 'papaya', 'grapes', 'apple', 'pear', 'plum',
                'litchi', 'pineapple', 'peach', 'cherry', 'jackfruit', 'grapefruit'
            ],
            'spices_and_condiments': [
                'turmeric', 'coriander', 'ginger', 'garlic', 'cumin', 'black pepper',
                'cardamom', 'tamarind', 'fenugreek'
            ],
            'plantation_crops': [
                'tea', 'coffee', 'coconut', 'arecanut', 'cashew'
            ],
            'dryland_and_fiber_crops': [
                'cotton', 'jute'
            ],
            'nuts': [
                'almond', 'hazelnut', 'walnut'
            ],
            'medicinal_and_misc': [
                'ashwagandha', 'moringa', 'betel leaf', 'tobacco'
            ],
            'tubers': [
                'tapioca'
            ]
        }

        self.month_map = {
            'jan': 'jan', 'january': 'jan',
            'feb': 'feb', 'february': 'feb',
            'mar': 'mar', 'march': 'mar',
            'apr': 'apr', 'april': 'apr',
            'may': 'may',
            'jun': 'jun', 'june': 'jun',
            'jul': 'jul', 'july': 'jul',
            'aug': 'aug', 'august': 'aug',
            'sep': 'sep', 'september': 'sep',
            'oct': 'oct', 'october': 'oct',
            'nov': 'nov', 'november': 'nov',
            'dec': 'dec', 'december': 'dec',
            'this month': datetime.now().strftime('%b').lower()
        }

        # Location aliases for normalization
        self.location_aliases = {
            'bombay': 'mumbai',
            'madras': 'chennai',
            'bangalore': 'bengaluru',
            'calcutta': 'kolkata',
            'pune': 'pune',
            'hyderabad': 'hyderabad',
            'ahmedabad': 'ahmedabad',
        }

    def translate_to_english(self, text: str, src_lang: str) -> str:
        """Translate text from source language to English"""
        return self.voice_handler.translate_to_english(text, src_lang)

    def translate_from_english(self, text: str, dest_lang: str) -> str:
        """Translate text from English to target language"""
        return self.voice_handler.translate_from_english(text, dest_lang)

    def get_peak_price_info(self, crop_name: str) -> str:
        """Get peak price and demand month information for a specific crop"""
        try:
            crop_name = crop_name.lower().strip()

            # Check if we have this crop in the dataset
            crop_data = self.augmented_df[self.augmented_df['crop'].str.lower() == crop_name]

            if crop_data.empty:
                return f"I don't have peak price information available for {crop_name.title()}."

            # Get all peak prices and months for this crop
            peak_prices = crop_data.iloc[:, 9].values  # 10th column (0-indexed as 9)
            peak_months = crop_data.iloc[:, 8].values  # 9th column (0-indexed as 8)

            # Find the highest peak price and corresponding month
            valid_prices = []
            valid_months = []

            for price, month in zip(peak_prices, peak_months):
                try:
                    if not pd.isna(price) and not pd.isna(month):
                        price_float = float(price)
                        valid_prices.append(price_float)
                        valid_months.append(month)
                except (ValueError, TypeError):
                    continue

            if not valid_prices:
                return f"I couldn't find valid peak price data for {crop_name.title()}."

            max_price = max(valid_prices)
            max_month = valid_months[valid_prices.index(max_price)]

            return (f"For {crop_name.title()}, the peak demand typically occurs in {max_month.capitalize()} "
                    f"with prices reaching ₹{max_price:.2f} per kg. This is based on historical market trends.")

        except Exception as e:
            self.logger.error(f"Error getting peak price info for {crop_name}: {e}")
            return f"Sorry, I'm having trouble accessing the peak price details for {crop_name.title()} right now."

    def get_irrigation_info(self, crop_name: str) -> str:
        """Get irrigation information in paragraph format"""
        try:
            crop_name = crop_name.lower().strip()
            crop_data = self.crop_steps_df[self.crop_steps_df['crop_name'].str.lower() == crop_name]

            if crop_data.empty:
                return f"I don't have irrigation details for {crop_name.title()} available."

            crop_data = crop_data.sort_values('days')

            response = [
                f"Here's how to manage watering for {crop_name.title()}. ",
                "The irrigation needs vary throughout the growth cycle. "
            ]

            for _, row in crop_data.iterrows():
                stage = row['stage'].lower()
                needs_irrigation = row['irrigation']
                method = row['irrigation_type']

                stage_text = f"During the {stage} stage, "

                if str(needs_irrigation).lower() == 'no':
                    stage_text += "natural rainfall is usually sufficient and additional irrigation isn't typically needed. "
                else:
                    if pd.isna(method):
                        stage_text += "you'll need to provide supplemental irrigation. "
                    else:
                        method = str(method).lower()
                        if "drip" in method:
                            stage_text += "drip irrigation is recommended for efficient water delivery. "
                        elif "sprinkler" in method:
                            stage_text += "sprinkler irrigation works well for this growth phase. "
                        elif "flood" in method:
                            stage_text += "flood irrigation can be used if properly managed. "
                        else:
                            stage_text += f"{method} irrigation is the preferred approach. "

                response.append(stage_text)

            response.append(
                "Keep in mind that actual water needs depend on your local weather conditions and soil type. "
                "Always check soil moisture before irrigating and adjust based on the plant's appearance."
            )

            return "".join(response)

        except Exception as e:
            self.logger.error(f"Error getting irrigation info for {crop_name}: {e}")
            return f"Sorry, I'm having trouble accessing the irrigation details for {crop_name.title()} right now."

    def get_fertilizer_info(self, crop_name: str) -> str:
        """Get fertilizer information in complete paragraph format"""
        try:
            crop_name = crop_name.lower().strip()
            crop_data = self.crop_steps_df[self.crop_steps_df['crop_name'].str.lower() == crop_name]

            if crop_data.empty:
                return f"I couldn't find any fertilizer information for {crop_name.title()} in my records."

            crop_data = crop_data.sort_values('days')

            response = [
                f"Let me walk you through the fertilizer requirements for {crop_name.title()}. ",
                "The needs change as the plant grows through different stages. "
            ]

            for _, row in crop_data.iterrows():
                stage = row['stage'].lower()
                fertilizer = row['fertilizer_type']
                dosage = row['fertilizer_dosage']

                stage_text = f"For the {stage} stage, "

                if pd.isna(fertilizer) or str(fertilizer).lower() in ['none', 'not needed']:
                    stage_text += "you generally won't need to apply any fertilizers. "
                else:
                    fert = str(fertilizer).replace(",", " and")

                    if pd.isna(dosage) or str(dosage).lower() in ['none', 'not needed']:
                        stage_text += f"you should use {fert}. "
                    else:
                        dosage = str(dosage)
                        dosage = dosage.replace("@", "at").replace(":", "").replace(";", "")

                        if "spray" in dosage.lower():
                            action = "spray"
                            dosage = dosage.replace("Spray", "").replace("spray", "").strip()
                        elif "top-dress" in dosage.lower():
                            action = "top-dress with"
                            dosage = dosage.replace("Top-dress", "").replace("top-dress", "").strip()
                        else:
                            action = "apply"

                        stage_text += f"you should {action} {fert} {dosage}. "

                response.append(stage_text)

            response.append(
                "Remember that these are general guidelines. "
                "Your specific soil conditions and local climate might require adjustments. "
                "It's always good practice to observe your plants and consult with local agricultural experts."
            )

            return "".join(response)

        except Exception as e:
            self.logger.error(f"Error getting fertilizer info for {crop_name}: {e}")
            return f"I'm sorry, I couldn't retrieve the fertilizer details for {crop_name.title()} at this time."

    def get_crop_steps(self, crop_name: str) -> str:
        """Get cultivation steps with robust error handling and natural language formatting"""
        try:
            # Normalize crop name and get data
            crop_name = crop_name.lower().strip()
            crop_data = self.crop_steps_df[self.crop_steps_df['crop_name'].str.lower() == crop_name]

            if crop_data.empty:
                return f"I don't currently have cultivation details for {crop_name.title()} in my database."

            # Helper function to safely parse days (handles ranges and single values)
            def parse_days(days_value):
                try:
                    if pd.isna(days_value):
                        return None
                    if isinstance(days_value, (int, float)):
                        return int(days_value)
                    if '-' in str(days_value):
                        parts = [p.strip() for p in str(days_value).split('-')]
                        valid_parts = [int(p) for p in parts if p.isdigit()]
                        return max(valid_parts) if valid_parts else None
                    return int(days_value) if str(days_value).isdigit() else None
                except (ValueError, TypeError):
                    return None

            # Process and sort data
            crop_data = crop_data.copy()
            crop_data['parsed_days'] = crop_data['days'].apply(parse_days)
            crop_data = crop_data.dropna(subset=['parsed_days']).sort_values('parsed_days')

            if crop_data.empty:
                return f"I have incomplete growth duration data for {crop_name.title()}."

            # Calculate total duration
            total_days = crop_data['parsed_days'].max()
            total_months = round(total_days / 30, 1)

            response = [
                f"Here's the complete cultivation guide for {crop_name.title()}:",
                f"This crop typically takes {total_days} days (~{total_months} months) from planting to harvest.\n"
            ]

            stage_templates = {
                'sowing': [
                    "Let's begin with the {stage} process:",
                    "The {stage} stage includes these important steps:"
                ],
                'vegetative': [
                    "During {stage} growth:",
                    "As the plants develop in the {stage} phase:"
                ],
                'flowering': [
                    "When {stage} begins:",
                    "The {stage} stage requires attention to:"
                ],
                'fruiting': [
                    "As fruits develop during {stage}:",
                    "The {stage} phase is crucial for:"
                ],
                'maturity': [
                    "Approaching {stage}:",
                    "Final {stage} considerations include:"
                ]
            }

            for _, row in crop_data.iterrows():
                stage = row['stage'].lower()
                templates = stage_templates.get(stage, ["During the {stage} stage:"])
                intro = random.choice(templates).format(stage=stage)

                stage_info = [
                    f"\n**{row['stage'].title()} Stage**",
                    intro
                ]

                # Irrigation information
                if pd.notna(row['irrigation']):
                    methods = {
                        'yes': ['requires irrigation', 'needs watering', 'benefits from irrigation'],
                        'no': ['does not require irrigation', 'should avoid overwatering']
                    }
                    irrig_text = random.choice(methods.get(row['irrigation'].lower(), ['requires irrigation']))
                    if pd.notna(row['irrigation_type']):
                        irrig_text += f", preferably using {row['irrigation_type'].lower()}"
                    stage_info.append(f"- {irrig_text.capitalize()}.")

                # Fertilizer information
                if pd.notna(row['fertilizer_type']):
                    fert_text = random.choice([
                        f"Apply {row['fertilizer_type']}",
                        f"Use {row['fertilizer_type']}",
                        f"Recommended fertilizer: {row['fertilizer_type']}"
                    ])
                    if pd.notna(row['fertilizer_dosage']):
                        fert_text += f" at {row['fertilizer_dosage']}"
                    stage_info.append(f"- {fert_text}.")

                # Pest/disease management
                if pd.notna(row['diseases']):
                    problem_text = random.choice([
                        f"Watch for {row['diseases'].lower()}",
                        f"Potential issues include {row['diseases'].lower()}",
                        f"Be alert for {row['diseases'].lower()}"
                    ])
                    if pd.notna(row['control_action']):
                        solution = random.choice([
                            f"Control with {row['control_action'].lower()}",
                            f"Manage by {row['control_action'].lower()}",
                            f"Treatment: {row['control_action'].lower()}"
                        ])
                        problem_text += f" - {solution}"
                    stage_info.append(f"- {problem_text}.")

                # Labor requirements
                if pd.notna(row['labour']):
                    stage_info.append(f"- Requires approximately {row['labour']} workers.")

                # Additional notes
                if pd.notna(row['notes']):
                    stage_info.append(f"- Note: {row['notes']}")

                response.extend(stage_info)

            # Add closing remarks
            closing = random.choice([
                "\nFollowing these practices should lead to a successful harvest!",
                "\nWith proper care, you'll achieve excellent results!",
                "\nThese guidelines will help you grow healthy crops!"
            ])
            response.append(closing)

            return "\n".join(response)

        except Exception as e:
            self.logger.error(f"Error retrieving {crop_name} steps: {str(e)}", exc_info=True)
            return f"I'm currently unable to access the cultivation details for {crop_name.title()}. Please try again later."

    def get_crop_steps_for_multiple(self, crop_names: List[str]) -> str:
        """Get cultivation steps for multiple crops, handling missing data gracefully"""
        responses = []
        for crop in crop_names:
            crop_response = self.get_crop_steps(crop)
            if "don't currently have cultivation details" in crop_response.lower():
                responses.append(f"**{crop.title()}**: Data not available")
            else:
                responses.append(crop_response)
        return "\n\n".join(responses)

    def process_assistance_request(self, query: str, suggested_crops: list) -> str:
        """Process user request for cultivation assistance with more flexible matching"""
        # First check if this is a multi-query request
        multi_query = self._check_multi_query(query)
        if multi_query:
            return self._handle_multi_query(multi_query, suggested_crops)

        # First check for irrigation or fertilizer specific queries
        normalized_query = query.lower().strip()

        # Check for irrigation queries
        irrigation_phrases = [
            'irrigation', 'watering', 'water needs', 'water requirements',
            'how much water', 'when to water', 'irrigation method',
            'irrigation type', 'irrigation needs'
        ]

        if any(phrase in normalized_query for phrase in irrigation_phrases):
            # Find which crop is being asked about
            for crop in suggested_crops:
                if crop.lower() in normalized_query:
                    return self.get_irrigation_info(crop)
            # If no specific crop mentioned, return info for all suggested crops
            responses = []
            for crop in suggested_crops:
                responses.append(self.get_irrigation_info(crop))
            return "\n\n".join(responses)

        # Check for fertilizer queries
        fertilizer_phrases = [
            'fertilizer', 'fertiliser', 'nutrients', 'plant food',
            'which fertilizer', 'what fertilizer', 'fertilizer type',
            'fertilizer dosage', 'how much fertilizer', 'when to fertilize'
        ]

        if any(phrase in normalized_query for phrase in fertilizer_phrases):
            # Find which crop is being asked about
            for crop in suggested_crops:
                if crop.lower() in normalized_query:
                    return self.get_fertilizer_info(crop)
            # If no specific crop mentioned, return info for all suggested crops
            responses = []
            for crop in suggested_crops:
                responses.append(self.get_fertilizer_info(crop))
            return "\n\n".join(responses)

        # Check for peak price queries
        peak_price_phrases = [
            'peak price', 'highest price', 'best price',
            'when is price highest', 'peak demand', 'demand month'
        ]

        if any(phrase in normalized_query for phrase in peak_price_phrases):
            # Find which crop is being asked about
            for crop in suggested_crops:
                if crop.lower() in normalized_query:
                    return self.get_peak_price_info(crop)
            # If no specific crop mentioned, return info for all suggested crops
            responses = []
            for crop in suggested_crops:
                responses.append(self.get_peak_price_info(crop))
            return "\n\n".join(responses)

        # Get all crops in our database
        all_crops = set(self.crop_steps_df['crop_name'].str.lower().unique())

        # Enhanced crop growing queries handling - moved up in priority
        growing_phrases = [
            'how to grow', 'growing', 'planting', 'cultivation',
            'steps for', 'process for', 'guide for', 'tell me about',
            'help me with', 'assist me with', 'information about',
            'details about', 'can you explain', 'what are the steps',
            'how do i grow', 'how can i grow', 'i want to grow',
            'teach me about', 'explain', 'describe', 'walk me through',
            'procedure for', 'method for', 'technique for',
            'tips for growing', 'advice for growing', 'recommendations for',
            'best way to grow', 'proper way to grow', 'right way to grow',
            'complete guide for', 'full process of', 'entire process of',
            'everything about', 'all about', 'what you know about',
            'share knowledge about', 'provide details on', 'give guidance on',
            'offer advice on', 'suggest methods for', 'recommend approach for'
        ]

        # Check if query is specifically asking about crop cultivation
        is_growing_query = any(phrase in normalized_query for phrase in growing_phrases)

        # Find all crops mentioned in the query
        mentioned_crops = []
        for crop in all_crops:
            crop_lower = crop.lower()
            # Check if crop is mentioned in query (as whole word)
            pattern = r'(^|\W)' + re.escape(crop_lower) + r'(\W|$)'
            if re.search(pattern, normalized_query):
                mentioned_crops.append(crop)

        # If it's a growing query and at least one crop is mentioned
        if is_growing_query and mentioned_crops:
            if len(mentioned_crops) == 1:
                return self.get_crop_steps(mentioned_crops[0])
            else:
                return self.get_crop_steps_for_multiple(mentioned_crops)

        # Also handle cases where query is just a crop name (implicit growing request)
        if len(mentioned_crops) == 1 and len(normalized_query.split()) <= 3:
            # If query is very short and contains a crop name, assume they want growing info
            return self.get_crop_steps(mentioned_crops[0])

        # Normalize the query further for general assistance
        normalized_query = query.lower().strip()

        # Check if the query is just a crop name or contains a crop name from suggestions
        matched_crops = []

        # First check if the entire query is a crop name
        if normalized_query in [c.lower() for c in suggested_crops]:
            matched_crops.append(normalized_query)
        else:
            # Check for partial matches with common prefixes
            common_prefixes = [
                'give me', 'tell me', 'show me', 'about',
                'for', 'details', 'steps', 'cultivation', 'assist me',
                'how to grow', 'growing', 'planting'
            ]

            # Check if query starts with any prefix followed by a crop name
            for crop in suggested_crops:
                crop_lower = crop.lower()

                # Case 1: Query is exactly the crop name
                if normalized_query == crop_lower:
                    matched_crops.append(crop_lower)
                    continue

                # Case 2: Query starts with common prefix followed by crop name
                for prefix in common_prefixes:
                    if normalized_query.startswith(prefix) and crop_lower in normalized_query[len(prefix):].strip():
                        matched_crops.append(crop_lower)
                        break

                # Case 3: Crop name appears anywhere in the query
                if crop_lower in normalized_query and crop_lower not in matched_crops:
                    # Make sure it's not part of another word
                    pattern = r'(^|\W)' + re.escape(crop_lower) + r'(\W|$)'
                    if re.search(pattern, normalized_query):
                        matched_crops.append(crop_lower)

        # Remove duplicates while preserving order
        matched_crops = list(dict.fromkeys(matched_crops))

        if matched_crops:
            return self.get_crop_steps_for_multiple(matched_crops)

        # Check if user just said "yes" or similar but also has follow-up questions
        if re.search(r'\b(yes|yeah|yup|yep|ok|okay|sure|please)\b', normalized_query):
            # Check if there's more to the query after the affirmation
            if len(normalized_query.split()) > 1:
                # Process the remaining part of the query
                remaining_query = re.sub(r'\b(yes|yeah|yup|yep|ok|okay|sure|please)\b', '', normalized_query).strip()
                if remaining_query:
                    assistance_response = self.process_assistance_request(remaining_query, suggested_crops)
                    if assistance_response:
                        return assistance_response

            # If no follow-up or no specific response, return all crop steps
            return self.get_crop_steps_for_multiple(suggested_crops)

        return None

    def _check_multi_query(self, query: str) -> List[str]:
        """Check if the query contains multiple questions"""
        # Common conjunctions that might indicate multiple queries
        conjunctions = ['and', 'also', 'as well as', 'plus', 'along with']

        # Check if any conjunction exists in the query
        for conj in conjunctions:
            if conj in query.lower():
                # Split the query into parts
                parts = re.split(fr'\b{conj}\b', query, flags=re.IGNORECASE)
                if len(parts) > 1:
                    return [part.strip() for part in parts if part.strip()]
        return None

    def _handle_multi_query(self, queries: List[str], suggested_crops: list) -> str:
        """Handle multiple queries in a single request"""
        responses = []
        for q in queries:
            # First check if this is a follow-up about crop cultivation
            assistance_response = self.process_assistance_request(q, suggested_crops)
            if assistance_response:
                responses.append(assistance_response)
                continue

            # Otherwise process as a regular query
            response = self.process_query(q)
            if response:
                responses.append(response)

        if responses:
            return "\n\n".join(responses)
        return None

    @staticmethod
    def clean_location_input(text: str) -> str:
        """Clean location strings for matching with dataset"""
        return text.lower().strip().replace(" ", "")

    def normalize_location(self, location: str) -> str:
        """Normalize location names using aliases with consistent cleaning"""
        if not location:
            return location

        # First clean the input (lowercase, strip, remove spaces)
        clean_loc = self.clean_location_input(location)

        # Then apply aliases if available
        for alias, canonical in self.location_aliases.items():
            if clean_loc == alias:
                return self.clean_location_input(canonical)

        return clean_loc

    def normalize_month(self, month: str) -> str:
        """Normalize month input to 3-letter format with validation"""
        if not month:
            return None

        month_lower = month.lower()

        # Handle "this month" specifically
        if month_lower == "this month":
            return datetime.now().strftime('%b').lower()

        normalized = self.month_map.get(month_lower, month_lower[:3] if len(month_lower) >= 3 else None)

        if normalized and normalized not in self.month_map.values():
            self.logger.warning(f"Invalid month format after normalization: {month} -> {normalized}")
            return None

        return normalized

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Improved entity extraction with better location and month handling"""
        prompt = f"""
        You are a natural language understanding engine for an agriculture assistant.

        Given the following query:
        "{query}"

        Extract entities as JSON with these keys:
        - "intent": One of ["soil_ph", "crop_recommendation", "soil_type", "rainfall", "moisture", "humidity", "irrigation", "fertilizer", "peak_price", "unknown"]
        - "district": Name of the district or city (like "Bombay", "Bangalore") if present
        - "state": Name of the state (like "Maharashtra", "Tamil Nadu") if present
        - "month": Month name (like "January", "Feb", "march", "this month") if mentioned
        - "year": The year if mentioned (use "this year" if user refers to current year)
        - "current_location": true if the user refers to "my location", "my district", or similar
        - "crop": Name of the crop if mentioned (like "wheat", "rice", "sesame")

        Important: If user says "this month" or "current month", set month to "this month".
        If user says "this year" or doesn't specify year, don't include year in response.

        Return only valid JSON. If a location like "Bombay" is used, do not correct it — just return as-is.
        """

        try:
            response = ollama.generate(
                model='llama2',
                prompt=prompt,
                format='json',
                system=SYSTEM_PROMPT,
                options={'temperature': 0.1}
            )

            raw_response = response.get('response')
            if not raw_response:
                raise ValueError("No response from model.")

            entities = json.loads(raw_response)

            # Validate required fields
            if 'intent' not in entities:
                entities['intent'] = 'unknown'

            # Check if query refers to "my location" even if not parsed by model
            if any(phrase in query.lower() for phrase in
                   ["my location", "my district", "my soil", "my state", "my area"]):
                entities['current_location'] = True

            # Normalize location names if present
            if 'district' in entities and entities['district']:
                entities['district'] = self.normalize_location(entities['district'])
            if 'state' in entities and entities['state']:
                entities['state'] = self.normalize_location(entities['state'])

            # Normalize month if present
            if 'month' in entities and entities['month']:
                entities['month'] = self.normalize_month(entities['month'])
                if not entities['month']:
                    self.logger.warning(f"Could not normalize month: {entities['month']}")

            return entities
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {e}")
            return {"intent": "unknown"}

    @staticmethod
    def get_current_month_year() -> tuple:
        """Get current month and year in India timezone"""
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        return now.strftime("%b").lower(), now.year

    def get_soil_ph(self, state: str, district: str, month: str = None) -> float:
        """Predict soil pH using the trained model with proper month handling"""
        # Normalize locations first
        norm_state = self.normalize_location(state)
        norm_district = self.normalize_location(district)
        # Normalize month input
        normalized_month = self.normalize_month(month) if month else None
        if not normalized_month:
            normalized_month, _ = self.get_current_month_year()
            self.logger.info(f"Using current month {normalized_month} as fallback")

        input_df = pd.DataFrame([{
            'state': self.clean_location_input(state),
            'district': self.clean_location_input(district),
            'month': normalized_month
        }])

        try:
            # Preprocess input
            input_df[['state', 'district', 'month']] = input_df[['state', 'district', 'month']].apply(
                lambda x: x.str.lower().str.strip().str.replace(" ", ""))

            # Convert month to cyclical features
            input_df['month_sin'] = np.sin(
                2 * np.pi * (pd.to_datetime(input_df['month'], format='%b').dt.month - 1) / 12)
            input_df['month_cos'] = np.cos(
                2 * np.pi * (pd.to_datetime(input_df['month'], format='%b').dt.month - 1) / 12)

            final_features = ['state', 'district', 'month_sin', 'month_cos']
            ph_value = round(self.model_ph.predict(input_df[final_features])[0], 2)
            self.logger.info(f"Predicted pH {ph_value} for {district}, {state} in {normalized_month}")
            return ph_value
        except Exception as e:
            self.logger.error(f"Error in soil pH prediction: {e}")
            # Fallback to district average if month processing fails
            try:
                avg_ph = self.augmented_df[
                    (self.augmented_df['state'].str.lower().str.replace(" ", "") == self.clean_location_input(state)) &
                    (self.augmented_df['district'].str.lower().str.replace(" ", "") == self.clean_location_input(
                        district))
                    ]['soil_ph'].mean()
                fallback_ph = round(avg_ph, 2) if not pd.isna(avg_ph) else 6.5
                self.logger.info(f"Using fallback pH {fallback_ph} for {district}, {state}")
                return fallback_ph
            except Exception as fallback_error:
                self.logger.error(f"Fallback failed: {fallback_error}")
                return 6.5  # Final fallback

    def get_soil_type(self, state: str, district: str) -> str:
        """Get soil type from dataset with fallback and proper location normalization"""
        # Normalize inputs first
        norm_state = self.normalize_location(state)
        norm_district = self.normalize_location(district)

        try:
            # Try with normalized names first
            soil_type = self.augmented_df[
                (self.augmented_df['state'].str.lower().str.replace(" ", "") == norm_state) &
                (self.augmented_df['district'].str.lower().str.replace(" ", "") == norm_district)
                ]['soil_type'].iloc[0]
            return soil_type.lower().replace(" ", "")
        except IndexError:
            self.logger.warning(f"Soil type not found for {norm_district}, {norm_state} - trying state default")
            # Try to get most common soil type for the state as a better fallback
            try:
                state_soil = self.augmented_df[
                    (self.augmented_df['state'].str.lower().str.replace(" ", "") == norm_state)
                ]['soil_type'].mode()[0]
                return state_soil.lower().replace(" ", "")
            except:
                return "loamy"

    def preprocess_input(self, user_input_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input for crop prediction with month normalization"""
        user_input_df = user_input_df.copy()
        user_input_df[['state', 'district', 'soil_type', 'month']] = user_input_df[
            ['state', 'district', 'soil_type', 'month']].apply(
            lambda x: x.str.lower().str.strip().str.replace(" ", ""))

        # Ensure month is normalized
        user_input_df['month'] = user_input_df['month'].apply(
            lambda m: self.normalize_month(m) or self.get_current_month_year()[0])

        # Cyclical month features
        try:
            user_input_df['month_sin'] = np.sin(
                2 * np.pi * (pd.to_datetime(user_input_df['month'], format='%b').dt.month - 1) / 12)
            user_input_df['month_cos'] = np.cos(
                2 * np.pi * (pd.to_datetime(user_input_df['month'], format='%b').dt.month - 1) / 12)
        except Exception as e:
            self.logger.error(f"Error processing month features: {e}")
            # Fallback to current month
            current_month = self.get_current_month_year()[0]
            user_input_df['month'] = current_month
            user_input_df['month_sin'] = np.sin(
                2 * np.pi * (pd.to_datetime(current_month, format='%b').dt.month - 1) / 12)
            user_input_df['month_cos'] = np.cos(
                2 * np.pi * (pd.to_datetime(current_month, format='%b').dt.month - 1) / 12)

        # Derived features
        user_input_df['fertility_index'] = np.where(
            (user_input_df['soil_ph'] > 6) & (user_input_df['soil_ph'] < 7.5) & (user_input_df['rainfall_mm'] > 500),
            1, 0
        )

        # District-month averages
        dm_avg_ph = self.augmented_df.groupby(['district', 'month'])['soil_ph'].mean().to_dict()
        dm_avg_rain = self.augmented_df.groupby(['district', 'month'])['rainfall_mm'].mean().to_dict()

        user_input_df['district_month_avg_ph'] = user_input_df.apply(
            lambda x: dm_avg_ph.get((x['district'], x['month']), x['soil_ph']), axis=1)
        user_input_df['district_month_avg_rain'] = user_input_df.apply(
            lambda x: dm_avg_rain.get((x['district'], x['month']), x['rainfall_mm']), axis=1)

        # Growing degree days
        base_temp = 10
        user_input_df['gdd'] = np.maximum(user_input_df['temperature_c'] - base_temp, 0)

        # Seasonal features
        user_input_df['is_monsoon'] = user_input_df['month'].isin(['jun', 'jul', 'aug', 'sep']).astype(int)

        # District encoding
        district_encoding = self.augmented_df.groupby('district')['crop'].agg(lambda x: x.value_counts().index[0])
        user_input_df['district_encoded'] = user_input_df['district'].map(district_encoding)

        final_features = [
            'soil_type', 'soil_ph', 'temperature_c', 'rainfall_mm',
            'month_sin', 'month_cos', 'fertility_index',
            'district_month_avg_ph', 'district_month_avg_rain',
            'gdd', 'is_monsoon', 'district_encoded'
        ]

        return user_input_df[final_features]

    def get_weather_data(self, latitude: float, longitude: float, year: int, month: str) -> tuple:
        """Get weather data with proper handling for current month's passed days"""
        try:
            # Normalize month input
            normalized_month = self.normalize_month(month) or self.get_current_month_year()[0]
            month_num = datetime.strptime(normalized_month, "%b").month
            current_month, current_year = self.get_current_month_year()
            current_month_num = datetime.strptime(current_month, "%b").month

            # Robust year handling
            if isinstance(year, str):
                year_lower = str(year).lower().strip()
                if year_lower in ["current year", "this year", "present year"]:
                    year = current_year
                else:
                    try:
                        # Extract digits only and convert
                        year_str = ''.join(filter(str.isdigit, year_lower))
                        year = int(year_str) if year_str else current_year
                        # Validate reasonable year range
                        if not (1900 <= year <= 2100):
                            year = current_year
                            self.logger.warning(
                                f"Year {year} out of range (1900-2100), using current year {current_year}")
                    except (ValueError, TypeError) as e:
                        year = current_year
                        self.logger.warning(f"Year conversion error: {str(e)}, using current year {current_year}")
            elif not isinstance(year, int):
                year = current_year
                self.logger.warning(f"Invalid year type {type(year)}, using current year {current_year}")

            # Determine date range for API request
            if year == current_year and month_num == current_month_num:
                # Current month - only get data up to yesterday
                today = datetime.now(pytz.timezone('Asia/Kolkata')).date()
                start_date = f"{year}-{month_num:02d}-01"
                end_date = f"{year}-{month_num:02d}-{(today.day - 1):02d}"
                self.logger.info(f"Getting partial month data for {normalized_month} {year}: days 1-{today.day - 1}")
                is_partial_month = True
            else:
                # Historical month - get full month data
                days_in_month = 31 if month_num in [1, 3, 5, 7, 8, 10, 12] else 30 if month_num != 2 else 28
                start_date = f"{year}-{month_num:02d}-01"
                end_date = f"{year}-{month_num:02d}-{days_in_month}"
                self.logger.info(f"Getting full month data for {normalized_month} {year}")
                is_partial_month = False

            # Handle future dates by using historical data
            if (year > current_year) or (year == current_year and month_num > current_month_num):
                self.logger.info(f"Requested future date {normalized_month} {year} - falling back to historical data")
                historical_year = current_year - 1
                start_date = f"{historical_year}-{month_num:02d}-01"
                end_date = f"{historical_year}-{month_num:02d}-{days_in_month}"
                year = historical_year

            # API request
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': start_date,
                'end_date': end_date,
                'daily': 'temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean',
                'timezone': 'auto'
            }

            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'daily' not in data:
                    raise ValueError("No daily data in API response")

                # Calculate averages from available data
                temp_values = [x for x in data['daily']['temperature_2m_mean'] if x is not None]
                rain_values = [x for x in data['daily']['precipitation_sum'] if x is not None]
                humidity_values = [x for x in data['daily']['relative_humidity_2m_mean'] if x is not None]

                if not temp_values or not rain_values or not humidity_values:
                    raise ValueError("Missing weather data in API response")

                temp_avg = np.mean(temp_values)
                monthly_rainfall = sum(rain_values)
                humidity_avg = np.mean(humidity_values)

                # Scale rainfall estimate for partial months
                if is_partial_month:
                    days_in_month = 31 if month_num in [1, 3, 5, 7, 8, 10, 12] else 30 if month_num != 2 else 28
                    days_received = len(data['daily']['time'])
                    if 0 < days_received < days_in_month:
                        scale_factor = days_in_month / days_received
                        monthly_rainfall *= scale_factor
                        self.logger.info(f"Scaled rainfall estimate by {scale_factor:.2f}x for partial month data")

                return temp_avg, monthly_rainfall, humidity_avg

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Weather API request failed: {e}")
                raise ValueError("Weather service unavailable")

        except Exception as e:
            self.logger.error(f"Weather data processing error: {e}")
            # Provide reasonable defaults based on month
            if normalized_month in ['jun', 'jul', 'aug', 'sep']:  # Monsoon season
                return 28.0, 150.0, 75.0
            elif normalized_month in ['mar', 'apr', 'may']:  # Summer
                return 35.0, 30.0, 50.0
            else:  # Winter
                return 25.0, 50.0, 60.0

    def get_crop_recommendation(self, state: str, district: str, month: str = None, year: int = None):
        """Get comprehensive crop recommendation with proper month handling"""
        # Get current date information
        current_month, current_year = self.get_current_month_year()

        # Handle month
        normalized_month = self.normalize_month(month) if month else current_month

        # Handle year - use current year if not specified
        year = year or current_year

        self.logger.info(f"Using month {normalized_month} {year} for crop recommendation")

        # Get coordinates - try automatically first
        try:
            g = geocoder.ip('me')
            lat, lon = g.latlng if g.ok else (20.5937, 78.9629)  # India center as fallback
        except Exception:
            lat, lon = (20.5937, 78.9629)

        # Get weather, soil data
        temperature, rainfall, humidity = self.get_weather_data(lat, lon, year, normalized_month)
        soil_type = self.get_soil_type(state, district)
        soil_ph = self.get_soil_ph(state, district, normalized_month)

        # Prepare and preprocess input
        user_input = pd.DataFrame([{
            'state': self.clean_location_input(state),
            'district': self.clean_location_input(district),
            'soil_type': soil_type,
            'month': normalized_month.lower(),
            'soil_ph': soil_ph,
            'rainfall_mm': rainfall,
            'temperature_c': temperature
        }])

        try:
            X_input = self.preprocess_input(user_input)
            predicted_category = self.label_encoder.inverse_transform(self.model.predict(X_input))[0]
        except Exception as e:
            self.logger.error(f"Error in crop prediction: {e}")
            predicted_category = "cereals"  # Default fallback category

        return {
            'state': state,
            'district': district,
            'category': predicted_category,
            'soil_type': soil_type,
            'soil_ph': soil_ph,
            'temperature': temperature,
            'rainfall': rainfall,
            'humidity': humidity,
            'month': normalized_month,
            'year': year
        }

    def format_crop_recommendation(self, result: dict) -> str:
        """Format crop recommendation response with specific crop suggestions"""
        category = result['category'].lower()
        crops = self.crop_categories.get(category, [])

        if not crops:
            return (
                f"In {result['district'].title()}, {result['state'].title()} during {result['month'].capitalize()} {result['year']}, "
                f"the average soil pH is {result['soil_ph']:.2f} and the dominant soil type is {result['soil_type']} soil. "
                f"Weather conditions are expected to be around {result['temperature']:.1f}°C with approximately {result['rainfall']:.1f}mm of rainfall "
                f"and {result['humidity']:.1f}% humidity.\n\n"
                f"Based on these conditions, the most suitable crops would be from the {category.capitalize()} category. "
                f"However, we don't have specific crop suggestions for this category at the moment.")

        # Select up to 5 crops to suggest
        suggested_crops = crops[:5]
        if len(crops) > 5:
            crop_list = ', '.join(suggested_crops[:-1]) + f", and {suggested_crops[-1]}"
        else:
            crop_list = ', '.join(suggested_crops)

        response = (
            f"In {result['district'].title()}, {result['state'].title()} during {result['month'].capitalize()} {result['year']}, "
            f"the average soil pH is {result['soil_ph']:.2f} and the dominant soil type is {result['soil_type']} soil. "
            f"Weather conditions are expected to be around {result['temperature']:.1f}°C with approximately {result['rainfall']:.1f}mm of rainfall "
            f"and {result['humidity']:.1f}% humidity.\n\n"
            f"Based on these conditions, you can grow crops like {crop_list}. "
            f"This recommendation takes into account the local soil characteristics and typical weather patterns for this time of year. Would you like my assistance in guiding you through their complete cultivation process?")

        # Store suggested crops for potential follow-up
        self.last_suggested_crops = suggested_crops

        return response

    async def process_query(self, query: str, input_lang: str = 'en') -> str:
        """Improved query processing with strict location and month handling"""
        # First translate to English if needed
        if input_lang != 'en':
            query = await self.translate_to_english(query, input_lang)

        # First check if this is a multi-query request
        multi_query = self._check_multi_query(query)
        if multi_query:
            responses = []
            for q in multi_query:
                response = self._process_single_query(q)
                if response:
                    responses.append(response)
            if responses:
                return "\n\n".join(responses)
            return "I couldn't process your multi-part query. Please try asking one question at a time."

        # Process as single query
        return self._process_single_query(query)

    def _process_single_query(self, query: str) -> str:
        """Process a single query (not containing multiple questions)"""
        normalized_query = query.lower().strip()

        # ✅ Always allow crop growing queries, even without previous recommendation
        all_crops = set(self.crop_steps_df['crop_name'].str.lower().unique())
        growing_phrases = [
            'how to grow', 'growing', 'planting', 'cultivation',
            'steps for', 'process for', 'guide for', 'tell me about',
            'help me with', 'assist me with', 'information about',
            'details about', 'can you explain', 'what are the steps',
            'how do i grow', 'how can i grow', 'i want to grow',
            'teach me about', 'explain', 'describe', 'walk me through',
            'procedure for', 'method for', 'technique for',
            'tips for growing', 'advice for growing', 'recommendations for',
            'best way to grow', 'proper way to grow', 'right way to grow',
            'complete guide for', 'full process of', 'entire process of',
            'everything about', 'all about', 'what you know about',
            'share knowledge about', 'provide details on', 'give guidance on',
            'offer advice on', 'suggest methods for', 'recommend approach for'
        ]

        is_growing_query = any(phrase in normalized_query for phrase in growing_phrases)
        mentioned_crops = [crop for crop in all_crops if
                           re.search(r'(^|\W)' + re.escape(crop) + r'(\W|$)', normalized_query)]

        if is_growing_query and mentioned_crops:
            if len(mentioned_crops) == 1:
                return self.get_crop_steps(mentioned_crops[0])
            else:
                return self.get_crop_steps_for_multiple(mentioned_crops)

        # ✅ Keep old follow-up assistance behavior
        if hasattr(self, 'last_suggested_crops'):
            assistance_response = self.process_assistance_request(query, self.last_suggested_crops)
            if assistance_response:
                return assistance_response

        entities = self.extract_entities(query)
        self.logger.info(f"Extracted entities: {entities}")

        # Handle irrigation, fertilizer and peak price queries for specific crops
        if 'crop' in entities and entities['crop']:
            crop_name = entities['crop'].lower().strip()

            if entities['intent'] == 'irrigation':
                return self.get_irrigation_info(crop_name)
            elif entities['intent'] == 'fertilizer':
                return self.get_fertilizer_info(crop_name)
            elif entities['intent'] == 'peak_price':
                return self.get_peak_price_info(crop_name)

        # Handle location detection
        if entities.get('current_location') or ('district' not in entities and 'state' not in entities):
            try:
                print("\nAttempting to detect your current location...")
                g = geocoder.ip('me')
                if g.ok and g.city and g.state:
                    detected_state = g.state.lower().strip().replace(" ", "")
                    detected_district = g.city.lower().strip().replace(" ", "")
                    print(f"Detected location: {detected_district.title()}, {detected_state.title()}")

                    # Only use detected location if we're dealing with "my location" query
                    if "my location" in query.lower() or "my district" in query.lower():
                        entities['district'] = detected_district
                        entities['state'] = detected_state
                else:
                    raise ValueError("Could not detect location.")
            except Exception as e:
                self.logger.error(f"Location detection failed: {e}")
                if not entities.get('district') or not entities.get('state'):
                    return "Sorry, I couldn't detect your location. Please specify your district and state in your query."

        # Validate we have location
        if 'district' not in entities or 'state' not in entities:
            return ("I need to know your location to provide accurate information. "
                    "Please include your district and state in your question.")

        # Normalize locations one final time before processing
        entities['district'] = self.normalize_location(entities['district'])
        entities['state'] = self.normalize_location(entities['state'])

        try:
            current_month, current_year = self.get_current_month_year()

            if entities['intent'] == 'soil_ph':
                month = entities.get('month')
                if month:
                    month = self.normalize_month(month)
                    if not month:
                        return "Invalid month specified. Please use a valid month like January, Feb, etc."
                else:
                    month = current_month

                ph_value = self.get_soil_ph(
                    entities['state'],
                    entities['district'],
                    month
                )
                month_display = month.capitalize() if month else "the current month"
                return (f"The estimated soil pH for {entities['district'].title()}, "
                        f"{entities['state'].title()} in {month_display} is approximately {ph_value:.2f}. "
                        f"This helps determine which crops will grow well in this area.")

            elif entities['intent'] == 'soil_type':
                soil_type = self.get_soil_type(
                    entities['state'],
                    entities['district']
                )
                return (f"The predominant soil type in {entities['district'].title()}, "
                        f"{entities['state'].title()} is {soil_type}. "
                        f"Different soil types are suitable for different crops.")

            elif entities['intent'] == 'rainfall':
                month = entities.get('month')
                year = entities.get('year') or current_year
                month = self.normalize_month(month) if month else current_month

                try:
                    g = geocoder.ip('me')
                    lat, lon = g.latlng if g.ok else (20.5937, 78.9629)
                except:
                    lat, lon = (20.5937, 78.9629)

                _, rainfall, _ = self.get_weather_data(lat, lon, year, month)
                return (f"The estimated rainfall in {entities['district'].title()}, {entities['state'].title()} "
                        f"during {month.capitalize()} {year} is approximately {rainfall:.1f}mm.")

            elif entities['intent'] in ['moisture', 'humidity']:
                month = entities.get('month')
                year = entities.get('year') or current_year
                month = self.normalize_month(month) if month else current_month

                try:
                    g = geocoder.ip('me')
                    lat, lon = g.latlng if g.ok else (20.5937, 78.9629)
                except:
                    lat, lon = (20.5937, 78.9629)

                _, _, humidity = self.get_weather_data(lat, lon, year, month)
                return (
                    f"The estimated relative humidity in {entities['district'].title()}, {entities['state'].title()} "
                    f"during {month.capitalize()} {year} is approximately {humidity:.1f}%.")

            elif entities['intent'] == 'temperature':
                month = entities.get('month')
                year = entities.get('year') or current_year
                month = self.normalize_month(month) if month else current_month

                try:
                    g = geocoder.ip('me')
                    lat, lon = g.latlng if g.ok else (20.5937, 78.9629)
                except:
                    lat, lon = (20.5937, 78.9629)

                temp, _, _ = self.get_weather_data(lat, lon, year, month)
                return (
                    f"The estimated temperature in {entities['district'].title()}, {entities['state'].title()} "
                    f"during {month.capitalize()} {year} is approximately {temp:.1f}°C.")

            elif entities['intent'] == 'crop_recommendation':
                month = entities.get('month')
                if month:
                    month = self.normalize_month(month)
                    if not month:
                        return "Invalid month specified. Please use a valid month like January, Feb, etc."
                else:
                    month = current_month

                year = entities.get('year') or current_year

                result = self.get_crop_recommendation(
                    entities['state'],
                    entities['district'],
                    month,
                    year
                )
                return self.format_crop_recommendation(result)

            elif entities['intent'] == 'peak_price':
                if 'crop' in entities and entities['crop']:
                    return self.get_peak_price_info(entities['crop'])
                elif hasattr(self, 'last_suggested_crops'):
                    responses = []
                    for crop in self.last_suggested_crops:
                        responses.append(self.get_peak_price_info(crop))
                    return "\n\n".join(responses)
                else:
                    return "Please specify which crop you want peak price information for."

            return ("I can help with:\n"
                    "- Soil pH information for your location\n"
                    "- Crop recommendations based on soil and weather conditions\n"
                    "- Soil type information for different regions\n"
                    "- Rainfall and humidity information for specific locations\n"
                    "- Irrigation requirements for specific crops\n"
                    "- Fertilizer recommendations for specific crops\n"
                    "- Peak price and demand month information for specific crops\n"
                    "Please ask a specific question including your location.")

        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            return ("I encountered an issue processing your request. "
                    "Please try again with a different query or more specific location details.")


# Initialize the agent globally
agent = None

def create_app():
    global agent
    try:
        agent = AgriculturalAgent()
        app.logger.info("Agricultural Agent initialized successfully")
    except Exception as e:
        app.logger.error(f"Failed to initialize AgriculturalAgent: {e}")
        raise

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/api/voice', methods=['POST'])
async def handle_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400

        audio_file = request.files['audio']

        # Create temp files
        temp_dir = tempfile.gettempdir()
        temp_input = os.path.join(temp_dir, f"voice_input_{random.randint(0, 10000)}.webm")
        temp_wav = os.path.join(temp_dir, f"voice_input_{random.randint(0, 10000)}.wav")

        # Save original
        audio_file.save(temp_input)

        # Convert to WAV
        try:
            sound = AudioSegment.from_file(temp_input, format="webm")
            sound = sound.set_frame_rate(16000).set_channels(1)
            sound.export(temp_wav, format="wav", codec="pcm_s16le")
        except Exception as e:
            app.logger.error(f"Audio conversion failed: {e}")
            return jsonify({"error": "Audio conversion error"}), 400

        # Recognize speech
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 4000
        recognizer.dynamic_energy_threshold = True

        try:
            with sr.AudioFile(temp_wav) as source:
                audio_data = recognizer.record(source)

                # Try Indian languages
                lang_codes = ["hi-IN", "en-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"]
                text = None
                detected_lang = "en"

                for code in lang_codes:
                    try:
                        text = recognizer.recognize_google(audio_data, language=code)
                        detected_lang = code.split("-")[0]
                        break
                    except:
                        continue

        except Exception as e:
            app.logger.error(f"Speech recognition failed: {e}")
            return jsonify({"error": "Speech recognition error"}), 400

        finally:
            # Cleanup temp files
            for f in [temp_input, temp_wav]:
                try:
                    os.remove(f)
                except:
                    pass

        if not text:
            return jsonify({"error": "Could not understand audio"}), 400

        try:
            # Translate to English
            english_text = await agent.translate_to_english(text, detected_lang)

            # Process query
            response_text = await agent.process_query(english_text, input_lang='en')

            # Translate back to original language
            translated_response = await agent.translate_from_english(response_text, detected_lang)

            # Generate audio response
            audio_path = os.path.join(temp_dir, f"response_{random.randint(0, 10000)}.mp3")
            tts = gTTS(text=translated_response, lang=detected_lang)
            tts.save(audio_path)

            return jsonify({
                "transcript": text,
                "language": detected_lang,
                "response": translated_response,
                "audio_url": f"/temp/{os.path.basename(audio_path)}"
            })

        except Exception as e:
            app.logger.error(f"Translation error: {e}")
            return jsonify({"error": "Translation service error"}), 500

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500
@app.route('/temp/<filename>')
def serve_temp_file(filename):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=False)
    else:
        return jsonify({"error": "File not found"}), 404


@app.route('/api/voice_query', methods=['POST'])
async def handle_text_to_speech():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # Detect language of the text
        detected_lang = translator.detect(text).lang

        # Generate speech
        audio_path = agent.voice_handler.text_to_speech(text, detected_lang)

        if not audio_path:
            return jsonify({'error': 'Failed to generate speech'}), 500

        return jsonify({
            'audio_url': f"/temp/{os.path.basename(audio_path)}"
        })

    except Exception as e:
        app.logger.error(f"Text-to-speech error: {e}")
        return jsonify({'error': 'Text-to-speech conversion failed'}), 500
@app.route('/api/query', methods=['POST'])
async def process_query():
    data = request.get_json()  # no await here
    query = data.get('query', '').strip()
    input_lang = data.get('language', 'en')

    if not query:
        return jsonify({'error': 'Empty query'}), 400

    try:
        response_text = await agent.process_query(query, input_lang)
        return jsonify({'response': response_text})
    except Exception as e:
        app.logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred while processing your query'}), 500

@app.route('/api/voice_query', methods=['POST'])
def process_voice_query():
    try:
        # Get the audio file from the request
        audio_file = request.files['audio']

        # Save temporarily
        temp_audio_path = os.path.join(tempfile.gettempdir(), f"voice_input_{random.randint(0, 10000)}.wav")
        audio_file.save(temp_audio_path)

        # Recognize speech
        with sr.AudioFile(temp_audio_path) as source:
            audio = agent.voice_handler.recognizer.record(source)

        try:
            # Try with automatic language detection
            text = agent.voice_handler.recognizer.recognize_google(audio, language=None)
            detected_lang = translator.detect(text).lang
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400

        # Process the query (will be translated to English internally)
        response = agent.process_query(text, detected_lang)

        # Translate response back to original language if needed
        if detected_lang != 'en':
            response = agent.translate_from_english(response, detected_lang)

        # Generate speech response
        audio_response_path = agent.voice_handler.text_to_speech(response, detected_lang)

        # Clean up temp files
        os.remove(temp_audio_path)

        if audio_response_path:
            return jsonify({
                'text_response': response,
                'audio_response': audio_response_path,
                'language': detected_lang
            })
        else:
            return jsonify({
                'text_response': response,
                'language': detected_lang
            })

    except Exception as e:
        app.logger.error(f"Error processing voice query: {e}")
        return jsonify({'error': 'An error occurred while processing your voice query'}), 500

if __name__ == '__main__':
    create_app()
    app.run(debug=True)