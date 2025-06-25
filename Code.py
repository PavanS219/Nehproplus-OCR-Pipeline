#!/usr/bin/env python3
"""
Medical Report OCR Parser with Ollama
Extract text from medical reports and convert to JSON using local Ollama with EasyOCR
"""

import os
import cv2
import numpy as np
import json
import requests
from pathlib import Path
from datetime import datetime
import easyocr
import traceback

# ====== CONFIGURATION VARIABLES ======
INPUT_IMAGE = "C:/Users/Pavan/Desktop/input_image.jpeg"
OUTPUT_FILE = "extracted_lab_report_formatted.json"
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2:1b"  # Model to use
DEBUG_MODE = True                # Enable detailed debugging output
# =====================================

class MedicalReportOCR:
    def __init__(self, ollama_url=OLLAMA_BASE_URL, model_name=OLLAMA_MODEL):
        """Initialize OCR processor and Ollama client"""
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Initialize EasyOCR reader
        try:
            print("🔄 Initializing EasyOCR reader...")
            self.ocr_reader = easyocr.Reader(['en'])
            print("✅ EasyOCR initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize EasyOCR: {e}")
            raise
        
        # Test Ollama connection
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {ollama_url}")
                
                # Check if model is available
                models = [model['name'] for model in response.json().get('models', [])]
                if model_name in models:
                    print(f"✅ Model {model_name} is available")
                else:
                    print(f"⚠️  Model {model_name} not found. Available models: {models}")
                    print(f"   Run: ollama pull {model_name}")
            else:
                print(f"❌ Failed to connect to Ollama at {ollama_url}")
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
            print("   Make sure Ollama is running: ollama serve")
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(sharpened)
            
            if DEBUG_MODE:
                print(f"   🖼️  Image preprocessing completed successfully")
            
            return enhanced
        except Exception as e:
            print(f"   ❌ Image preprocessing error: {e}")
            raise
    
    def extract_text_easyocr(self, image_path):
        """Extract text using EasyOCR"""
        try:
            if DEBUG_MODE:
                print(f"   🔍 Starting EasyOCR text extraction...")
            
            # Process image with preprocessing
            processed_img = self.preprocess_image(image_path)
            
            # Extract text using EasyOCR
            results = self.ocr_reader.readtext(processed_img)
            
            extracted_texts = []
            full_text_parts = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence results
                    cleaned_text = text.strip()
                    if cleaned_text:
                        extracted_texts.append({
                            'text': cleaned_text,
                            'confidence': round(confidence * 100, 2),
                            'bbox': bbox
                        })
                        full_text_parts.append(cleaned_text)
            
            # Combine all text
            full_text = ' '.join(full_text_parts)
            
            if DEBUG_MODE:
                print(f"   📝 EasyOCR extracted {len(extracted_texts)} text blocks")
                print(f"   📏 Full text length: {len(full_text)} characters")
                if len(full_text) > 0:
                    preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
                    print(f"   👀 Text preview: {preview}")
            
            return full_text, extracted_texts
            
        except Exception as e:
            print(f"   ❌ OCR extraction failed: {e}")
            if DEBUG_MODE:
                print(f"   🔧 Full error traceback:")
                traceback.print_exc()
            return "", []
    
    def generate_json_with_ollama(self, extracted_text, image_filename):
        """Use Ollama to convert extracted text to structured JSON"""
        
        if DEBUG_MODE:
            print(f"   🤖 Starting Ollama processing...")
            print(f"   📊 Input text length: {len(extracted_text)} characters")
        
        # Truncate text if too long to avoid token limits
        max_text_length = 6000  # Reduced for 1b model
        if len(extracted_text) > max_text_length:
            extracted_text = extracted_text[:max_text_length] + "\n[TEXT TRUNCATED DUE TO LENGTH]"
            if DEBUG_MODE:
                print(f"   ✂️  Text truncated to {max_text_length} characters")
        
        prompt = f"""You are an expert medical report parser. I have extracted text from a medical report image using OCR. Please analyze this text and convert it into a well-structured JSON format.

The extracted text from the medical report is:
{extracted_text}

Please create a comprehensive JSON structure that includes:

1. **hospital_info**: Hospital name, address, phone, website, etc.
2. **patient_info**: Patient details like name, age, gender, ID, etc.
3. **doctor_info**: Referring doctor, consultant, pathologist, etc.
4. **report_info**: Report type, dates (collection, report), sample info, etc.
5. **test_results**: Array of all tests with:
   - test_name
   - result_value  
   - reference_range
   - unit
   - status (normal/abnormal if determinable)
6. **additional_info**: Any notes, interpretations, or other relevant information

Guidelines:
- Extract ALL available information from the text
- If a field is not found, include it with null value
- For test results, try to identify patterns like "TEST_NAME VALUE RANGE UNIT"
- Preserve exact values and ranges as found in the text
- Clean up obvious OCR errors where possible
- Make the JSON as comprehensive and accurate as possible

Return ONLY the JSON structure, no additional text or explanations."""

        try:
            if DEBUG_MODE:
                print(f"   📡 Sending request to Ollama API...")
                print(f"   🔗 URL: {self.ollama_url}/api/generate")
                print(f"   🏷️  Model: {self.model_name}")
            
            # Call Ollama API
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1024  # Reduced for 1b model
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_data,
                timeout=120
            )
            
            if DEBUG_MODE:
                print(f"   📥 Ollama response status: {response.status_code}")
                print(f"   📏 Response content length: {len(response.text)} characters")
            
            if response.status_code != 200:
                error_msg = f'Ollama API error: HTTP {response.status_code}'
                if DEBUG_MODE:
                    print(f"   ❌ {error_msg}")
                    print(f"   📄 Response headers: {dict(response.headers)}")
                    print(f"   📄 Response content: {response.text[:500]}...")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': response.text,
                    'status_code': response.status_code
                }
            
            # Parse the response
            try:
                result = response.json()
                if DEBUG_MODE:
                    print(f"   ✅ Successfully parsed Ollama response JSON")
                    print(f"   🔑 Response keys: {list(result.keys())}")
            except json.JSONDecodeError as e:
                error_msg = f'Failed to parse Ollama response as JSON: {str(e)}'
                if DEBUG_MODE:
                    print(f"   ❌ {error_msg}")
                    print(f"   📄 Raw response: {response.text[:1000]}...")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': response.text
                }
            
            json_text = result.get('response', '').strip()
            
            if DEBUG_MODE:
                print(f"   📏 JSON text length: {len(json_text)} characters")
                if len(json_text) == 0:
                    print(f"   ⚠️  WARNING: Empty response from Ollama!")
                    print(f"   🔍 Full Ollama result: {result}")
                else:
                    preview = json_text[:300] + "..." if len(json_text) > 300 else json_text
                    print(f"   👀 JSON preview: {preview}")
            
            if not json_text:
                return {
                    'success': False,
                    'error': 'Empty response from Ollama',
                    'raw_response': json_text,
                    'full_ollama_result': result
                }
            
            # Clean up the response to get just the JSON
            original_json_text = json_text
            
            if json_text.startswith('```json'):
                json_text = json_text[7:]
                if DEBUG_MODE:
                    print(f"   🧹 Removed ```json prefix")
            elif json_text.startswith('```'):
                json_text = json_text[3:]
                if DEBUG_MODE:
                    print(f"   🧹 Removed ``` prefix")
            
            if json_text.endswith('```'):
                json_text = json_text[:-3]
                if DEBUG_MODE:
                    print(f"   🧹 Removed ``` suffix")
            
            json_text = json_text.strip()
            
            if DEBUG_MODE and json_text != original_json_text:
                print(f"   🧹 Cleaned JSON text length: {len(json_text)} characters")
            
            # Try to find JSON in the response if it's not at the beginning
            if not json_text.startswith('{') and not json_text.startswith('['):
                # Look for JSON patterns in the text
                import re
                json_match = re.search(r'(\{.*\})', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                    if DEBUG_MODE:
                        print(f"   🔍 Extracted JSON from response using regex")
                else:
                    if DEBUG_MODE:
                        print(f"   ⚠️  No JSON structure found in response!")
                        print(f"   📄 Full cleaned text: {json_text}")
            
            # Parse and return JSON
            try:
                parsed_json = json.loads(json_text)
                if DEBUG_MODE:
                    print(f"   ✅ Successfully parsed JSON structure")
                    print(f"   🔑 JSON keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'}")
            except json.JSONDecodeError as e:
                error_msg = f'JSON parsing error: {str(e)}'
                if DEBUG_MODE:
                    print(f"   ❌ {error_msg}")
                    print(f"   📄 JSON text that failed to parse: {json_text[:500]}...")
                    print(f"   🔧 JSON error position: line {e.lineno}, column {e.colno}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': json_text,
                    'original_response': original_json_text,
                    'json_error_details': {
                        'line': e.lineno,
                        'column': e.colno,
                        'message': e.msg
                    }
                }
            
            # Add metadata
            parsed_json['_metadata'] = {
                'source_image': image_filename,
                'extraction_method': 'easyocr_ollama',
                'processing_timestamp': datetime.now().isoformat(),
                'model_used': self.model_name
            }
            
            if DEBUG_MODE:
                print(f"   🎉 JSON processing completed successfully!")
            
            return {
                'success': True,
                'json_data': parsed_json,
                'raw_response': json_text
            }
            
        except requests.RequestException as e:
            error_msg = f'Ollama request error: {str(e)}'
            if DEBUG_MODE:
                print(f"   ❌ {error_msg}")
                print(f"   🔧 Full error traceback:")
                traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
        except Exception as e:
            error_msg = f'Unexpected error in Ollama processing: {str(e)}'
            if DEBUG_MODE:
                print(f"   ❌ {error_msg}")
                print(f"   🔧 Full error traceback:")
                traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
    
    def process_image(self, image_path):
        """Process a single medical report image"""
        image_filename = os.path.basename(image_path)
        print(f"📄 Processing: {image_filename}")
        
        try:
            # Extract text using EasyOCR
            extracted_text, extraction_details = self.extract_text_easyocr(image_path)
            
            if not extracted_text.strip():
                error_msg = 'No text extracted from image'
                if DEBUG_MODE:
                    print(f"   ❌ {error_msg}")
                
                return {
                    'success': False,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'error': error_msg
                }
            
            print(f"   📝 Extracted {len(extraction_details)} text blocks")
            
            # Generate structured JSON using Ollama
            ollama_result = self.generate_json_with_ollama(extracted_text, image_filename)
            
            if ollama_result['success']:
                if DEBUG_MODE:
                    print(f"   ✅ Successfully generated JSON structure")
                
                return {
                    'success': True,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'extracted_text': extracted_text,
                    'extraction_details': extraction_details,
                    'structured_json': ollama_result['json_data'],
                    'ollama_raw_response': ollama_result['raw_response']
                }
            else:
                if DEBUG_MODE:
                    print(f"   ❌ Failed to generate JSON: {ollama_result['error']}")
                
                return {
                    'success': False,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'error': ollama_result['error'],
                    'extracted_text': extracted_text,
                    'ollama_raw_response': ollama_result.get('raw_response'),
                    'ollama_error_details': ollama_result
                }
                
        except Exception as e:
            error_msg = f'Processing error: {str(e)}'
            if DEBUG_MODE:
                print(f"   ❌ {error_msg}")
                print(f"   🔧 Full error traceback:")
                traceback.print_exc()
            
            return {
                'success': False,
                'image_path': image_path,
                'image_filename': image_filename,
                'error': error_msg
            }

def main():
    """Main processing function"""
    
    # Validate input image
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Input image not found: {INPUT_IMAGE}")
        return
    
    print(f"📂 Input image: {INPUT_IMAGE}")
    print(f"📂 Output file: {OUTPUT_FILE}")
    
    # Initialize OCR processor
    try:
        ocr_processor = MedicalReportOCR()
    except Exception as e:
        print(f"❌ Failed to initialize OCR processor: {str(e)}")
        if DEBUG_MODE:
            traceback.print_exc()
        return
    
    # Process the image
    print(f"\n{'='*50}")
    print(f"Processing Medical Report")
    print(f"{'='*50}")
    
    result = ocr_processor.process_image(INPUT_IMAGE)
    
    if result['success']:
        # Save structured JSON
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(result['structured_json'], f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Successfully saved: {OUTPUT_FILE}")
            
            # Display summary of extracted data
            json_data = result['structured_json']
            hospital_name = json_data.get('hospital_info', {})
            if isinstance(hospital_name, dict):
                hospital_name = hospital_name.get('hospital_name', 'N/A')
            
            patient_name = json_data.get('patient_info', {})
            if isinstance(patient_name, dict):
                patient_name = patient_name.get('name', 'N/A')
            
            test_results = json_data.get('test_results', [])
            test_count = len(test_results) if isinstance(test_results, list) else 0
            
            print(f"   📋 Hospital: {hospital_name}")
            print(f"   👤 Patient: {patient_name}")
            print(f"   🧪 Tests found: {test_count}")
            
            # Save raw text as well
            raw_text_file = OUTPUT_FILE.replace('.json', '_raw_text.txt')
            with open(raw_text_file, 'w', encoding='utf-8') as f:
                f.write(result['extracted_text'])
            print(f"   📝 Raw text saved: {raw_text_file}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to save JSON: {str(e)}")
            if DEBUG_MODE:
                traceback.print_exc()
            return False
    else:
        # Save error information
        error_data = {
            'error': result['error'],
            'image_path': result['image_path'],
            'extracted_text': result.get('extracted_text', ''),
            'timestamp': datetime.now().isoformat(),
            'debug_info': result.get('ollama_error_details', {})
        }
        
        error_file = OUTPUT_FILE.replace('.json', '_error.json')
        
        try:
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            print(f"   💾 Error details saved to: {error_file}")
        except Exception as e:
            print(f"   ❌ Failed to save error details: {str(e)}")
        
        print(f"   ❌ Failed to process: {result['error']}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Medical Report OCR Processing with EasyOCR + Ollama")
    print(f"📂 Input image: {INPUT_IMAGE}")
    print(f"📂 Output file: {OUTPUT_FILE}")
    print(f"🤖 Using: EasyOCR + Ollama {OLLAMA_MODEL}")
    print(f"🔧 Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    
    try:
        success = main()
        if success:
            print(f"\n✅ Processing completed successfully!")
            print(f"📄 Check output file: {OUTPUT_FILE}")
        else:
            print(f"\n❌ Processing failed. Check error file for details.")
    except KeyboardInterrupt:
        print("\n⏸️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if DEBUG_MODE:
            print("🔧 Full error traceback:")
            traceback.print_exc()
