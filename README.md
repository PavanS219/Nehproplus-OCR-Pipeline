# Nehproplus-OCR-Pipeline

A Python-based tool that extracts text from medical report images using EasyOCR and converts it into structured JSON format using local Ollama language models.

## 🚀 Features

- **OCR Text Extraction**: Uses EasyOCR for accurate text recognition from medical report images
- **Image Preprocessing**: Applies denoising, sharpening, and contrast enhancement for better OCR results
- **AI-Powered Structuring**: Leverages Ollama language models to convert raw OCR text into structured JSON
- **Comprehensive Data Extraction**: Extracts hospital info, patient details, doctor info, test results, and more
- **Debug Mode**: Detailed logging and debugging information
- **Error Handling**: Robust error handling with detailed error reporting
- **Multiple Output Formats**: Saves both structured JSON and raw extracted text

## 📋 Requirements

### System Requirements
- Python 3.7+
- Local Ollama installation
- Sufficient disk space for OCR models and language models

### Python Dependencies
```
opencv-python>=4.5.0
numpy>=1.21.0
requests>=2.25.0
easyocr>=1.6.0
```

### External Dependencies
- **Ollama**: Local language model server
- **EasyOCR Models**: Downloaded automatically on first run

## 🛠️ Installation

### 1. Install Python Dependencies
```bash
pip install opencv-python numpy requests easyocr
```

### 2. Install Ollama
Visit [Ollama's website](https://ollama.ai) and follow installation instructions for your OS.

#### For Linux/macOS:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### For Windows:
Download and install from the official website.

### 3. Download Language Model
```bash
# Start Ollama server
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:1b
```

### 4. Clone/Download the Script
Save the provided Python script as `medical_ocr_parser.py`

## ⚙️ Configuration

Edit the configuration variables at the top of the script:

```python
# ====== CONFIGURATION VARIABLES ======
INPUT_IMAGE = "C:/Users/Pavan/Desktop/input_image.jpeg" # Path to your medical report image
OUTPUT_FILE = "extracted_lab_report_formatted.json" # Output JSON file name
OLLAMA_BASE_URL = "http://localhost:11434" # Ollama server URL
OLLAMA_MODEL = "llama3.2:1b" # Model to use
DEBUG_MODE = True # Enable detailed logging
# =====================================
```

### Supported Image Formats
- JPEG/JPG
- PNG
- BMP
- TIFF
- WebP

### Available Ollama Models
- `llama3.2:1b` (lightweight, faster)
- `llama3.2:3b` (balanced performance)
- `llama3:8b` (higher accuracy, slower)
- `mistral:7b` (alternative option)

## 🚀 Usage

### Basic Usage
1. **Start Ollama server**:
   ```bash
   ollama serve
   ```

2. **Place your medical report image** in the specified path

3. **Run the script**:
   ```bash
   python medical_ocr_parser.py
   ```

### Command Line Example
```bash
# Start Ollama in background
ollama serve &

# Run the OCR parser
python medical_ocr_parser.py
```

## 📊 Output Structure

The script generates a structured JSON with the following sections:

```json
{
  "hospital_info": {
    "hospital_name": "Hospital Name",
    "address": "Hospital Address",
    "phone": "Contact Number",
    "website": "Website URL"
  },
  "patient_info": {
    "name": "Patient Name",
    "age": "Age",
    "gender": "Gender",
    "patient_id": "ID Number"
  },
  "doctor_info": {
    "referring_doctor": "Doctor Name",
    "consultant": "Consultant Name",
    "pathologist": "Pathologist Name"
  },
  "report_info": {
    "report_type": "Lab Report Type",
    "collection_date": "Sample Collection Date",
    "report_date": "Report Generation Date",
    "sample_type": "Sample Information"
  },
  "test_results": [
    {
      "test_name": "Test Name",
      "result_value": "Result Value",
      "reference_range": "Normal Range",
      "unit": "Unit of Measurement",
      "status": "normal/abnormal"
    }
  ],
  "additional_info": {
    "notes": "Additional Notes",
    "interpretations": "Medical Interpretations"
  },
  "_metadata": {
    "source_image": "input_image.jpeg",
    "extraction_method": "easyocr_ollama",
    "processing_timestamp": "2025-01-XX...",
    "model_used": "llama3.2:1b"
  }
}
```

## 📁 Output Files

The script generates multiple output files:

1. **`extracted_lab_report_formatted.json`**: Main structured JSON output
2. **`extracted_lab_report_formatted_raw_text.txt`**: Raw OCR extracted text
3. **`extracted_lab_report_formatted_error.json`**: Error details (if processing fails)


### For Better AI Processing
- Use appropriate model size for your hardware
- Adjust temperature settings for more consistent results
- Ensure sufficient RAM for the chosen model

### Model Recommendations by Hardware

| Hardware | Recommended Model | RAM Required |
|----------|------------------|--------------|
| Low-end laptop | `llama3.2:1b` | 2-4 GB |
| Mid-range system | `llama3.2:3b` | 4-8 GB |
| High-end system | `llama3:8b` | 8-16 GB |

## 📝 Example Usage

```python
# Initialize the OCR processor
ocr_processor = MedicalReportOCR(
    ollama_url="http://localhost:11434",
    model_name="llama3.2:1b"
)

# Process a medical report image
result = ocr_processor.process_image("path/to/medical_report.jpg")

if result['success']:
    print("✅ Processing successful!")
    structured_data = result['structured_json']
    # Use the structured data as needed
else:
    print(f"❌ Processing failed: {result['error']}")
```

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure compliance with medical data privacy regulations when processing actual medical reports.
