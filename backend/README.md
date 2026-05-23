# PixelScript Backend

A FastAPI Python application for image processing and font replacement in AI-generated images.

## Tech Stack

- **FastAPI** - Modern Python web framework
- **Python** 3.9+ - Programming language
- **Pillow (PIL)** - Image processing library
- **uvicorn** - ASGI server
- **pydantic** - Data validation

## Getting Started

### Installation

First, create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Development

Run the development server:

```bash
uvicorn main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000)

API documentation will be at [http://localhost:8000/docs](http://localhost:8000/docs)

### Project Structure

```
backend/
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── routers/             # API route handlers
│   ├── upload.py       # Image upload endpoint
│   └── process.py      # Image processing endpoint
├── services/            # Business logic
│   └── image_processor.py
└── models/              # Pydantic models for validation
    └── schemas.py
```

## API Endpoints

### Health Check
- `GET /health` - Check API health status
- Response: `{"status": "ok"}`

### Image Upload
- `POST /api/upload` - Upload image file
- Request: multipart/form-data with image file
- Response: `{"file_id": "string", "filename": "string"}`

### Image Processing
- `POST /api/process` - Process image with font replacement
- Request body:
  ```json
  {
    "file_id": "string",
    "font_name": "string",
    "processing_options": {}
  }
  ```
- Response: `{"file_id": "string", "status": "completed", "result_url": "string"}`

## Environment Variables

Create a `.env` file in the backend directory:

```bash
DEBUG=True
API_HOST=0.0.0.0
API_PORT=8000
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=52428800
```

## Features

- [x] Health check endpoint
- [ ] Image upload with validation
- [ ] Font detection in images
- [ ] Font replacement engine
- [ ] Image processing pipeline
- [ ] Error handling and validation
- [ ] CORS support for frontend communication

## Dependencies

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support
- `pillow` - Image processing
- `pydantic` - Data validation
- `python-dotenv` - Environment configuration

## Running Tests

```bash
pytest
```

## CORS Configuration

The backend is configured to accept requests from the frontend at `http://localhost:3000` during development.

## Performance Notes

- File uploads are limited to 50MB by default
- Image processing may take time depending on image size and processing complexity
- Consider implementing async processing for large batches

## Troubleshooting

### Port Already in Use
If port 8000 is already in use:
```bash
uvicorn main:app --reload --port 8001
```

### Module Not Found Errors
Ensure your virtual environment is activated and all dependencies are installed:
```bash
pip install -r requirements.txt
```
