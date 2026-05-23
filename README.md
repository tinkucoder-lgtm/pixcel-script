# PixelScript

A web platform that takes AI-generated images and replaces all text fonts in one click.

## Project Structure

```
pixelscript/
├── frontend/          # Next.js React application
├── backend/           # FastAPI Python application
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## Prerequisites

- **Node.js** 16+ (for frontend)
- **Python** 3.9+ (for backend)
- **npm** or **yarn** (for frontend dependency management)

## Quick Start

### Clone and Install

```bash
# Frontend
cd frontend
npm install

# Backend
cd ../backend
pip install -r requirements.txt
```

### Run Development Servers

**Terminal 1 - Frontend (runs on http://localhost:3000):**
```bash
cd frontend
npm run dev
```

**Terminal 2 - Backend (runs on http://localhost:8000):**
```bash
cd backend
uvicorn main:app --reload
```

## Frontend Documentation

See [frontend/README.md](./frontend/README.md) for detailed frontend setup and development instructions.

## Backend Documentation

See [backend/README.md](./backend/README.md) for detailed backend setup and development instructions.

## Features

- Upload AI-generated images
- Automatic font detection
- Font replacement in one click
- Real-time preview
- Batch processing support

## Tech Stack

### Frontend
- Next.js 14+
- React 18+
- TypeScript
- Tailwind CSS

### Backend
- FastAPI
- Python 3.9+
- PIL/Pillow for image processing
- uvicorn (ASGI server)

## API Endpoints

- `POST /api/upload` - Upload image for processing
- `POST /api/process` - Process image with font replacement
- `GET /api/health` - Health check

## License

MIT
