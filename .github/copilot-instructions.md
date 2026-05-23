# PixelScript Project Instructions

## Project Overview
PixelScript is a web platform that takes AI-generated images and replaces all text fonts in one click.

### Project Structure
- **frontend/**: Next.js React application with TypeScript
- **backend/**: FastAPI Python application for image processing

## Development Guidelines

### Frontend Development
- Location: `/frontend`
- Framework: Next.js with TypeScript
- Package Manager: npm
- Key Scripts:
  - `npm run dev` - Start development server
  - `npm run build` - Build for production
  - `npm run lint` - Run ESLint

### Backend Development
- Location: `/backend`
- Framework: FastAPI
- Python Version: 3.9+
- Key Commands:
  - `pip install -r requirements.txt` - Install dependencies
  - `uvicorn main:app --reload` - Start development server
  - Port: 8000

## Setup Instructions

### First Time Setup
1. Install Node.js 16+ (for frontend)
2. Install Python 3.9+ (for backend)
3. Follow setup steps in frontend/README.md
4. Follow setup steps in backend/README.md

### Running Locally
- Frontend: `cd frontend && npm run dev` (runs on http://localhost:3000)
- Backend: `cd backend && uvicorn main:app --reload` (runs on http://localhost:8000)

## Important Notes
- Both frontend and backend must be running for full functionality
- Frontend makes API calls to http://localhost:8000 (backend)
- Environment variables should be configured via .env.local files in each directory
