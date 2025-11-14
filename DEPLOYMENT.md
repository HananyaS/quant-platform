# 🚀 Deployment Guide

This guide covers deploying the Quantitative Trading Platform to various hosting services.

## 📋 Table of Contents

1. [Streamlit Cloud (Recommended)](#streamlit-cloud-recommended)
2. [Local Deployment](#local-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Heroku](#heroku)
5. [AWS/GCP/Azure](#cloud-platforms)

---

## 1. Streamlit Cloud (Recommended)

**Streamlit Community Cloud** is the easiest way to deploy Streamlit apps for free!

### Prerequisites
- GitHub account
- Git repository with your code
- Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Required Files (Already Included)

✅ **requirements.txt** - Python dependencies
```
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
yfinance>=0.2.28
streamlit>=1.28.0
matplotlib>=3.7.0
plotly>=5.14.0
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0
torch>=2.0.0
# ... (full list in requirements.txt)
```

✅ **packages.txt** - System-level packages (optional)
```
# Only needed for system packages like TA-Lib
```

✅ **.streamlit/config.toml** - Streamlit configuration
```toml
[theme]
primaryColor = "#1f77b4"
# ... (already configured)
```

### Deployment Steps

#### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Deploy quant trading platform"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/quant.git
git branch -M main
git push -u origin main
```

#### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/quant`
5. Set **Main file path**: `app.py`
6. Click **"Deploy"**

#### Step 3: Wait for Build (5-10 minutes)

Streamlit Cloud will:
- Install all dependencies from `requirements.txt`
- Set up the Python environment
- Launch your app

#### Step 4: Your App is Live! 🎉

You'll get a URL like: `https://YOUR_USERNAME-quant-app-xxxxx.streamlit.app`

### Troubleshooting Streamlit Cloud

**Problem: "ModuleNotFoundError: No module named 'quant_framework'"**
- ✅ **Fixed!** The path resolution in `app.py` handles this automatically.

**Problem: Build timeout or memory issues**
- PyTorch is large (~750MB). Consider using CPU-only version:
  ```txt
  # In requirements.txt, replace:
  torch>=2.0.0
  torchvision>=0.15.0
  
  # With:
  torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
  torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
  ```

**Problem: "Error: Out of memory"**
- Streamlit Cloud has 1GB RAM limit on free tier
- Reduce `sequence_length` in Deep Learning tab
- Use smaller datasets for testing

**Problem: Slow data loading**
- Yahoo Finance rate limits: Use cached data
- Consider pre-downloading data and including CSVs in repo

---

## 2. Local Deployment

### Quick Start (Already Installed)

```bash
# Windows
START_HERE.bat

# Mac/Linux
chmod +x START_HERE.sh
./START_HERE.sh
```

### Manual Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install package (optional but recommended)
pip install -e .

# 3. Run the app
streamlit run app.py
```

### Development Mode

```bash
# Run with auto-reload on file changes
streamlit run app.py --server.runOnSave true

# Run on specific port
streamlit run app.py --server.port 8080

# Run in headless mode (no browser)
streamlit run app.py --server.headless true
```

---

## 3. Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if needed for TA-Lib)
# RUN apt-get update && apt-get install -y build-essential wget

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t quant-trading-platform .

# Run container
docker run -p 8501:8501 quant-trading-platform

# Run with volume (for development)
docker run -p 8501:8501 -v $(pwd):/app quant-trading-platform
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./results:/app/results
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

Run with: `docker-compose up`

---

## 4. Heroku

### Prerequisites
- Heroku account
- Heroku CLI installed

### Create Required Files

**Procfile:**
```
web: sh setup.sh && streamlit run app.py
```

**setup.sh:**
```bash
#!/bin/bash

mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

### Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-quant-app

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

---

## 5. Cloud Platforms (AWS/GCP/Azure)

### AWS EC2

1. **Launch EC2 instance** (t2.medium or larger)
2. **SSH into instance**
3. **Install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip nginx -y
   ```
4. **Clone repository and install:**
   ```bash
   git clone YOUR_REPO_URL
   cd quant
   pip3 install -r requirements.txt
   ```
5. **Run with systemd:**
   ```ini
   # /etc/systemd/system/streamlit.service
   [Unit]
   Description=Streamlit Quant Platform
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/quant
   ExecStart=/usr/local/bin/streamlit run app.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
6. **Start service:**
   ```bash
   sudo systemctl enable streamlit
   sudo systemctl start streamlit
   ```

### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/quant-platform
gcloud run deploy quant-platform \
  --image gcr.io/PROJECT-ID/quant-platform \
  --platform managed \
  --region us-central1 \
  --memory 2Gi
```

### Azure App Service

```bash
az webapp up \
  --name quant-platform \
  --resource-group myResourceGroup \
  --runtime "PYTHON:3.10"
```

---

## 📊 Resource Requirements

### Minimum (Basic Strategies Only)
- **RAM**: 512MB
- **CPU**: 1 core
- **Storage**: 1GB

### Recommended (ML Models)
- **RAM**: 2GB
- **CPU**: 2 cores
- **Storage**: 5GB

### Optimal (Deep Learning)
- **RAM**: 4GB+
- **CPU**: 4 cores (or GPU)
- **Storage**: 10GB

---

## 🔒 Security Considerations

### For Production Deployment

1. **API Keys**: Never commit API keys to Git
   ```python
   # Use Streamlit secrets
   api_key = st.secrets["alpaca"]["api_key"]
   ```

2. **Rate Limiting**: Implement rate limiting for API calls
   ```python
   import streamlit as st
   from streamlit_autorefresh import st_autorefresh
   
   # Refresh every 60 seconds (avoid rate limits)
   st_autorefresh(interval=60000, key="datarefresh")
   ```

3. **HTTPS**: Always use HTTPS in production
   - Streamlit Cloud: Automatic
   - Other platforms: Use Let's Encrypt or cloud provider SSL

4. **Authentication**: Add password protection
   ```python
   import streamlit as st
   
   def check_password():
       if "password_correct" not in st.session_state:
           st.text_input("Password", type="password", key="password")
           if st.session_state.get("password") == st.secrets["password"]:
               st.session_state["password_correct"] = True
               st.rerun()
           elif st.session_state.get("password"):
               st.error("Incorrect password")
           return False
       return True
   
   if not check_password():
       st.stop()
   ```

---

## 📈 Performance Optimization

### Caching

```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(symbol):
    return yf.download(symbol, start=start, end=end)

@st.cache_resource
def load_model():
    return torch.load("model.pth")
```

### Data Management

- Pre-download common datasets (SPY, QQQ, etc.)
- Use parquet format instead of CSV (faster loading)
- Implement pagination for large result sets

### UI Responsiveness

- Use `st.spinner()` for long operations
- Implement progress bars with `st.progress()`
- Lazy load charts and data

---

## 🆘 Support

**Deployment Issues?**
- Check [Streamlit Community Forum](https://discuss.streamlit.io)
- See [Streamlit Docs](https://docs.streamlit.io)
- Open GitHub issue

**General Questions?**
- See [START_HERE.md](START_HERE.md)
- Check [README.md](README.md)
- Review example scripts in `examples/`

---

**Happy Deploying! 🚀📈**

