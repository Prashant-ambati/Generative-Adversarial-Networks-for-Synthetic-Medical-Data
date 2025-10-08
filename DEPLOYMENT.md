# Deployment Guide

**Built by Prashant Ambati**

This guide provides instructions for deploying the Synthetic Medical Data GAN project.

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/prashantambati/synthetic-medical-gan.git
cd synthetic-medical-gan
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the demo**
```bash
python run_demo.py --mode all
```

### Using Docker

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

2. **Access the dashboard**
- Open http://localhost:8501 for the Streamlit dashboard
- Open http://localhost:8888 for Jupyter notebooks (if enabled)

## 🌐 Cloud Deployment Options

### Streamlit Cloud

1. Fork the repository to your GitHub account
2. Connect to Streamlit Cloud
3. Deploy the `dashboard/app.py` file
4. Set Python version to 3.9

### Heroku

1. **Create Heroku app**
```bash
heroku create synthetic-medical-gan
```

2. **Add buildpacks**
```bash
heroku buildpacks:add heroku/python
```

3. **Deploy**
```bash
git push heroku main
```

### AWS EC2

1. **Launch EC2 instance** (t3.medium recommended)
2. **Install Docker**
```bash
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```

3. **Clone and deploy**
```bash
git clone https://github.com/prashantambati/synthetic-medical-gan.git
cd synthetic-medical-gan
docker-compose up -d
```

### Google Cloud Run

1. **Build and push to Container Registry**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/synthetic-medical-gan
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy --image gcr.io/PROJECT_ID/synthetic-medical-gan --platform managed
```

## 🔧 Configuration

### Environment Variables

- `STREAMLIT_SERVER_PORT`: Port for Streamlit (default: 8501)
- `PYTHONPATH`: Path to source code (default: /app/src)

### Model Configuration

Edit `src/train_gan.py` to modify:
- Network architecture
- Training parameters
- Data preprocessing

## 📊 Monitoring

### Health Checks

The application includes health check endpoints:
- `/health` - Basic health status
- `/metrics` - Training metrics

### Logging

Logs are written to:
- `logs/training.log` - Training progress
- `logs/evaluation.log` - Model evaluation
- `logs/app.log` - Application logs

## 🔒 Security Considerations

1. **Data Privacy**
   - Ensure input data is properly anonymized
   - Use secure data transfer protocols
   - Implement access controls

2. **Model Security**
   - Validate input parameters
   - Implement rate limiting
   - Monitor for adversarial inputs

3. **Infrastructure Security**
   - Use HTTPS in production
   - Implement proper authentication
   - Regular security updates

## 📈 Scaling

### Horizontal Scaling

Use load balancers with multiple instances:
```yaml
# docker-compose.yml
services:
  app:
    deploy:
      replicas: 3
```

### Vertical Scaling

Increase resource allocation:
- CPU: 2+ cores for training
- Memory: 8GB+ for large datasets
- GPU: Optional for faster training

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure PYTHONPATH is set correctly
   - Install all dependencies from requirements.txt

2. **Memory Issues**
   - Reduce batch size in training
   - Use data streaming for large datasets

3. **Performance Issues**
   - Enable GPU acceleration if available
   - Optimize data loading pipeline

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Support

For deployment issues:
1. Check the GitHub Issues page
2. Review the documentation
3. Contact: Prashant Ambati

---

**Built by Prashant Ambati** | [GitHub](https://github.com/prashantambati/synthetic-medical-gan)