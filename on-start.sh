#!/bin/bash

# Set AWS region
export AWS_DEFAULT_REGION=us-east-1

# Copy notebook dari S3 ke folder SageMaker (yang diakses Jupyter)
sudo -u ec2-user aws s3 cp s3://order-567/notebooks/train_model.ipynb /home/ec2-user/SageMaker/train_model.ipynb

# Debug log (opsional)
echo "✅ Notebook berhasil di-copy ke /home/ec2-user/SageMaker/" >> /home/ec2-user/SageMaker/lifecycle_log.txt
ls -lah /home/ec2-user/SageMaker/ >> /home/ec2-user/SageMaker/lifecycle_log.txt
