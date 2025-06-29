#!/bin/bash
# on-start.sh

sudo -u ec2-user aws s3 cp s3://order-567/notebooks/train_model.ipynb /home/ec2-user/SageMaker/train_model.ipynb
