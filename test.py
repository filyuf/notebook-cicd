# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import boto3
import joblib


# 2. Load CSV from S3
s3_path = 's3://order-567/Students.csv'  # 🔁 Ganti dengan path kamu
data = pd.read_csv(s3_path)

print("✅ Data berhasil dimuat dari S3")
print(data.head())

# 3. Pilih kolom fitur dan target
selected_columns = [
    'Avg_Daily_Usage_Hours',
    'Most_Used_Platform',
    'Affects_Academic_Performance',
    'Conflicts_Over_Social_Media',
    'Sleep_Hours_Per_Night',
    'Mental_Health_Score',
    'Addicted_Score'
]

df = data[selected_columns].copy()

# 4. Encode kolom kategorikal
categorical_columns = ['Most_Used_Platform', 'Affects_Academic_Performance', 'Conflicts_Over_Social_Media']
le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# 5. Pisahkan fitur dan label
X = df.drop('Addicted_Score', axis=1)
y = df['Addicted_Score']

# 6. Split data (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Buat dan latih model XGBoost
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 8. Prediksi dan evaluasi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Prediksi selesai\nMSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 9. Visualisasi hasil
try: 
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual', marker='o')
    plt.plot(y_pred, label='Predicted', marker='x')
    plt.title('Prediksi Addicted_Score vs Aktual')
    plt.xlabel('Data Index')
    plt.ylabel('Addicted Score')
    plt.legend()
    plt.grid(True)
    
    # Simpan gambar sebelum plt.show()
    image_filename = 'skor_kecanduan.png'
    plt.savefig(image_filename, bbox_inches='tight')  # bbox_inches supaya tidak terpotong
    print(f"✅ Grafik berhasil disimpan sebagai {image_filename}")
    
    # Tampilkan grafik setelah disimpan
    plt.show()

except Exception as e:
    print("❌ Error saat evaluasi/visualisasi:", e)


# Inisialisasi client boto3
s3 = boto3.client('s3')

# Ganti nama bucket & lokasi penyimpanan di S3
bucket_name = 'order-567'
s3_object_key = 'visuals/prediksi_addicted_score.png'

# Upload ke S3
s3.upload_file(image_filename, bucket_name, s3_object_key)

print(f"✅ Berhasil upload ke S3: s3://{bucket_name}/{s3_object_key}")


