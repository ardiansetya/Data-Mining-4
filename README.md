### Import Library dan Load Data**
```python
file_path = 'produksiPadi.xlsx'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error 

df = pd.read_excel(file_path)
df.head()
```

**Penjelasan:**
1. **`file_path`**: Menyimpan nama file Excel tempat data disimpan (`produksiPadi.xlsx`).
2. **`import` statements**: Mengimpor pustaka yang dibutuhkan:
   - `pandas` untuk manipulasi data tabular.
   - `numpy` untuk operasi numerik.
   - Modul dari `sklearn`:
     - `train_test_split` untuk membagi data menjadi data latih dan uji.
     - `LinearRegression` untuk membuat model regresi linier.
     - `mean_squared_error` dan `mean_absolute_percentage_error` untuk mengukur kinerja model.
3. **`pd.read_excel(file_path)`**: Membaca file Excel dan menyimpannya dalam DataFrame `df`.
4. **`df.head()`**: Menampilkan 5 baris pertama data untuk memeriksa apakah data sudah terbaca dengan benar.

---

### Membuat Lagged Data**
```python
df['Lagged_Produksi'] = df['Produksi Padi(Ton)'].shift(1)
df = df.dropna()  
```

**Penjelasan:**
1. **`shift(1)`**:
   - Membuat kolom baru (`Lagged_Produksi`) yang berisi nilai dari tahun sebelumnya di kolom `Produksi Padi(Ton)`.
2. **`dropna()`**:
   - Menghapus baris pertama atau baris lain yang memiliki nilai kosong (NaN) karena proses `shift(1)`.

**Tujuan**:
- Membuat data historis (lagged data) yang diperlukan untuk model prediksi.

---

### Menyiapkan Variabel untuk Model**
```python
X = df[['Lagged_Produksi']]   
y = df['Produksi Padi(Ton)'] 
```

**Penjelasan:**
1. **`X`**: Variabel independen (input) untuk model, yaitu data dari kolom `Lagged_Produksi`.
2. **`y`**: Variabel dependen (target/output) untuk model, yaitu data dari kolom `Produksi Padi(Ton)`.

**Tujuan**:
- Menentukan input dan output untuk proses pelatihan model.

---

### Membagi Data Latih dan Uji**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state=42, shuffle=False)
```

**Penjelasan:**
1. **`train_test_split`**:
   - Membagi data menjadi dua bagian: 
     - **70%** untuk data latih (`X_train`, `y_train`).
     - **30%** untuk data uji (`X_test`, `y_test`).
   - **`shuffle=False`**: Tidak mengacak data agar urutan waktu tetap terjaga (karena ini data berbasis waktu).

**Tujuan**:
- Memisahkan data latih dan uji untuk mengevaluasi performa model.

---

### **Cell 5: Melatih Model**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

**Penjelasan:**
1. **`LinearRegression()`**: Membuat model regresi linier.
2. **`fit(X_train, y_train)`**:
   - Melatih model menggunakan data latih (`X_train` dan `y_train`).

**Tujuan**:
- Model belajar hubungan antara data historis (`Lagged_Produksi`) dan data produksi aktual (`Produksi Padi(Ton)`).

---

### Evaluasi Model**
```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Print evaluation metrics
print("Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
```

**Penjelasan:**
1. **`predict(X_test)`**: Menggunakan model untuk memprediksi nilai produksi (`y_pred`) berdasarkan data uji (`X_test`).
2. **Metode evaluasi**:
   - **MSE (Mean Squared Error)**: Rata-rata kuadrat dari selisih antara nilai aktual dan prediksi.
   - **RMSE (Root Mean Squared Error)**: Akar kuadrat dari MSE, digunakan untuk memahami kesalahan dalam satuan asli data.
   - **MAPE (Mean Absolute Percentage Error)**: Rata-rata kesalahan prediksi dalam persentase.
3. **`print`**: Menampilkan nilai metrik evaluasi untuk menilai performa model.

---

### Prediksi untuk Tahun 2023 dan 2024**
```python
latest_production = df['Produksi Padi(Ton)'].iloc[-1]  # Get the latest production value
prod_2023 = model.predict([[latest_production]])[0]  # Predict for 2023
prod_2024 = model.predict([[prod_2023]])[0]         # Predict for 2024

print("\nPredictions:")
print(f"Production for 2023: {prod_2023:.2f}")
print(f"Production for 2024: {prod_2024:.2f}")
```

**Penjelasan:**
1. **`latest_production`**: Mengambil nilai produksi terakhir di dataset.
2. **`prod_2023`**: Memanfaatkan model untuk memprediksi produksi tahun 2023 berdasarkan nilai terakhir (`latest_production`).
3. **`prod_2024`**: Menggunakan hasil prediksi produksi 2023 untuk memprediksi produksi tahun 2024.
4. **`print`**: Menampilkan hasil prediksi untuk tahun 2023 dan 2024.

**Tujuan**:
- Menggunakan model untuk melakukan prediksi jangka pendek.
