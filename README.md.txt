# 🌱 Plant Growth Analyzer

Aplikasi web interaktif untuk menganalisis data pertumbuhan tanaman dengan visualisasi yang komprehensif dan analisis statistik mendalam.

![Plant Growth Analyzer](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)

## ✨ Fitur Utama

### 📊 Analisis Data Komprehensif
- **Tren Pertumbuhan Individual**: Visualisasi pertumbuhan setiap plot dalam format line chart dan scatter plot
- **Perbandingan Multi-Parameter**: Analisis perbandingan tinggi, diameter, dan jumlah kanopi
- **Rata-rata Mingguan**: Tracking rata-rata pertumbuhan per minggu dengan berbagai tipe visualisasi

### 🔬 Analisis Statistik Lanjutan
- **Laju Pertumbuhan**: Perhitungan laju pertumbuhan keseluruhan dan perubahan mingguan
- **Uji ANOVA**: Analisis statistik untuk menguji perbedaan signifikan antar plot
- **Analisis Korelasi**: Heatmap korelasi antar parameter

### 📈 Visualisasi Interaktif
- **Pair Plot**: Scatter matrix untuk melihat hubungan antar parameter
- **Distribusi Data**: Histogram dengan KDE dan box plot
- **Charts yang Dapat Diunduh**: Export visualisasi dalam format PNG

### 🎛️ Fitur Tambahan
- **Filter Dinamis**: Filter berdasarkan plot dan rentang minggu
- **Export Data**: Download data terfilter dan summary report dalam CSV
- **Responsive Design**: Tampilan yang optimal di berbagai perangkat

## 🚀 Demo Live

Aplikasi dapat diakses di: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

## 📋 Format Data yang Dibutuhkan

Aplikasi ini membutuhkan file CSV dengan kolom berikut:

| Kolom | Deskripsi | Tipe Data |
|-------|-----------|-----------|
| `minggu` | Nomor minggu pengamatan | Integer |
| `tinggi_cm` | Tinggi tanaman dalam cm | Float |
| `diameter_cm` | Diameter batang dalam cm | Float |
| `jumlah_kanopi` | Jumlah kanopi/daun | Integer |
| `plot` | Identifikasi plot | String |

### Contoh Data:
```csv
minggu,tinggi_cm,diameter_cm,jumlah_kanopi,plot
1,10.5,2.1,3,A
1,12.2,2.3,4,B
2,15.3,2.8,5,A
2,16.7,3.1,6,B
```

## 🛠️ Instalasi Lokal

### Prerequisites
- Python 3.8 atau lebih tinggi
- pip (Python package manager)

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/plant-growth-analyzer.git
   cd plant-growth-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```

4. **Buka browser**
   ```
   http://localhost:8501
   ```

## 🔧 Dependencies

- `streamlit==1.28.0` - Framework aplikasi web
- `pandas==2.0.3` - Manipulasi data
- `plotly==5.15.0` - Visualisasi interaktif
- `numpy==1.24.3` - Komputasi numerik
- `scipy==1.11.1` - Analisis statistik
- `kaleido==0.2.1` - Export gambar

## 📱 Integrasi Mobile (Flutter)

Aplikasi ini dapat diintegrasikan dengan Flutter menggunakan:

1. **WebView**: Tampilkan aplikasi Streamlit dalam WebView
2. **REST API**: Buat API wrapper untuk komunikasi data
3. **Hybrid Approach**: Kombinasi native UI dengan web components

Lihat folder `flutter_integration/` untuk contoh implementasi.

## 🎯 Cara Penggunaan

### 1. Upload Data
- Klik tombol "Choose CSV file" di sidebar
- Pilih file CSV dengan format yang sesuai
- Data akan otomatis divalidasi dan diproses

### 2. Filter Data
- **Pilih Plot**: Multiselect untuk memilih plot yang ingin dianalisis
- **Rentang Minggu**: Slider untuk memilih periode analisis
- **Parameter**: Checkbox untuk memilih parameter yang ditampilkan

### 3. Analisis Data
- **Growth Trends**: Lihat tren pertumbuhan individual
- **Comparison**: Bandingkan pertumbuhan antar plot
- **Weekly Averages**: Analisis rata-rata mingguan
- **Correlation**: Analisis korelasi antar parameter
- **Data Table**: View dan export data terfilter

### 4. Export Results
- Download charts dalam format PNG
- Export data dalam format CSV
- Generate summary reports

## 🧪 Fitur Analisis Lanjutan

### Laju Pertumbuhan
```python
# Perhitungan laju pertumbuhan keseluruhan
growth_rate = (final_value - initial_value) / (final_week - initial_week)

# Perhitungan perubahan mingguan
weekly_change = (current_week_value - previous_week_value) / week_difference
```

### Uji ANOVA
```python
# Uji statistik untuk perbedaan antar plot
f_statistic, p_value = stats.f_oneway(*plot_data)
```

### Analisis Korelasi
```python
# Matrix korelasi untuk semua parameter numerik
correlation_matrix = df[['tinggi_cm', 'diameter_cm', 'jumlah_kanopi', 'minggu']].corr()
```

## 🎨 Kustomisasi

### Tema dan Styling
File `.streamlit/config.toml` dapat dimodifikasi untuk mengubah:
- Warna primer dan sekunder
- Background color
- Font family dan size

### Menambah Parameter Baru
1. Tambahkan kolom baru di CSV
2. Update validasi data di `load_and_process_data()`
3. Tambahkan parameter di fungsi analisis
4. Update visualisasi dan export functions

## 📊 Contoh Visualisasi

### Growth Trends
- Line charts untuk tracking pertumbuhan dari waktu ke waktu
- Scatter plots untuk melihat distribusi data
- Multiple series untuk perbandingan antar plot

### Statistical Analysis
- Box plots untuk melihat distribusi dan outliers
- Heatmaps untuk analisis korelasi
- Bar charts untuk perbandingan rata-rata

### Distribution Analysis
- Histograms dengan KDE curves
- Pair plots untuk hubungan antar variabel
- Summary statistics tables

## 🚀 Deployment

### Streamlit Community Cloud
1. Push code ke GitHub repository
2. Connect repository di [share.streamlit.io](https://share.streamlit.io)
3. Deploy dengan satu klik

### Alternative Deployment
- **Heroku**: Untuk deployment dengan custom domain
- **Docker**: Untuk containerized deployment
-