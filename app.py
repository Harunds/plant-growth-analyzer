import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
import base64
from scipy import stats # Import untuk ANOVA

# Pastikan kaleido dan scipy terinstal untuk menyimpan gambar dan statistik
# pip install kaleido scipy

# Page config
st.set_page_config(
    page_title="🌱 Plant Growth Analyzer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #3498db;
    margin: 1rem 0;
}

.stAlert {
    margin-top: 1rem;
}

/* Style for the custom download link to look more like a button */
.stButton > button { /* Existing button style */
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
    border: none;
}

/* Custom download link style - makes it look like a button */
.stDownloadLink button {
    background-color: #008CBA; /* Blue */
    color: white;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
    border: none;
}
/* Ensure the markdown link also looks like a button */
a[download] {
    background-color: #008CBA; /* Blue */
    color: white !important; /* Important to override default link color */
    padding: 10px 20px;
    text-align: center;
    text-decoration: none !important; /* Remove underline */
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
    border: none;
}
a[download]:hover {
    background-color: #005f7a; /* Darker blue on hover */
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

class StreamlitPlantAnalyzer:
    def __init__(self):
        self.color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
            '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43',
            '#10AC84', '#EE5A24', '#0984E3', '#6C5CE7', '#FD79A8'
        ]
        self.plot_template = "plotly_white" # Default Plotly theme for cleaner look

    @st.cache_data
    def load_and_process_data(_self, uploaded_file):
        """Load and process uploaded CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            
            required_columns = ['minggu', 'tinggi_cm', 'diameter_cm', 'jumlah_kanopi', 'plot']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return None
            
            df = df.dropna(subset=required_columns)
            
            df['minggu'] = df['minggu'].astype(int)
            df['tinggi_cm'] = pd.to_numeric(df['tinggi_cm'], errors='coerce').fillna(0)
            df['diameter_cm'] = pd.to_numeric(df['diameter_cm'], errors='coerce').fillna(0)
            df['jumlah_kanopi'] = pd.to_numeric(df['jumlah_kanopi'], errors='coerce').fillna(0).astype(int)
            df['plot'] = df['plot'].astype(str).str.strip()
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def show_data_info(self, df):
        """Display data information"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Data Points", len(df))
        
        with col2:
            st.metric("Number of Plots", df['plot'].nunique())
        
        with col3:
            st.metric("Week Range", f"{df['minggu'].min()} - {df['minggu'].max()}")
        
        with col4:
            st.metric("Total Weeks", df['minggu'].max() - df['minggu'].min() + 1)
        
        with st.expander("📊 Summary Statistics"):
            summary_stats = df.groupby('plot').agg({
                'tinggi_cm': ['mean', 'max', 'min', 'std'],
                'diameter_cm': ['mean', 'max', 'min', 'std'],
                'jumlah_kanopi': ['mean', 'max', 'min', 'std']
            }).round(2)
            
            st.dataframe(summary_stats, use_container_width=True)
            
    def create_growth_trend_chart(self, df, parameter, title, y_label, chart_type='line', height=500):
        """Create individual growth trend chart with selectable type"""
        if chart_type == 'line':
            fig = go.Figure()
            plots = sorted(df['plot'].unique())
            for i, plot in enumerate(plots):
                plot_data = df[df['plot'] == plot].sort_values('minggu')
                color = self.color_palette[i % len(self.color_palette)]
                
                fig.add_trace(go.Scatter(
                    x=plot_data['minggu'],
                    y=plot_data[parameter],
                    mode='lines+markers', # Tetap ada marker di sini karena ini individual trend, bukan comparison
                    name=f'Plot {plot}',
                    line=dict(color=color, width=3),
                    marker=dict(size=6, color=color),
                    hovertemplate=
                        f'<b>Plot {plot}</b><br>' +
                        'Minggu: %{x}<br>' +
                        f'{y_label}: ' + '%{y}<br>' +
                        '<extra></extra>'
                ))
        elif chart_type == 'scatter':
            fig = px.scatter(df, x='minggu', y=parameter, color='plot',
                             title=title,
                             labels={'minggu': 'Minggu', parameter: y_label, 'plot': 'Plot'},
                             color_discrete_sequence=self.color_palette,
                             hover_name='plot',
                             template=self.plot_template)
            fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode='markers'))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#2c3e50')),
            xaxis_title=dict(text="Minggu", font=dict(size=14)),
            yaxis_title=dict(text=y_label, font=dict(size=14)),
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11)
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            template=self.plot_template,
            height=height
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        
        return fig
    
    def create_comparison_chart(self, df):
        """Create multi-parameter comparison chart"""
        parameters = ['tinggi_cm', 'diameter_cm', 'jumlah_kanopi']
        titles = ['Tinggi Tanaman (cm)', 'Diameter Batang (cm)', 'Jumlah Kanopi']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Tren Tinggi Tanaman', 'Tren Diameter Batang', 'Tren Jumlah Kanopi', 'Rata-rata Pertumbuhan per Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        plots = sorted(df['plot'].unique())
        
        # Individual parameter trends (Line chart)
        for param_idx, (param, title) in enumerate(zip(parameters, titles)):
            row = param_idx // 2 + 1
            col = param_idx % 2 + 1
            
            for plot_idx, plot in enumerate(plots):
                plot_data = df[df['plot'] == plot].sort_values('minggu')
                color = self.color_palette[plot_idx % len(self.color_palette)]
                
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['minggu'],
                        y=plot_data[param],
                        mode='lines', # DIGANTI: 'lines+markers' menjadi 'lines'
                        name=f'Plot {plot}',
                        line=dict(color=color, width=2),
                        # marker=dict(size=4), # MARKER DIHAPUS
                        showlegend=param_idx == 0,
                        legendgroup=f'plot_{plot}'
                    ),
                    row=row, col=col
                )
            fig.update_xaxes(title_text="Minggu", row=row, col=col, title_font=dict(size=12))
            fig.update_yaxes(title_text=f"{title.split(' ')[0]}", row=row, col=col, title_font=dict(size=12))
        
        # Summary bar chart
        summary = df.groupby('plot')[parameters].mean().reset_index()
        
        for param_idx, (param, title) in enumerate(zip(parameters, titles)):
            fig.add_trace(
                go.Bar(
                    x=[f'Plot {p}' for p in summary['plot']],
                    y=summary[param],
                    name=title,
                    marker_color=self.color_palette[param_idx],
                    showlegend=True,
                    legendgroup='summary_bars'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="🌱 Perbandingan Pertumbuhan Antar Plot",
            title_font=dict(size=22, color='#2c3e50'),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11)
            ),
            margin=dict(l=40, r=40, t=100, b=40),
            template=self.plot_template
        )
        
        fig.update_xaxes(title_text="Plot", row=2, col=2, title_font=dict(size=12))
        fig.update_yaxes(title_text="Rata-rata Nilai", row=2, col=2, title_font=dict(size=12))
        
        return fig
    
    def create_correlation_heatmap(self, df):
        """Create correlation heatmap"""
        correlation_data = df[['tinggi_cm', 'diameter_cm', 'jumlah_kanopi', 'minggu']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.columns,
            colorscale='Viridis',
            zmid=0,
            text=correlation_data.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🔥 Korelasi Antar Parameter",
            title_font=dict(size=20, color='#2c3e50'),
            width=600,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12),
            margin=dict(l=60, r=60, t=80, b=60),
            template=self.plot_template
        )
        
        return fig

    # --- FUNGSI BARU UNTUK ANALISIS PERTUMBUHAN KOMPARATIF ---
    def calculate_growth_rate_per_plot(self, df, parameter):
        """Calculate overall growth rate for each plot."""
        growth_rates = []
        for plot in sorted(df['plot'].unique()):
            plot_data = df[df['plot'] == plot].sort_values('minggu')
            if len(plot_data) > 1:
                initial_week = plot_data['minggu'].min()
                final_week = plot_data['minggu'].max()
                
                # Check if there's actual growth period
                if final_week > initial_week:
                    # Menggunakan rata-rata untuk nilai awal dan akhir jika ada beberapa data di minggu yang sama
                    initial_value = plot_data[plot_data['minggu'] == initial_week][parameter].mean()
                    final_value = plot_data[plot_data['minggu'] == final_week][parameter].mean()
                    
                    rate = (final_value - initial_value) / (final_week - initial_week)
                    growth_rates.append({'Plot': plot, f'Rata-rata Laju Pertumbuhan {parameter.replace("_", " ").title()}': rate})
        return pd.DataFrame(growth_rates)

    def calculate_weekly_growth_change(self, df, parameter):
        """Calculate week-over-week growth change for each plot using weekly averages."""
        # --- PERUBAHAN UTAMA DI SINI ---
        # Langkah 1: Hitung rata-rata parameter per plot per minggu
        weekly_avg_df = df.groupby(['plot', 'minggu'])[parameter].mean().reset_index()
        weekly_avg_df = weekly_avg_df.rename(columns={parameter: 'avg_param_value'})

        weekly_changes = []
        for plot in sorted(weekly_avg_df['plot'].unique()):
            # Langkah 2: Ambil data rata-rata untuk plot ini, urutkan berdasarkan minggu
            plot_data_avg = weekly_avg_df[weekly_avg_df['plot'] == plot].sort_values('minggu')

            # Pastikan ada setidaknya dua titik data untuk menghitung perubahan pertumbuhan
            if len(plot_data_avg) < 2:
                continue

            for i in range(1, len(plot_data_avg)):
                prev_week_data = plot_data_avg.iloc[i-1]
                curr_week_data = plot_data_avg.iloc[i]
                
                week_diff = curr_week_data['minggu'] - prev_week_data['minggu']
                
                # Hanya hitung jika ada selisih minggu positif
                if week_diff > 0:
                    change = (curr_week_data['avg_param_value'] - prev_week_data['avg_param_value']) / week_diff
                    weekly_changes.append({
                        'Plot': plot,
                        'Minggu': curr_week_data['minggu'],
                        f'Perubahan Mingguan {parameter.replace("_", " ").title()}': change
                    })
        return pd.DataFrame(weekly_changes)
    
    def plot_growth_rate(self, df_growth_rates, parameter_name, title, y_label, height=400):
        """Plot overall growth rates per plot."""
        if df_growth_rates.empty:
            return go.Figure() # Return empty figure if no data
        
        fig = px.bar(df_growth_rates, x='Plot', y=parameter_name,
                     title=title,
                     labels={'Plot': 'Plot', parameter_name: y_label},
                     color='Plot', color_discrete_sequence=self.color_palette,
                     template=self.plot_template)
        
        fig.update_layout(
            title_font=dict(size=18, color='#2c3e50'),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            plot_bgcolor='white', paper_bgcolor='white',
            height=height,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def plot_weekly_growth_change(self, df_weekly_change, parameter_name, title, y_label, height=500):
        """Plot weekly growth changes per plot."""
        if df_weekly_change.empty:
            return go.Figure() # Return empty figure if no data
            
        fig = px.line(df_weekly_change, x='Minggu', y=parameter_name, color='Plot',
                      title=title,
                      labels={'Minggu': 'Minggu', parameter_name: y_label, 'Plot': 'Plot'},
                      color_discrete_sequence=self.color_palette,
                      template=self.plot_template,
                      markers=False) # DIGANTI: markers=False
        
        fig.update_layout(
            title_font=dict(size=18, color='#2c3e50'),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            hovermode='x unified',
            plot_bgcolor='white', paper_bgcolor='white',
            height=height,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10)
            )
        )
        fig.update_xaxes(dtick=1) # Ensure integer ticks for weeks
        return fig

    def perform_anova(self, df, parameter):
        """Perform ANOVA test on a given parameter across different plots."""
        plot_data = [df[df['plot'] == plot][parameter].tolist() for plot in sorted(df['plot'].unique())]
        
        # Remove empty lists if a plot has no data for the parameter after filtering
        plot_data = [data for data in plot_data if data]

        if len(plot_data) < 2:
            return None, "Not enough groups (plots) for ANOVA. Need at least 2 plots with data."
        
        # ANOVA requires at least 2 data points per group, but scipy.stats.f_oneway handles single data points
        # The main issue is if all groups are empty or have insufficient variance.
        
        try:
            f_statistic, p_value = stats.f_oneway(*plot_data)
            return f_statistic, p_value
        except ValueError as e:
            return None, f"Error performing ANOVA: {e}. Ensure there is variance within groups and sufficient data points."

    # --- FUNGSI BARU: Pair Plot ---
    def create_pair_plot(self, df, title="Distribusi & Hubungan Antar Parameter"):
        """Create a pair plot (scatter matrix) for numerical parameters."""
        numerical_cols = ['tinggi_cm', 'diameter_cm', 'jumlah_kanopi', 'minggu']
        # Filter only available numerical columns
        plot_cols = [col for col in numerical_cols if col in df.columns]

        if len(plot_cols) < 2:
            st.warning("Tidak cukup kolom numerik untuk membuat Pair Plot (minimal 2).")
            return go.Figure()

        # Rename columns for better labels in the plot
        labels = {
            'tinggi_cm': 'Tinggi (cm)',
            'diameter_cm': 'Diameter (cm)',
            'jumlah_kanopi': 'Jumlah Kanopi',
            'minggu': 'Minggu'
        }
        
        fig = px.scatter_matrix(
            df,
            dimensions=plot_cols,
            color='plot', # Color points by plot
            color_discrete_sequence=self.color_palette,
            title=title,
            labels=labels # Apply custom labels
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False) # Hide diagonal and upper half for cleaner look
        fig.update_layout(
            title_font=dict(size=20, color='#2c3e50'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial", size=12),
            height=700, # Adjust height for better visibility
            margin=dict(l=40, r=40, t=80, b=40),
            template=self.plot_template
        )
        return fig
    
    # --- FUNGSI BARU: Distribusi Data (Histogram/KDE) ---
    def create_distribution_plot(self, df, parameter, title, x_label):
        """Create a histogram with KDE for a given parameter."""
        fig = px.histogram(df, x=parameter,
                           color='plot', # Color histograms by plot
                           color_discrete_sequence=self.color_palette,
                           marginal="box", # Add box plot on top for summary stats
                           histnorm='density', # Normalize to density to show KDE effectively
                           title=title,
                           labels={parameter: x_label, 'count': 'Kepadatan Data', 'plot': 'Plot'},
                           template=self.plot_template)
        
        fig.update_layout(
            title_font=dict(size=18, color='#2c3e50'),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(family="Arial", size=12),
            height=500,
            margin=dict(l=40, r=40, t=60, b=40),
            bargap=0.1 # Add gap between bars for better readability
        )
        return fig

    def download_button(self, fig, filename="chart.png", label="Download Chart"):
        try:
            current_height = fig.layout.height if fig.layout.height else 500
            current_width = fig.layout.width if fig.layout.width else 800

            if "Box Plot" in label:
                current_height = 650
            # Ensure width is reasonable for download, Streamlit's use_container_width is approx 700-1000px
            # For consistent width, let's set a standard.
            current_width = 900 # A good standard width for downloaded charts

            img_bytes = fig.to_image(format="png", scale=2, height=current_height, width=current_width)
            b64 = base64.b64encode(img_bytes).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{label}</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating download link: {e}. Make sure 'kaleido' is installed (`pip install kaleido`).")


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 Plant Growth Analyzer</h1>
        <p>Analisis Data Pertumbuhan Tanaman yang Fleksibel dan Interaktif</p>
    </div>
    """, unsafe_allow_html=True)
    
    analyzer = StreamlitPlantAnalyzer()
    
    # Sidebar
    st.sidebar.header("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="File harus berformat CSV dengan kolom: minggu, tinggi_cm, diameter_cm, jumlah_kanopi, plot"
    )
    
    if uploaded_file is not None:
        df = analyzer.load_and_process_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Data loaded successfully! {len(df)} rows loaded.")
            
            st.subheader("📊 Data Overview")
            analyzer.show_data_info(df)
            
            st.sidebar.header("🔧 Filters")
            
            available_plots = sorted(df['plot'].unique())
            selected_plots = st.sidebar.multiselect(
                "Select Plots",
                available_plots,
                default=available_plots,
                help="Choose which plots to include in analysis"
            )
            
            min_week, max_week = int(df['minggu'].min()), int(df['minggu'].max())
            week_range = st.sidebar.slider(
                "Week Range",
                min_value=min_week,
                max_value=max_week,
                value=(min_week, max_week),
                help="Select the range of weeks to analyze"
            )
            
            st.sidebar.header("📈 Parameters")
            show_tinggi = st.sidebar.checkbox("Tinggi Tanaman (cm)", value=True)
            show_diameter = st.sidebar.checkbox("Diameter Batang (cm)", value=True)
            show_kanopi = st.sidebar.checkbox("Jumlah Kanopi", value=True)
            
            filtered_df = df[
                (df['plot'].isin(selected_plots)) &
                (df['minggu'] >= week_range[0]) &
                (df['minggu'] <= week_range[1])
            ]
            
            if len(filtered_df) == 0:
                st.error("❌ No data available with selected filters!")
                return
            
            st.sidebar.success(f"📊 {len(filtered_df)} data points selected")
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([ # Tambah tab6
                "📈 Growth Trends", 
                "🔄 Comparison", 
                "📊 Weekly Averages", 
                "🔥 Correlation", 
                "📋 Data Table", 
                "🔍 Data Distribution & Relationships" # Nama tab baru
            ])
            
            with tab1:
                st.subheader("📈 Individual Growth Trends")
                chart_type = st.radio("Pilih Tipe Grafik:", ('Line', 'Scatter'), key='trend_chart_type')
                
                parameters_to_show = []
                if show_tinggi:
                    parameters_to_show.append(('tinggi_cm', 'Tren Pertumbuhan Tinggi Tanaman', 'Tinggi Tanaman (cm)'))
                if show_diameter:
                    parameters_to_show.append(('diameter_cm', 'Tren Pertumbuhan Diameter Batang', 'Diameter Batang (cm)'))
                if show_kanopi:
                    parameters_to_show.append(('jumlah_kanopi', 'Tren Pertumbuhan Jumlah Kanopi', 'Jumlah Kanopi'))
                
                for param, title, y_label in parameters_to_show:
                    fig_height = 500
                    if chart_type == 'scatter':
                        fig_height = 600
                    
                    fig = analyzer.create_growth_trend_chart(filtered_df, param, title, y_label, chart_type.lower(), height=fig_height)
                    st.plotly_chart(fig, use_container_width=True)
                    analyzer.download_button(fig, filename=f"{param}_{chart_type.lower()}_trend.png", label=f"Download {title} Chart ({chart_type})")
                    st.markdown("---")
            
            with tab2:
                st.subheader("🔄 Multi-Parameter Comparison & Advanced Analysis")
                comparison_fig = analyzer.create_comparison_chart(filtered_df)
                st.plotly_chart(comparison_fig, use_container_width=True)
                analyzer.download_button(comparison_fig, filename="comparison_chart.png", label="Download Comparison Chart")

                st.markdown("---")
                st.subheader("🚀 Analisis Laju Pertumbuhan")

                if len(filtered_df['minggu'].unique()) < 2:
                    st.info("⚠️ Data historis minimal 2 minggu diperlukan untuk menghitung laju pertumbuhan.")
                else:
                    parameters_to_analyze_growth = ['tinggi_cm', 'diameter_cm', 'jumlah_kanopi']
                    growth_analysis_param = st.selectbox("Pilih Parameter untuk Analisis Laju Pertumbuhan:", 
                                                         parameters_to_analyze_growth, 
                                                         format_func=lambda x: x.replace('_', ' ').title(), key='growth_param_select')

                    if growth_analysis_param:
                        # --- Overall Growth Rate ---
                        st.markdown("#### Laju Pertumbuhan Keseluruhan per Plot")
                        overall_growth_df = analyzer.calculate_growth_rate_per_plot(filtered_df, growth_analysis_param)
                        if not overall_growth_df.empty:
                            st.dataframe(overall_growth_df, use_container_width=True)
                            
                            overall_growth_title = f"Rata-rata Laju Pertumbuhan {growth_analysis_param.replace('_', ' ').title()} per Plot"
                            overall_growth_ylabel = f"Laju Pertumbuhan ({'cm/minggu' if 'cm' in growth_analysis_param else 'unit/minggu'})"
                            fig_overall_growth = analyzer.plot_growth_rate(overall_growth_df, overall_growth_df.columns[1], overall_growth_title, overall_growth_ylabel)
                            st.plotly_chart(fig_overall_growth, use_container_width=True)
                            analyzer.download_button(fig_overall_growth, filename=f"overall_growth_rate_{growth_analysis_param}.png", label=f"Download {overall_growth_title} Chart")
                        else:
                            st.info(f"Tidak ada data laju pertumbuhan keseluruhan untuk {growth_analysis_param}.")
                        
                        st.markdown("---")

                        # --- Weekly Growth Change ---
                        st.markdown("#### Perubahan Pertumbuhan Mingguan per Plot")
                        # Memanggil calculate_weekly_growth_change yang sudah diperbaiki
                        weekly_change_df = analyzer.calculate_weekly_growth_change(filtered_df, growth_analysis_param)
                        if not weekly_change_df.empty:
                            st.dataframe(weekly_change_df, use_container_width=True)

                            weekly_change_title = f"Perubahan Mingguan {growth_analysis_param.replace('_', ' ').title()} per Plot"
                            weekly_change_ylabel = f"Perubahan ({'cm/minggu' if 'cm' in growth_analysis_param else 'unit/minggu'})"
                            fig_weekly_change = analyzer.plot_weekly_growth_change(weekly_change_df, weekly_change_df.columns[2], weekly_change_title, weekly_change_ylabel)
                            st.plotly_chart(fig_weekly_change, use_container_width=True)
                            analyzer.download_button(fig_weekly_change, filename=f"weekly_change_{growth_analysis_param}.png", label=f"Download {weekly_change_title} Chart")
                        else:
                            st.info(f"Tidak ada data perubahan mingguan untuk {growth_analysis_param}. Pastikan ada data minimal 2 minggu berturut-turut.")
                
                st.markdown("---")
                st.subheader("🔬 Uji Statistik (ANOVA)")
                
                if filtered_df['plot'].nunique() < 2:
                    st.info("⚠️ ANOVA membutuhkan setidaknya 2 plot untuk perbandingan statistik.")
                else:
                    anova_param = st.selectbox("Pilih Parameter untuk Uji ANOVA:", 
                                               parameters_to_analyze_growth, 
                                               format_func=lambda x: x.replace('_', ' ').title(), key='anova_param_select')
                    
                    if anova_param:
                        # Untuk ANOVA, kita ingin membandingkan rata-rata keseluruhan (atau rata-rata akhir)
                        # Ini mengambil semua nilai parameter untuk setiap plot untuk ANOVA
                        f_statistic, p_value = analyzer.perform_anova(filtered_df, anova_param)
                        
                        if f_statistic is not None:
                            st.write(f"**Hasil Uji ANOVA untuk Rata-rata {anova_param.replace('_', ' ').title()} antar Plot:**")
                            st.write(f"F-statistik: `{f_statistic:.4f}`")
                            st.write(f"P-value: `{p_value:.4f}`")
                            
                            alpha = st.slider("Tingkat Signifikansi (Alpha):", 0.01, 0.10, 0.05, 0.01, help="Nilai p-value di bawah Alpha menunjukkan perbedaan signifikan.")
                            
                            if p_value < alpha:
                                st.success(f"**Kesimpulan:** Dengan tingkat signifikansi {alpha}, terdapat perbedaan yang **signifikan secara statistik** pada rata-rata {anova_param.replace('_', ' ').title()} antar plot.")
                            else:
                                st.info(f"**Kesimpulan:** Dengan tingkat signifikansi {alpha}, tidak terdapat perbedaan yang signifikan secara statistik pada rata-rata {anova_param.replace('_', ' ').title()} antar plot.")
                            st.markdown("*(Catatan: ANOVA menguji apakah ada setidaknya satu plot yang rata-ratanya berbeda dari yang lain. Untuk mengetahui plot mana yang berbeda, dibutuhkan uji post-hoc seperti Tukey HSD, yang lebih kompleks.)*")
                        else:
                            st.warning(p_value) # Tampilkan pesan error dari perform_anova
                
            
            with tab3:
                st.subheader("🗓️ Rata-rata Pertumbuhan per Minggu per Plot")
                if not filtered_df.empty:
                    weekly_avg = filtered_df.groupby(['minggu', 'plot']).agg({
                        'tinggi_cm': 'mean',
                        'diameter_cm': 'mean',
                        'jumlah_kanopi': 'mean'
                    }).round(2).reset_index()
                    
                    st.dataframe(weekly_avg, use_container_width=True)

                    st.write("Visualisasi Rata-rata Pertumbuhan per Minggu per Plot:")
                    
                    weekly_chart_type = st.radio("Pilih Tipe Grafik Rata-rata Mingguan:", ('Line', 'Bar', 'Box Plot'), key='weekly_chart_type')

                    # Tinggi
                    st.markdown("#### Rata-rata Tinggi Tanaman")
                    fig_weekly_tinggi_height = 500
                    if weekly_chart_type == 'Line':
                        fig_weekly_tinggi = px.line(weekly_avg, x='minggu', y='tinggi_cm', color='plot',
                                                  title='Rata-rata Tinggi Tanaman per Minggu',
                                                  labels={'tinggi_cm': 'Tinggi (cm)', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                  color_discrete_sequence=analyzer.color_palette,
                                                  template=analyzer.plot_template,
                                                  markers=False) # DIGANTI: markers=False
                        fig_weekly_tinggi.update_layout(hovermode='x unified')
                    elif weekly_chart_type == 'Bar':
                        fig_weekly_tinggi = px.bar(weekly_avg, x='minggu', y='tinggi_cm', color='plot', barmode='group',
                                                  title='Rata-rata Tinggi Tanaman per Minggu',
                                                  labels={'tinggi_cm': 'Tinggi (cm)', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                  color_discrete_sequence=analyzer.color_palette,
                                                  template=analyzer.plot_template)
                    elif weekly_chart_type == 'Box Plot':
                         fig_weekly_tinggi_height = 650
                         # Box plot pakai filtered_df (data individual), bukan weekly_avg (data rata-rata)
                         # karena box plot memang untuk melihat distribusi data individual
                         fig_weekly_tinggi = px.box(filtered_df, x='minggu', y='tinggi_cm', color='plot',
                                                  title='Distribusi Tinggi Tanaman per Minggu (Box Plot)',
                                                  labels={'tinggi_cm': 'Tinggi (cm)', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                  color_discrete_sequence=analyzer.color_palette,
                                                  template=analyzer.plot_template)
                    
                    fig_weekly_tinggi.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(family="Arial", size=12),
                                                    title_font=dict(size=18), xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14),
                                                    height=fig_weekly_tinggi_height)
                    st.plotly_chart(fig_weekly_tinggi, use_container_width=True)
                    analyzer.download_button(fig_weekly_tinggi, filename=f"weekly_avg_tinggi_{weekly_chart_type.lower()}.png", label=f"Download Rata-rata Tinggi Chart ({weekly_chart_type})")
                    st.markdown("---")

                    # Diameter
                    st.markdown("#### Rata-rata Diameter Batang")
                    fig_weekly_diameter_height = 500
                    if weekly_chart_type == 'Line':
                        fig_weekly_diameter = px.line(weekly_avg, x='minggu', y='diameter_cm', color='plot',
                                                   title='Rata-rata Diameter Batang per Minggu',
                                                   labels={'diameter_cm': 'Diameter (cm)', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                   color_discrete_sequence=analyzer.color_palette,
                                                   template=analyzer.plot_template,
                                                   markers=False) # DIGANTI: markers=False
                        fig_weekly_diameter.update_layout(hovermode='x unified')
                    elif weekly_chart_type == 'Bar':
                        fig_weekly_diameter = px.bar(weekly_avg, x='minggu', y='diameter_cm', color='plot', barmode='group',
                                                  title='Rata-rata Diameter Batang per Minggu',
                                                  labels={'diameter_cm': 'Diameter (cm)', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                  color_discrete_sequence=analyzer.color_palette,
                                                  template=analyzer.plot_template)
                    elif weekly_chart_type == 'Box Plot':
                         fig_weekly_diameter_height = 650
                         fig_weekly_diameter = px.box(filtered_df, x='minggu', y='diameter_cm', color='plot',
                                                  title='Distribusi Diameter Batang per Minggu (Box Plot)',
                                                  labels={'diameter_cm': 'Diameter (cm)', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                  color_discrete_sequence=analyzer.color_palette,
                                                  template=analyzer.plot_template)

                    fig_weekly_diameter.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(family="Arial", size=12),
                                                      title_font=dict(size=18), xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14),
                                                      height=fig_weekly_diameter_height)
                    st.plotly_chart(fig_weekly_diameter, use_container_width=True)
                    analyzer.download_button(fig_weekly_diameter, filename=f"weekly_avg_diameter_{weekly_chart_type.lower()}.png", label=f"Download Rata-rata Diameter Chart ({weekly_chart_type})")
                    st.markdown("---")

                    # Kanopi
                    st.markdown("#### Rata-rata Jumlah Kanopi")
                    fig_weekly_kanopi_height = 500
                    if weekly_chart_type == 'Line':
                        fig_weekly_kanopi = px.line(weekly_avg, x='minggu', y='jumlah_kanopi', color='plot',
                                                    title='Rata-rata Jumlah Kanopi per Minggu',
                                                    labels={'jumlah_kanopi': 'Jumlah Kanopi', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                    color_discrete_sequence=analyzer.color_palette,
                                                    template=analyzer.plot_template,
                                                    markers=False) # DIGANTI: markers=False
                        fig_weekly_kanopi.update_layout(hovermode='x unified')
                    elif weekly_chart_type == 'Bar':
                        fig_weekly_kanopi = px.bar(weekly_avg, x='minggu', y='jumlah_kanopi', color='plot', barmode='group',
                                                    title='Rata-rata Jumlah Kanopi per Minggu',
                                                    labels={'jumlah_kanopi': 'Jumlah Kanopi', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                    color_discrete_sequence=analyzer.color_palette,
                                                    template=analyzer.plot_template)
                    elif weekly_chart_type == 'Box Plot':
                         fig_weekly_kanopi_height = 650
                         fig_weekly_kanopi = px.box(filtered_df, x='minggu', y='jumlah_kanopi', color='plot',
                                                    title='Distribusi Jumlah Kanopi per Minggu (Box Plot)',
                                                    labels={'jumlah_kanopi': 'Jumlah Kanopi', 'minggu': 'Minggu', 'plot': 'Plot'},
                                                    color_discrete_sequence=analyzer.color_palette,
                                                    template=analyzer.plot_template)

                    fig_weekly_kanopi.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(family="Arial", size=12),
                                                    title_font=dict(size=18), xaxis_title_font=dict(size=14), yaxis_title_font=dict(size=14),
                                                    height=fig_weekly_kanopi_height)
                    st.plotly_chart(fig_weekly_kanopi, use_container_width=True)
                    analyzer.download_button(fig_weekly_kanopi, filename=f"weekly_avg_kanopi_{weekly_chart_type.lower()}.png", label=f"Download Rata-rata Kanopi Chart ({weekly_chart_type})")

                else:
                    st.info("No data available to calculate weekly averages for selected filters.")
            
            with tab4:
                st.subheader("🔥 Parameter Correlation Analysis")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    correlation_fig = analyzer.create_correlation_heatmap(filtered_df)
                    st.plotly_chart(correlation_fig, use_container_width=True)
                    analyzer.download_button(correlation_fig, filename="correlation_heatmap.png", label="Download Correlation Heatmap")

                with col2:
                    st.write("**Correlation Insights:**")
                    
                    corr_matrix = filtered_df[['tinggi_cm', 'diameter_cm', 'jumlah_kanopi', 'minggu']].corr()
                    
                    correlations = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            correlations.append({
                                'Variables': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                                'Correlation': round(corr_val, 3),
                                'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate' if abs(corr_val) > 0.4 else 'Weak'
                            })
                    
                    correlations_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
                    st.dataframe(correlations_df, use_container_width=True)
                    
                    st.write("**Key Insights:**")
                    if not correlations_df.empty:
                        strongest_corr = correlations_df.iloc[0]
                        st.write(f"• Strongest correlation: {strongest_corr['Variables']} ({strongest_corr['Correlation']})")
                        
                        time_corr = correlations_df[correlations_df['Variables'].str.contains('minggu')]
                        if not time_corr.empty:
                            time_strongest = time_corr.iloc[0]
                            st.write(f"• Time correlation: {time_strongest['Variables']} ({time_strongest['Correlation']})")
                    else:
                        st.write("• No correlations to display")
            
            with tab5:
                st.subheader("📋 Filtered Data Table")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"Showing {len(filtered_df)} rows")
                with col2:
                    csv_buffer = BytesIO()
                    filtered_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Filtered Data CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"plant_data_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                with col3:
                    summary_report = filtered_df.groupby('plot').agg({
                        'tinggi_cm': ['count', 'mean', 'std', 'min', 'max'],
                        'diameter_cm': ['mean', 'std', 'min', 'max'],
                        'jumlah_kanopi': ['mean', 'std', 'min', 'max']
                    }).round(2)
                    
                    summary_buffer = BytesIO()
                    summary_report.to_csv(summary_buffer)
                    st.download_button(
                        label="📊 Download Summary Report CSV",
                        data=summary_buffer.getvalue(),
                        file_name=f"plant_summary_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                page_size = st.select_slider("Rows per page", options=[10, 25, 50, 100], value=25)
                
                if len(filtered_df) > page_size:
                    total_pages = (len(filtered_df) - 1) // page_size + 1
                    page = st.selectbox("Page", range(1, total_pages + 1))
                    start_idx = (page - 1) * page_size
                    end_idx = min(start_idx + page_size, len(filtered_df))
                    display_df = filtered_df.iloc[start_idx:end_idx]
                else:
                    display_df = filtered_df
                
                st.dataframe(display_df, use_container_width=True)
                
                with st.expander("📊 Detailed Statistics"):
                    st.write("**Descriptive Statistics:**")
                    st.dataframe(filtered_df.describe(), use_container_width=True)

            # --- TAB BARU: Data Distribution & Relationships ---
            with tab6:
                st.subheader("🔍 Distribusi Data & Hubungan Antar Parameter")
                st.markdown("Visualisasi ini membantu Anda memahami sebaran nilai dari setiap parameter dan bagaimana parameter-parameter tersebut saling berhubungan.")
                
                st.markdown("#### Pair Plot (Scatter Matrix)")
                st.info("Pair Plot menampilkan scatter plot untuk setiap pasangan parameter numerik, dengan histogram/KDE pada diagonal. Ini membantu melihat korelasi visual dan distribusi data.")
                pair_plot_fig = analyzer.create_pair_plot(filtered_df)
                st.plotly_chart(pair_plot_fig, use_container_width=True)
                analyzer.download_button(pair_plot_fig, filename="pair_plot.png", label="Download Pair Plot")
                
                st.markdown("---")
                st.markdown("#### Distribusi Parameter Individu")
                st.info("Histogram dengan KDE menunjukkan sebaran nilai untuk parameter yang dipilih, dengan kotak (box plot) di atasnya untuk ringkasan statistik (median, kuartil, outlier).")
                
                distribution_param = st.selectbox(
                    "Pilih Parameter untuk Distribusi:",
                    ['tinggi_cm', 'diameter_cm', 'jumlah_kanopi', 'minggu'],
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key='distribution_param_select'
                )

                if distribution_param:
                    param_labels = {
                        'tinggi_cm': 'Tinggi Tanaman (cm)',
                        'diameter_cm': 'Diameter Batang (cm)',
                        'jumlah_kanopi': 'Jumlah Kanopi',
                        'minggu': 'Minggu'
                    }
                    dist_fig = analyzer.create_distribution_plot(
                        filtered_df,
                        distribution_param,
                        f'Distribusi {param_labels[distribution_param]}',
                        param_labels[distribution_param]
                    )
                    st.plotly_chart(dist_fig, use_container_width=True)
                    analyzer.download_button(dist_fig, filename=f"distribution_{distribution_param}.png", label=f"Download Distribusi {param_labels[distribution_param]}")


    else:
        st.info("👆 Please upload a CSV file to get started!")
        
        with st.expander("📋 Expected CSV Format"):
            example_data = pd.DataFrame({
                'minggu': [1, 1, 1, 2, 2, 2],
                'tinggi_cm': [10.5, 12.2, 11.8, 15.3, 16.7, 14.9],
                'diameter_cm': [2.1, 2.3, 2.0, 2.8, 3.1, 2.6],
                'jumlah_kanopi': [3, 4, 3, 5, 6, 4],
                'plot': ['A', 'B', 'C', 'A', 'B', 'C']
            })
            
            st.write("Your CSV file should have the following columns:")
            st.dataframe(example_data, use_container_width=True)
            
            st.write("**Column Descriptions:**")
            st.write("- `minggu`: Week number (integer)")
            st.write("- `tinggi_cm`: Plant height in centimeters (decimal)")
            st.write("- `diameter_cm`: Stem diameter in centimeters (decimal)")
            st.write("- `jumlah_kanopi`: Number of canopy/leaves (integer)")
            st.write("- `plot`: Plot identifier (text/number)")


if __name__ == "__main__":
    main()
