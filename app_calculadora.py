import streamlit as st
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Configuración de página de Streamlit
st.set_page_config(page_title="Calculadora de Intervalos de Confianza", layout="wide")

# ──────────────────────────────────────────────
# FUNCIONES DE CÁLCULO
# ──────────────────────────────────────────────

def ic_media_varianza_conocida(x_barra, sigma, n, confianza):
    alpha = 1 - confianza
    z = stats.norm.ppf(1 - alpha / 2)
    ee = sigma / np.sqrt(n)
    me = z * ee
    inf, sup = x_barra - me, x_barra + me
    return (inf, sup), z, ee, me

def ic_media_varianza_desconocida(x_barra, s, n, confianza):
    alpha = 1 - confianza
    gl = n - 1
    t = stats.t.ppf(1 - alpha / 2, df=gl)
    ee = s / np.sqrt(n)
    me = t * ee
    inf, sup = x_barra - me, x_barra + me
    return (inf, sup), gl, t, ee, me

def ic_varianza(s2, n, confianza):
    alpha = 1 - confianza
    gl = n - 1
    chi2_inf = stats.chi2.ppf(alpha / 2, df=gl)
    chi2_sup = stats.chi2.ppf(1 - alpha / 2, df=gl)
    var_inf = (gl * s2) / chi2_sup
    var_sup = (gl * s2) / chi2_inf
    return (var_inf, var_sup), gl, chi2_inf, chi2_sup

def ic_proporcion(p_hat, n, confianza):
    alpha = 1 - confianza
    z = stats.norm.ppf(1 - alpha / 2)
    ee = np.sqrt(p_hat * (1 - p_hat) / n)
    me = z * ee
    inf = max(0.0, p_hat - me)
    sup = min(1.0, p_hat + me)
    return (inf, sup), z, ee, me

# ──────────────────────────────────────────────
# INTERFAZ DE USUARIO (STREAMLIT)
# ──────────────────────────────────────────────

st.title("📊 Calculadora de Intervalos de Confianza")
st.write("Selecciona el tipo de intervalo que deseas calcular en la barra lateral e ingresa los datos correspondientes.")

# Barra lateral para selección del tipo de IC
tipo_ic = st.sidebar.selectbox(
    "Tipo de Intervalo de Confianza:",
    [
        "Media — Varianza poblacional CONOCIDA",
        "Media — Varianza poblacional DESCONOCIDA",
        "Varianza",
        "Proporción"
    ]
)

# Confianza común para todos
confianza_sel = st.sidebar.selectbox("Nivel de confianza:", ["90%", "95%", "99%", "Otro"])
if confianza_sel == "Otro":
    confianza = st.sidebar.number_input(
    "Valor de confianza personalizado:", 
    min_value=0.01, 
    max_value=0.99, 
    value=0.955, 
    step=0.001,
    format="%.3f"
)
else:
    confianza = float(confianza_sel.replace("%", "")) / 100.0

# Estilo CSS para mejorar presentación de fórmulas y resultados
st.markdown("""
<style>
.result-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid crimson;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# 1. MEDIA VARIANZA CONOCIDA
if tipo_ic == "Media — Varianza poblacional CONOCIDA":
    st.header("IC para la Media (μ) — Varianza Poblacional Conocida")
    
    col1, col2 = st.columns(2)
    with col1:
        x_barra = st.number_input("Media muestral (x̄):", value=0.0)
        sigma = st.number_input("Desviación estándar poblacional (σ):", value=1.0, min_value=0.0001)
        n = st.number_input("Tamaño de muestra (n):", value=30, min_value=2)
    
    if st.button("Calcular Intervalo"):
        intervalo, Z, ee, me = ic_media_varianza_conocida(x_barra, sigma, n, confianza)
        inf, sup = intervalo
        
        st.markdown(f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f} ≤ μ ≤ {sup:.4f}] = {confianza*100:.2f}%</div>', unsafe_allow_html=True)
        
        st.write(f"**Valor crítico Z:** {Z:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")

        #fig_col1 = st.columns(1)

        #with fig_col1:
        # Gráfico de Intervalo
        fig1, ax1 = plt.subplots(figsize=(6, 1.02))
        ax1.hlines(y=1, xmin=inf, xmax=sup, colors='crimson', linewidth=4, label=f'IC {confianza*100:.2f}%')
        ax1.plot([inf, inf], [0.85, 1.15], color='crimson', lw=2.5)
        ax1.plot([sup, sup], [0.85, 1.15], color='crimson', lw=2.5)
        #ax1.plot(x_barra, 1, 'o', color='navy', markersize=10, label=f'Media x̄ = {x_barra}')
        ax1.text(inf, 1.25, f'Lim Inf: {inf:.4f}', horizontalalignment='center', fontweight='bold', color='crimson')
        ax1.text(sup, 1.25, f'Lim Sup: {sup:.4f}', horizontalalignment='center', fontweight='bold', color='crimson')
        ax1.set_ylim(0.4, 1.6)
        ax1.set_yticks([])
        ax1.tick_params(axis='both', labelsize=7)
        ax1.set_xlabel('Escala de la Media (μ)', fontsize=7)
        ax1.set_title(f'Gráfico del Intervalo de Confianza', fontsize=9)
        ax1.legend(loc='lower right', fontsize=7)
        rango = sup - inf if (sup - inf) > 0 else 1.0
        ax1.set_xlim(inf - rango*0.2, sup + rango*0.2)
        ax1.grid(True, axis='x', alpha=0.3)
        st.pyplot(fig1)

        
        # Fórmulas en LaTeX
        st.subheader("Modelos Teóricos y Fórmulas")
        st.latex(r"IC \left[ \bar{X} - Z_{1-{\alpha \over 2}} * {\sigma \over {\sqrt n}} \leq \mu \leq  \bar{X} + Z_{1-{\alpha \over 2}} * {\sigma \over {\sqrt n}} \right]=1-\alpha")
        st.latex(r"X \sim N(\mu ; \sigma) \quad | \quad \bar{X} \sim N\left(\mu ; {\sigma \over {\sqrt n}}\right) \quad | \quad Z_{obs} = {\bar{X} - \mu \over {\sigma \over {\sqrt n}}} \sim N(0;1)")
        
        # Gráficos
        #fig_col2 = st.columns(1)
        
        #with fig_col2:
        # Gráfico Densidad Normal
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        x_vals = np.linspace(-4, 4, 500)
        y_vals = stats.norm.pdf(x_vals, 0, 1)
        ax2.plot(x_vals, y_vals, label='N(0,1)', color='darkorange', lw=2)
        x_fill = np.linspace(-Z, Z, 200)
        ax2.fill_between(x_fill, stats.norm.pdf(x_fill, 0, 1), color='orange', alpha=0.4, label='Confianza')
        ax2.axvline(-Z, color='coral', linestyle='--', linewidth=1.5, label=f'-Z = {-Z:.4f}')
        ax2.axvline(Z, color='coral', linestyle='--', linewidth=1.5, label=f'Z = {Z:.4f}')
        ax2.set_ylim(0,max(y_vals)+0.05)
        ax2.set_title("Región de Confianza")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

# 2. MEDIA VARIANZA DESCONOCIDA
elif tipo_ic == "Media — Varianza poblacional DESCONOCIDA":
    st.header("IC para la Media (μ) — Varianza Poblacional Desconocida")
    
    col1, col2 = st.columns(2)
    with col1:
        x_barra = st.number_input("Media muestral (x̄):", value=0.0)
        s = st.number_input("Desviación estándar muestral (s):", value=1.0, min_value=0.0001)
        n = st.number_input("Tamaño de muestra (n):", value=30, min_value=2)
        
    if st.button("Calcular Intervalo"):
        intervalo, gl, t, ee, me = ic_media_varianza_desconocida(x_barra, s, n, confianza)
        inf, sup = intervalo
        
        st.markdown(f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f} ≤ μ ≤ {sup:.4f}] = {confianza*100:.2f}%</div>', unsafe_allow_html=True)
        
        st.write(f"**Grados de libertad (n-1):** {gl}")
        st.write(f"**Valor crítico t:** {t:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")

        fig1, ax1 = plt.subplots(figsize=(6, 1.02))
        ax1.hlines(y=1, xmin=inf, xmax=sup, colors='crimson', linewidth=4, label=f'IC {confianza*100:.2f}%')
        ax1.plot([inf, inf], [0.85, 1.15], color='crimson', lw=2.5)
        ax1.plot([sup, sup], [0.85, 1.15], color='crimson', lw=2.5)
        #ax1.plot(x_barra, 1, 'o', color='navy', markersize=10, label=f'Media x̄ = {x_barra}')
        ax1.text(inf, 1.25, f'Lim Inf: {inf:.4f}', horizontalalignment='center', fontweight='bold', color='crimson')
        ax1.text(sup, 1.25, f'Lim Sup: {sup:.4f}', horizontalalignment='center', fontweight='bold', color='crimson')
        ax1.set_ylim(0.4, 1.6)
        ax1.set_yticks([])
        ax1.tick_params(axis='both', labelsize=7)
        ax1.set_xlabel('Escala de la Media (μ)', fontsize=7)
        ax1.set_title('Gráfico del Intervalo de Confianza', fontsize=9)
        ax1.legend(loc='lower right', fontsize=7)
        rango = sup - inf if (sup - inf) > 0 else 1.0
        ax1.set_xlim(inf - rango*0.2, sup + rango*0.2)
        ax1.grid(True, axis='x', alpha=0.3)
        st.pyplot(fig1)
        
        st.subheader("Modelos Teóricos y Fórmulas")
        st.latex(r"IC \left[ \bar{X} - t_{n-1; {1- {\alpha \over 2}}} * {s \over {\sqrt n}} \leq \mu \leq  \bar{X} + t_{n-1; {1- {\alpha \over 2}}} * {s \over {\sqrt n}} \right] =1-\alpha")
        st.latex(r"X \sim N(\mu ; \sigma) \quad | \quad \bar{X} \sim N\left(\mu ; {S \over {\sqrt n}}\right) \quad | \quad t_{obs}= {\bar{X} - \mu \over {S \over {\sqrt n}}} \sim t_{n-1}")
        
        #fig_col1, fig_col2 = st.columns(2)
            
        #with fig_col2:
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        x_vals = np.linspace(-4, 4, 500)
        y_vals = stats.t.pdf(x_vals, df=gl)
        ax2.plot(x_vals, y_vals, label=f't-Student (gl={gl})', color='darkorange', lw=2)
        x_fill = np.linspace(-t, t, 200)
        ax2.fill_between(x_fill, stats.t.pdf(x_fill, df=gl), color='orange', alpha=0.4, label='Confianza')
        ax2.axvline(-t, color='coral', linestyle='--', linewidth=1.5, label=f'-t = {-t:.4f}')
        ax2.axvline(t, color='coral', linestyle='--', linewidth=1.5, label=f't = {t:.4f}')
        ax2.set_title("Región de Confianza (t)")
        ax2.set_ylim(0,max(y_vals)+0.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

# 3. VARIANZA
elif tipo_ic == "Varianza":
    st.header("IC para la Varianza (σ²)")
    
    col1, col2 = st.columns(2)
    with col1:
        s2 = st.number_input("Varianza muestral (s²):", value=1.0, min_value=0.0001)
        n = st.number_input("Tamaño de muestra (n):", value=30, min_value=2)
        
    if st.button("Calcular Intervalo"):
        intervalo, gl, chi2_inf, chi2_sup = ic_varianza(s2, n, confianza)
        var_inf, var_sup = intervalo
        
        st.markdown(f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{var_inf:.4f} ≤ σ² ≤ {var_sup:.4f}] = {confianza*100:.2f}%</div>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(6, 1.02))
        ax1.hlines(y=1, xmin=var_inf, xmax=var_sup, colors='crimson', linewidth=4, label=f'IC {confianza*100:.2f}%')
        ax1.plot([var_inf, var_inf], [0.85, 1.15], color='crimson', lw=2.5)
        ax1.plot([var_sup, var_sup], [0.85, 1.15], color='crimson', lw=2.5)
        #ax1.plot(s2, 1, 'o', color='navy', markersize=10, label=f'Varianza s² = {s2}')
        ax1.text(var_inf, 1.25, f'Lim Inf: {var_inf:.4f}', horizontalalignment='center', fontweight='bold', color='crimson')
        ax1.text(var_sup, 1.25, f'Lim Sup: {var_sup:.4f}', horizontalalignment='center', fontweight='bold', color='crimson')
        ax1.set_ylim(0.4, 1.6)
        ax1.set_yticks([])
        ax1.tick_params(axis='both', labelsize=7)
        ax1.set_xlabel('Escala de la Varianza (σ²)', fontsize=7)
        ax1.set_title('Gráfico del Intervalo de Confianza', fontsize=9)
        ax1.legend(loc='lower right', fontsize=7)
        rango = var_sup - var_inf
        ax1.set_xlim(var_inf - rango*0.2, var_sup + rango*0.2)
        ax1.grid(True, axis='x', alpha=0.3)
        st.pyplot(fig1)
        
        st.subheader("Modelos Teóricos y Fórmulas")
        st.latex(r"IC \left[\frac{(n-1)s^2}{\chi^2_{1-{\alpha \over 2}}} \leq \sigma^2 \leq \frac{(n-1)s^2}{\chi^2_{\alpha \over 2}}\right]= 1- \alpha")
        st.latex(r"X \sim N(\mu ; \sigma) \quad | \quad S^2 \sim \chi^2_n \quad | \quad \chi^2_{obs} = {(n-1)S^2 \over \sigma^2} \sim \chi^2_{n-1}")
        
        #fig_col1, fig_col2 = st.columns(2)
        #with fig_col1: 
        #with fig_col2:
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        x_vals = np.linspace(stats.chi2.ppf(0.0001, df=gl), stats.chi2.ppf(0.9999, df=gl), 500)
        y_vals = stats.chi2.pdf(x_vals, df=gl)
        ax2.plot(x_vals, y_vals, label=f'Chi-Cuadrado (gl={gl})', color='darkorange', lw=2)
        x_fill = np.linspace(chi2_inf, chi2_sup, 200)
        ax2.fill_between(x_fill, stats.chi2.pdf(x_fill, df=gl), color='orange', alpha=0.4, label='Confianza')
        ax2.axvline(chi2_inf, color='coral', linestyle='--', linewidth=1.5, label=f'χ²_inf = {chi2_inf:.4f}')
        ax2.axvline(chi2_sup, color='coral', linestyle='--', linewidth=1.5, label=f'χ²_sup = {chi2_sup:.4f}')
        ax2.set_ylim(0,max(y_vals)+0.05)
        ax2.set_title("Región de Confianza (χ²)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

# 4. PROPORCION
elif tipo_ic == "Proporción":
    st.header("IC para la Proporción (p)")
    
    col1, col2 = st.columns(2)
    with col1:
        p_hat = st.number_input("Proporción muestral p̂ (entre 0 y 1):", value=0.5, min_value=0.0, max_value=1.0)
        n = st.number_input("Tamaño de muestra (n):", value=30, min_value=2)
        
    if st.button("Calcular Intervalo"):
        intervalo, z, ee, me = ic_proporcion(p_hat, n, confianza)
        inf, sup = intervalo
        
        st.markdown(f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f} ≤ p ≤ {sup:.4f}] = {confianza*100:.2f}%</div>', unsafe_allow_html=True)
        
        st.write(f"**Valor crítico Z:** {z:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")

        fig1, ax1 = plt.subplots(figsize=(6, 1.02))
        ax1.hlines(y=1, xmin=inf, xmax=sup, colors='crimson', linewidth=4, label=f'IC {confianza*100:.2f}%')
        ax1.plot([inf, inf], [0.85, 1.15], color='crimson', lw=2.5)
        ax1.plot([sup, sup], [0.85, 1.15], color='crimson', lw=2.5)
        #ax1.plot(p_hat, 1, 'o', color='navy', markersize=10, label=f'Proporción p̂ = {p_hat}')
        ax1.text(inf, 1.25, f'Lim Inf: {inf:.4f}', horizontalalignment='center', fontweight='bold', color='crimson')
        ax1.text(sup, 1.25, f'Lim Sup: {sup:.4f}', horizontalalignment='center', fontweight='bold', color='crimson')
        ax1.set_ylim(0.4, 1.6)
        ax1.set_yticks([])
        ax1.tick_params(axis='both', labelsize=7)
        ax1.set_xlabel('Escala de la Proporción (p)', fontsize=7)
        ax1.set_title('Gráfico del Intervalo de Confianza', fontsize=9)
        ax1.legend(loc='lower right', fontsize=7)
        rango = sup - inf if (sup - inf) > 0 else 1.0
        ax1.set_xlim(inf - rango*0.2, sup + rango*0.2)
        ax1.grid(True, axis='x', alpha=0.3)
        st.pyplot(fig1)
        
        st.subheader("Modelos Teóricos y Fórmulas")
        st.latex(r"IC \left[ \hat{p} - Z_{1-{\alpha \over 2}} * \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \leq p \leq \hat{p} + Z_{1-{\alpha \over 2}} *\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \right]=1-\alpha")
        st.latex(r"X \sim Bi(n ; p) \quad | \quad \hat p \sim Bi(n ; p) \quad | \quad Z_{obs}={\hat p - p \over {\sqrt {p(1-p) \over n}}} \sim N(0;1)")
        
        #fig_col1, fig_col2 = st.columns(2)
        #with fig_col1: 
        #with fig_col2:
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        x_vals = np.linspace(-4, 4, 500)
        y_vals = stats.norm.pdf(x_vals, 0, 1)
        ax2.plot(x_vals, y_vals, label='N(0,1)', color='darkorange', lw=2)
        x_fill = np.linspace(-z, z, 200)
        ax2.fill_between(x_fill, stats.norm.pdf(x_fill, 0, 1), color='orange', alpha=0.4, label='Confianza')
        ax2.axvline(-z, color='coral', linestyle='--', linewidth=1.5, label=f'-Z = {-z:.4f}')
        ax2.axvline(z, color='coral', linestyle='--', linewidth=1.5, label=f'Z = {z:.4f}')
        ax2.set_ylim(0,max(y_vals)+0.05)
        ax2.set_title("Región de Confianza (Z)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
