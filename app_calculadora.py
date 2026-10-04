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
    amplitud=sup-inf
    return (inf, sup), z, ee, me, amplitud

def ic_media_varianza_desconocida(x_barra, s, n, confianza):
    alpha = 1 - confianza
    gl = n - 1
    t = stats.t.ppf(1 - alpha / 2, df=gl)
    ee = s / np.sqrt(n)
    me = t * ee
    inf, sup = x_barra - me, x_barra + me
    amplitud=sup-inf
    return (inf, sup), gl, t, ee, me, amplitud

def ic_varianza(s2, n, confianza):
    alpha = 1 - confianza
    gl = n - 1
    chi2_inf = stats.chi2.ppf(alpha / 2, df=gl)
    chi2_sup = stats.chi2.ppf(1 - alpha / 2, df=gl)
    var_inf = (gl * s2) / chi2_sup
    var_sup = (gl * s2) / chi2_inf
    amplitud=var_sup-var_inf
    return (var_inf, var_sup), gl, chi2_inf, chi2_sup, amplitud

def ic_proporcion(p_hat, n, confianza):
    alpha = 1 - confianza
    z = stats.norm.ppf(1 - alpha / 2)
    ee = np.sqrt(p_hat * (1 - p_hat) / n)
    me = z * ee
    inf = max(0.0, p_hat - me)
    sup = min(1.0, p_hat + me)
    amplitud=sup-inf
    return (inf, sup), z, ee, me, amplitud

def ic_dif_medias_varianzas_conocidas(x_barra1, x_barra2, sigma1, sigma2, n1, n2, confianza):
    alpha = 1 - confianza
    z = stats.norm.ppf(1 - alpha / 2)

    # Error estándar para diferencia de medias con sigmas conocidas
    ee = np.sqrt((sigma1**2 / n1) + (sigma2**2 / n2))

    # Margen de error y estimación puntual
    me = z * ee
    dif_medias = x_barra1 - x_barra2

    inf, sup = dif_medias - me, dif_medias + me
    amplitud=sup-inf
    return (inf, sup), z, ee, me, dif_medias, amplitud

def ic_dif_medias_varianzas_desconocidas_iguales(x_barra1, x_barra2, s1, s2, n1, n2, confianza):
    alpha = 1 - confianza
    df = n1 + n2 - 2
    t = stats.t.ppf(1 - alpha / 2, df=df)

    # (Sp^2)
    sp2 = (((n1 - 1) * (s1**2)) + ((n2 - 1) * (s2**2))) / df

    # Error estándar 
    ee = np.sqrt(sp2 * ((1 / n1) + (1 / n2)))

    # Margen de error y estimación puntual
    me = t * ee
    dif_medias = x_barra1 - x_barra2

    inf, sup = dif_medias - me, dif_medias + me
    amplitud=sup-inf
    return (inf, sup), t, ee, me, dif_medias, df, amplitud

def ic_dif_proporciones(p1_hat, p2_hat, n1, n2, confianza):
    alpha = 1 - confianza
    z = stats.norm.ppf(1 - alpha / 2)

    # Error estándar para la diferencia de proporciones
    ee = np.sqrt(
        (p1_hat * (1 - p1_hat) / n1) + (p2_hat * (1 - p2_hat) / n2)
    )

    # Margen de error y estimación puntual
    me = z * ee
    dif_prop = p1_hat - p2_hat

    inf, sup = dif_prop - me, dif_prop + me
    amplitud=sup-inf
    return (inf, sup), z, ee, me, dif_prop, amplitud

# ──────────────────────────────────────────────
# INTERFAZ DE USUARIO (STREAMLIT)
# ──────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("cropped-logo_FCE.png", width=500)
st.markdown(
    "<h1 style='text-align: center;'>Calculadora de Intervalos de"
    " Confianza</h1>",
    unsafe_allow_html=True,
)
st.write("Elaborado por la Profesora Dra. Natalia Salaberry")
st.write("Seleccione el tipo de intervalo que desea calcular y el nivel de confianza en el menu lateral e ingrese a continuación los datos correspondientes.")

# Lateral para selección del tipo de IC
tipo_ic = st.sidebar.selectbox(
    "Tipo de Intervalo de Confianza:",
    [   "Seleccione un intervalo",
        "Media — Varianza poblacional CONOCIDA",
        "Media — Varianza poblacional DESCONOCIDA",
        "Varianza",
        "Proporción",
        "Diferencia de Medias — Vars CONOCIDAS",
        "Diferencia de Medias — Vars DESCONOCIDAS (Iguales)",
        "Diferencia de Proporciones"
    ]
)

# Nivel de Confianza 
confianza_sel = st.sidebar.selectbox("Nivel de confianza:", ["0%","90%", "95%", "99%", "Otro"])
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

st.sidebar.image("esquema.png", use_container_width=True)

# 1. MEDIA VARIANZA CONOCIDA
if tipo_ic == "Media — Varianza poblacional CONOCIDA":
    st.header("IC para la Media (μ) — Varianza Poblacional Conocida")
    
    col1, col2 = st.columns(2)
    with col1:
        x_barra = st.number_input("Media muestral (x̄):", value=0.0)
        sigma = st.number_input("Desviación estándar poblacional (σ):", value=1.0, min_value=0.0001)
        n = st.number_input("Tamaño de muestra (n):", value=30, min_value=2)
    
    if st.button("Calcular Intervalo"):
        intervalo, Z, ee, me, amplitud = ic_media_varianza_conocida(x_barra, sigma, n, confianza)
        inf, sup = intervalo
        
        st.markdown(f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f} ≤ μ ≤ {sup:.4f}] = {confianza*100:.2f}%</div>', unsafe_allow_html=True)

        st.markdown("<h3 style='font-size: 18px;'>Componentes Clave</h3>",unsafe_allow_html=True,)
        st.write(f"**Valor crítico Z:** {Z:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")
        st.write(f"**Amplitud IC:** {amplitud:.4f}")

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
        st.subheader("Fórmula utilizada y Distribuciones")
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
        intervalo, gl, t, ee, me, amplitud = ic_media_varianza_desconocida(x_barra, s, n, confianza)
        inf, sup = intervalo
        
        st.markdown(f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f} ≤ μ ≤ {sup:.4f}] = {confianza*100:.2f}%</div>', unsafe_allow_html=True)

        st.markdown("<h3 style='font-size: 18px;'>Componentes Clave</h3>",unsafe_allow_html=True,)
        st.write(f"**Grados de libertad (n-1):** {gl}")
        st.write(f"**Valor crítico t:** {t:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")
        st.write(f"**Amplitud IC:** {amplitud:.4f}")

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
        
        st.subheader("Fórmula utilizada y Distribuciones")
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
        intervalo, gl, chi2_inf, chi2_sup, amplitud = ic_varianza(s2, n, confianza)
        var_inf, var_sup = intervalo
        
        st.markdown(f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{var_inf:.4f} ≤ σ² ≤ {var_sup:.4f}] = {confianza*100:.2f}%</div>', unsafe_allow_html=True)

        st.markdown("<h3 style='font-size: 18px;'>Componentes Clave</h3>",unsafe_allow_html=True,)
        st.write(f"**Grados de libertad (n-1):** {gl}")
        st.write(f"**Valor crítico inferior:** {chi2_inf:.4f}")
        st.write(f"**Valor crítico superior:** {chi2_sup:.4f}")
        st.write(f"**Amplitud IC:** {amplitud:.4f}")
        
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
        
        st.subheader("Fórmula utilizada y Distribuciones")
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
        intervalo, z, ee, me, amplitud = ic_proporcion(p_hat, n, confianza)
        inf, sup = intervalo
        
        st.markdown(f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f} ≤ p ≤ {sup:.4f}] = {confianza*100:.2f}%</div>', unsafe_allow_html=True)

        st.markdown("<h3 style='font-size: 18px;'>Componentes Clave</h3>",unsafe_allow_html=True,)
        st.write(f"**Valor crítico Z:** {z:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")
        st.write(f"**Amplitud IC:** {amplitud:.4f}")

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
        
        st.subheader("Fórmula utilizada y Distribuciones")
        st.latex(r"IC \left[ \hat{p} - Z_{1-{\alpha \over 2}} * \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \leq p \leq \hat{p} + Z_{1-{\alpha \over 2}} *\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} \right]=1-\alpha")
        st.latex(r"X \sim Bi(n ; p) \quad | \quad \hat p \sim N(\bar p ; {\sqrt {p(1-p) \over n}}) \quad | \quad Z_{obs}={\hat p - p \over {\sqrt {p(1-p) \over n}}} \sim N(0;1)")
        
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
        
#DIF MEDIAS CON VAR CONOCIDAS
elif tipo_ic == "Diferencia de Medias — Vars CONOCIDAS":
    st.header(
        "IC para la Diferencia de Medias (μ₁ - μ₂) — Varianzas Poblacionales"
        " Conocidas"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Muestra 1")
        x_barra1 = st.number_input("Media muestral (x̄₁):", value=0.0, key="x1")
        sigma1 = st.number_input(
            "Desviación estándar (σ₁):",
            value=1.0,
            min_value=0.0001,
            key="sig1",
        )
        n1 = st.number_input(
            "Tamaño de muestra (n₁):", value=30, min_value=2, key="n1"
        )

    with col2:
        st.subheader("Muestra 2")
        x_barra2 = st.number_input("Media muestral (x̄₂):", value=0.0, key="x2")
        sigma2 = st.number_input("Desviación estándar (σ₂):",value=1.0,min_value=0.0001,key="sig2",)
        n2 = st.number_input("Tamaño de muestra (n₂):", value=30, min_value=2, key="n2")

    if st.button("Calcular Intervalo"):
        # Llamada a la función de diferencia de medias con sigmas conocidas
        intervalo, Z, ee, me, dif_medias, amplitud = ic_dif_medias_varianzas_conocidas(
            x_barra1, x_barra2, sigma1, sigma2, n1, n2, confianza
        )
        inf, sup = intervalo

        st.markdown(
            f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f}'
            f" ≤ μ₁ - μ₂ ≤ {sup:.4f}] = {confianza*100:.2f}%</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<h3 style='font-size: 18px;'>Componentes Clave</h3>",unsafe_allow_html=True,)
        st.write(f"**Valor crítico Z:** {Z:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")
        st.write(f"**Amplitud IC:** {amplitud:.4f}")

        # Gráfico de Intervalo
        fig1, ax1 = plt.subplots(figsize=(6, 1.02))
        ax1.hlines(y=1,xmin=inf,xmax=sup,colors="crimson",linewidth=4,label=f"IC {confianza*100:.2f}%",)
        ax1.plot([inf, inf], [0.85, 1.15], color="crimson", lw=2.5)
        ax1.plot([sup, sup], [0.85, 1.15], color="crimson", lw=2.5)

        ax1.text(inf,1.25,f"Lim Inf: {inf:.4f}",horizontalalignment="center",fontweight="bold",color="crimson",fontsize=8,)
        ax1.text(sup,1.25,f"Lim Sup: {sup:.4f}",horizontalalignment="center",fontweight="bold",color="crimson",fontsize=8,)
        ax1.set_ylim(0.4, 1.6)
        ax1.set_yticks([])
        ax1.tick_params(axis="both", labelsize=7)
        ax1.set_xlabel("Escala de la Diferencia de Medias (μ₁ - μ₂)", fontsize=7)
        ax1.set_title("Gráfico del Intervalo de Confianza", fontsize=9)
        ax1.legend(loc="lower right", fontsize=7)
        rango = sup - inf if (sup - inf) > 0 else 1.0
        ax1.set_xlim(inf - rango * 0.2, sup + rango * 0.2)
        ax1.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig1)

        # Fórmulas en LaTeX
        st.subheader("Fórmula utilizada y Distribuciones")
        st.latex(
            r"\small IC \left[ (\bar{X}_1 - \bar{X}_2) - Z_{1-{\alpha \over 2}} \cdot"
            r" \sqrt{{\sigma_1^2 \over n_1} + {\sigma_2^2 \over n_2}} \leq \mu_1"
            r" - \mu_2 \leq (\bar{X}_1 - \bar{X}_2) + Z_{1-{\alpha \over 2}}"
            r" \cdot \sqrt{{\sigma_1^2 \over n_1} + {\sigma_2^2 \over n_2}}"
            r" \right] = 1-\alpha"
        )
        st.latex( r"\small X_1 \sim N(\mu ; \sigma) \quad | \quad X_2 \sim N(\mu ; \sigma)")
        st.latex(r"\small Z_{obs} = {(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2) \over"
            r" \sqrt{{\sigma_1^2 \over n_1} + {\sigma_2^2 \over n_2}}} \sim"
            r" N(0;1)"
        )
        
        # Gráfico Densidad Normal
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        x_vals = np.linspace(-4, 4, 500)
        y_vals = stats.norm.pdf(x_vals, 0, 1)
        ax2.plot(x_vals, y_vals, label="N(0,1)", color="darkorange", lw=2)
        x_fill = np.linspace(-Z, Z, 200)
        ax2.fill_between(x_fill,stats.norm.pdf(x_fill, 0, 1),color="orange",alpha=0.4,label="Confianza",)
        ax2.axvline(-Z,color="coral",linestyle="--",linewidth=1.5,label=f"-Z = {-Z:.4f}",)
        ax2.axvline(Z, color="coral", linestyle="--", linewidth=1.5, label=f"Z = {Z:.4f}")
        ax2.set_ylim(0, max(y_vals) + 0.05)
        ax2.set_title("Región de Confianza")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

# DIF MEDIAS CON VAR DESCONOCIDAS (IGUALES)
elif (tipo_ic== "Diferencia de Medias — Vars DESCONOCIDAS (Iguales)"):
    st.header(
        "IC para la Diferencia de Medias (μ₁ - μ₂) — Varianzas Poblacionales"
        " Desconocidas pero Iguales"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Muestra 1")
        x_barra1 = st.number_input("Media muestral (x̄₁):", value=0.0, key="x1")
        s1 = st.number_input(
            "Desviación estándar muestral (S₁):",
            value=1.0,
            min_value=0.0001,
            key="s1",
        )
        n1 = st.number_input(
            "Tamaño de muestra (n₁):", value=30, min_value=2, key="n1"
        )

    with col2:
        st.subheader("Muestra 2")
        x_barra2 = st.number_input("Media muestral (x̄₂):", value=0.0, key="x2")
        s2 = st.number_input(
            "Desviación estándar muestral (S₂):",
            value=1.0,
            min_value=0.0001,
            key="s2",
        )
        n2 = st.number_input(
            "Tamaño de muestra (n₂):", value=30, min_value=2, key="n2"
        )

    if st.button("Calcular Intervalo"):
        (
            intervalo,
            t,
            ee,
            me,
            dif_medias,
            df,
            amplitud,
        ) = ic_dif_medias_varianzas_desconocidas_iguales(
            x_barra1, x_barra2, s1, s2, n1, n2, confianza
        )
        inf, sup = intervalo

        st.markdown(
            f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f}'
            f" ≤ μ₁ - μ₂ ≤ {sup:.4f}] = {confianza*100:.2f}%</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<h3 style='font-size: 18px;'>Componentes Clave</h3>",unsafe_allow_html=True,)
        st.write(f"**Grados de libertad:** {df:.0f}")
        st.write(f"**Valor crítico t:** {t:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")
        st.write(f"**Amplitud IC:** {amplitud:.4f}")

        # Gráfico de Intervalo
        fig1, ax1 = plt.subplots(figsize=(6, 1.02))
        ax1.hlines(y=1,xmin=inf,xmax=sup,colors="crimson",linewidth=4,label=f"IC {confianza*100:.2f}%",)
        ax1.plot([inf, inf], [0.85, 1.15], color="crimson", lw=2.5)
        ax1.plot([sup, sup], [0.85, 1.15], color="crimson", lw=2.5)

        ax1.text(inf,1.25,f"Lim Inf: {inf:.4f}",horizontalalignment="center",fontweight="bold",color="crimson",fontsize=8,)
        ax1.text(sup,1.25,f"Lim Sup: {sup:.4f}",horizontalalignment="center",fontweight="bold",color="crimson",fontsize=8,)
        ax1.set_ylim(0.4, 1.6)
        ax1.set_yticks([])
        ax1.tick_params(axis="both", labelsize=7)
        ax1.set_xlabel("Escala de la Diferencia de Medias (μ₁ - μ₂)", fontsize=7)
        ax1.set_title("Gráfico del Intervalo de Confianza", fontsize=9)
        ax1.legend(loc="lower right", fontsize=7)
        rango = sup - inf if (sup - inf) > 0 else 1.0
        ax1.set_xlim(inf - rango * 0.2, sup + rango * 0.2)
        ax1.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig1)

        # Fórmulas en LaTeX
        st.subheader("Fórmula utilizada y Distribuciones")
        st.latex(
            r"\small IC \left[ (\bar{X}_1 - \bar{X}_2) - t_{1-{\alpha \over 2}, n_1+n_2-2}"
            r" \cdot S_p \sqrt{{1 \over n_1} + {1 \over n_2}} \leq \mu_1 -\mu_2"
            r" \leq (\bar{X}_1 - \bar{X}_2) + t_{1-{\alpha \over 2},"
            r" n_1+n_2-2} \cdot S_p \sqrt{{1 \over n_1} + {1 \over n_2}} \right]"
            r" = 1-\alpha"
        )
        st.latex( r"\small X_1 \sim N(\mu ; \sigma) \quad | \quad X_2 \sim N(\mu ; \sigma)")
        st.latex( r" \small S_p^2 = {(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2 \over n_1 + n_2 - 2}"
            r" \quad | \quad T_{obs} = {(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)"
            r" \over S_p \sqrt{{1 \over n_1} + {1 \over n_2}}} \sim t_{n_1+n_2-2}"
        )

        # Gráfico Densidad t-Student
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        x_vals = np.linspace(-4, 4, 500)
        y_vals = stats.t.pdf(x_vals, df=df)
        ax2.plot(x_vals, y_vals, label=f"t({df} g.l.)", color="darkorange", lw=2)
        x_fill = np.linspace(-t, t, 200)
        ax2.fill_between(x_fill,stats.t.pdf(x_fill, df=df),color="orange",alpha=0.4,label="Confianza",)
        ax2.axvline(-t,color="coral",linestyle="--",linewidth=1.5,label=f"-t = {-t:.4f}",)
        ax2.axvline(t, color="coral", linestyle="--", linewidth=1.5, label=f"t = {t:.4f}")
        ax2.set_ylim(0, max(y_vals) + 0.05)
        ax2.set_title("Región de Confianza (Distribución t-Student)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)


# DIFERENCIA DE PROPORCIONES
elif tipo_ic == "Diferencia de Proporciones":
    st.header("IC para la Diferencia de Proporciones (p₁ - p₂)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Muestra 1")
        p1_hat = st.number_input(
            "Proporción muestral p̂₁ (entre 0 y 1):",
            value=0.50,
            min_value=0.0,
            max_value=1.0,
            key="p1",
        )
        n1 = st.number_input(
            "Tamaño de muestra (n₁):", value=30, min_value=1, key="n1_p"
        )

    with col2:
        st.subheader("Muestra 2")
        p2_hat = st.number_input(
            "Proporción muestral p̂₂ (entre 0 y 1):",
            value=0.40,
            min_value=0.0,
            max_value=1.0,
            key="p2",
        )
        n2 = st.number_input(
            "Tamaño de muestra (n₂):", value=30, min_value=1, key="n2_p"
        )
    
        
    if st.button("Calcular Intervalo"):
        # Llamada a la función de diferencia de proporciones
        intervalo, Z, ee, me, dif_prop, amplitud = ic_dif_proporciones(p1_hat, p2_hat, n1, n2, confianza)
        inf, sup = intervalo

        st.markdown(
            f'<div class="result-box"><b>Intervalo calculado:</b><br>IC [{inf:.4f}'
            f" ≤ p₁ - p₂ ≤ {sup:.4f}] = {confianza*100:.2f}%</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<h3 style='font-size: 18px;'>Componentes Clave</h3>",unsafe_allow_html=True,)
        st.write(f"**Valor crítico Z:** {Z:.4f}")
        st.write(f"**Error estándar:** {ee:.4f}")
        st.write(f"**Margen de error:** ± {me:.4f}")
        st.write(f"**Amplitud IC:** {amplitud:.4f}")
        

        # Gráfico de Intervalo
        fig1, ax1 = plt.subplots(figsize=(6, 1.02))
        ax1.hlines(y=1,xmin=inf,xmax=sup,colors="crimson",linewidth=4,label=f"IC {confianza*100:.2f}%",)
        ax1.plot([inf, inf], [0.85, 1.15], color="crimson", lw=2.5)
        ax1.plot([sup, sup], [0.85, 1.15], color="crimson", lw=2.5)

        ax1.text(inf,1.25,f"Lim Inf: {inf:.4f}",horizontalalignment="center",fontweight="bold",color="crimson",fontsize=8,)
        ax1.text(sup,1.25,f"Lim Sup: {sup:.4f}",horizontalalignment="center",fontweight="bold",color="crimson",fontsize=8,)
        ax1.set_ylim(0.4, 1.6)
        ax1.set_yticks([])
        ax1.tick_params(axis="both", labelsize=7)
        ax1.set_xlabel(
            "Escala de la Diferencia de Proporciones (p₁ - p₂)", fontsize=7
        )
        ax1.set_title("Gráfico del Intervalo de Confianza", fontsize=9)
        ax1.legend(loc="lower right", fontsize=7)
        rango = sup - inf if (sup - inf) > 0 else 1.0
        ax1.set_xlim(inf - rango * 0.2, sup + rango * 0.2)
        ax1.grid(True, axis="x", alpha=0.3)
        st.pyplot(fig1)

        # Fórmulas en LaTeX
        st.subheader("Fórmula utilizada y Distribuciones")
        st.latex(
            r"\scriptsize IC \left[ (\hat{p}_1 - \hat{p}_2) - Z_{1-{\alpha \over 2}} \cdot"
            r" \sqrt{{\hat{p}_1(1-\hat{p}_1) \over n_1} + {\hat{p}_2(1-\hat{p}_2)"
            r" \over n_2}} \leq p_1 - p_2 \leq (\hat{p}_1 - \hat{p}_2) +"
            r" Z_{1-{\alpha \over 2}} \cdot \sqrt{{\hat{p}_1(1-\hat{p}_1) \over"
            r" n_1} + {\hat{p}_2(1-\hat{p}_2) \over n_2}} \right] = 1-\alpha"
        )
        st.latex( r"\small X_1 \sim Bi(n_1 ; p_1) \quad | \quad X_2 \sim Bi(n_2 ; p_2)")
        st.latex(
            r"\small Z_{obs} = {(\hat{p}_1 - \hat{p}_2) - (p_1 - p_2) \over"
            r" \sqrt{{\hat{p}_1(1-\hat{p}_1) \over n_1} + {\hat{p}_2(1-\hat{p}_2)"
            r" \over n_2}}} \sim N(0;1)"
        )

        # Gráfico Densidad Normal
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        x_vals = np.linspace(-4, 4, 500)
        y_vals = stats.norm.pdf(x_vals, 0, 1)
        ax2.plot(x_vals, y_vals, label="N(0,1)", color="darkorange", lw=2)
        x_fill = np.linspace(-Z, Z, 200)
        ax2.fill_between(x_fill,stats.norm.pdf(x_fill, 0, 1),color="orange",alpha=0.4,label="Confianza",)
        ax2.axvline(-Z,color="coral",linestyle="--",linewidth=1.5,label=f"-Z = {-Z:.4f}",)
        ax2.axvline(Z, color="coral", linestyle="--", linewidth=1.5, label=f"Z = {Z:.4f}")
        ax2.set_ylim(0, max(y_vals) + 0.05)
        ax2.set_title("Región de Confianza")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
