import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pipeline import (
    aplicar_stress_test,
    criar_features_temporais,
    gerar_dados_simulados,
    limpar_anomalias,
    treinar_e_avaliar,
)

st.set_page_config(
    page_title="Mobilidade Carioca | Previsão de Demanda - Ramal Japeri",
    page_icon="🚆",
    layout="wide",
)

st.title("🚆 Mobilidade Carioca — Previsão de Demanda (Ramal Japeri)")
st.markdown(
    "Pipeline preditivo de Machine Learning para antecipação do fluxo pendular "
    "**Nova Iguaçu → Central do Brasil**, tratamento de anomalias via IQR e "
    "dimensionamento preventivo de frotas."
)

# Sidebar - Parâmetros de Simulação
st.sidebar.header("⚙️ Parâmetros da Simulação")
dias_simulacao = st.sidebar.slider("Janela histórica (dias)", min_value=30, max_value=120, value=60, step=15)
ativar_stress = st.sidebar.checkbox("Ativar Stress Test (5.000 e -100 pax)", value=True)
capacidade_trem = st.sidebar.number_input("Capacidade por composição (pax)", value=1250, step=50)
capacidade_onibus = st.sidebar.number_input("Capacidade por ônibus alimentador (pax)", value=80, step=5)

# Execução do Pipeline
df_raw = gerar_dados_simulados(dias=dias_simulacao)
df_entrada = aplicar_stress_test(df_raw) if ativar_stress else df_raw.copy()
df_clean, stats = limpar_anomalias(df_entrada)
df_feat = criar_features_temporais(df_clean)
res = treinar_e_avaliar(df_feat)

test_df = res["test_df"]

# KPIs Principais
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE (Erro Médio Absoluto)", f"{res['mae']:.2f} pax")
col2.metric("RMSE (Raiz do Erro Quadrático)", f"{res['rmse']:.2f} pax")
col3.metric("Outliers IQR Removidos", f"{stats['outliers_removidos']}")
col4.metric("Leituras Negativas Filtradas", f"{stats['negativos_removidos']}")

st.divider()

# Gráfico 1: Real vs Previsto no Conjunto de Teste
st.subheader("📈 Validação Temporal: Demanda Real vs. Predição (Regressão Linear)")
ultimas_horas = st.slider("Exibir últimas N horas do conjunto de teste", min_value=48, max_value=len(test_df), value=168, step=24)
df_plot = test_df.tail(ultimas_horas)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_plot["timestamp"],
        y=df_plot["passageiros"],
        mode="lines",
        name="Demanda Real (Limpa)",
        line=dict(color="#1f77b4", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=df_plot["timestamp"],
        y=df_plot["predicao"],
        mode="lines",
        name="Predição (Regressão Linear)",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    )
)
fig.update_layout(
    xaxis_title="Data / Hora",
    yaxis_title="Passageiros / Hora",
    hovermode="x unified",
    height=420,
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(fig, use_container_width=True)

# Seção Operacional: Conversão Modal
st.divider()
st.subheader("🚌 Simulador Operacional: Dimensionamento Preventivo na Central do Brasil")
demanda_pico_prevista = float(test_df["predicao"].max())
trens_necessarios = demanda_pico_prevista / capacidade_trem
onibus_equivalentes = demanda_pico_prevista / capacidade_onibus

c1, c2, c3 = st.columns(3)
c1.metric("Pico Horário Previsto", f"{demanda_pico_prevista:,.0f} pax/h")
c2.metric("Composições Equivalentes", f"{trens_necessarios:.2f} trens")
c3.metric("Ônibus Alimentadores Equivalentes", f"{onibus_equivalentes:.0f} ônibus")

st.info(
    f"💡 **Impacto Operacional:** Uma variação não prevista equivalente a 1 composição "
    f"({capacidade_trem} passageiros) exige a mobilização imediata de aproximadamente "
    f"**{round(capacidade_trem / capacidade_onibus)} ônibus alimentadores** no sistema rodoviário."
)
