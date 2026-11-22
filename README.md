# 🚆 Mobilidade Carioca — Previsão de Demanda no Ramal Japeri

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Linear%20Regression-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-brightgreen)

Pipeline de **Ciência de Dados e Machine Learning** voltado à previsão horária de demanda de passageiros no fluxo pendular do **Ramal Japeri (Nova Iguaçu → Central do Brasil)**, na Região Metropolitana do Rio de Janeiro.

---

## 🎯 Problema de Negócio

Na malha de transporte metropolitano do Rio de Janeiro, existe uma forte assimetria de capacidade entre os modais:
* **1 composição ferroviária (6 carros):** comporta entre **1.200 e 1.300 passageiros** no horário de pico.
* **1 ônibus convencional:** comporta em média **80 passageiros**.

Qualquer flutuação não prevista equivalente a uma única composição exige a injeção imediata de cerca de **15 ônibus alimentadores** no sistema rodoviário para evitar o colapso dos terminais de integração. Antecipar a demanda com precisão permite o dimensionamento preventivo de frotas e a injeção estratégica de trens reguladores.

---

## 🏗️ Arquitetura do Pipeline

```text
[Simulação Senoidal + Picos] ➔ [Stress Test (Anomalias)] ➔ [Limpeza IQR & Regras] ➔ [Feature Engineering (Lags)] ➔ [Regressão Linear (Validação Temporal)]
```

1. **Simulação & Injeção de Ruído (`gerar_dados_simulados`):** Modelagem matemática do fluxo diário com picos pendulares (07h, 08h, 17h e 18h) e redução aos fins de semana.
2. **Stress Test & Limpeza (`limpar_anomalias`):** Simulação de falhas reais de sensores (pico falso de `5.000` passageiros às 02h e leitura negativa de `-100` às 00h), tratadas via filtro de regra de negócio (`passageiros >= 0`) e **Intervalo Interquartil (IQR)**.
3. **Engenharia de Recursos (`criar_features_temporais`):** Criação de memória autorregressiva de curto prazo (`lag_1`, `lag_2`, `media_3h`) e marcadores temporais (`is_pico`, `is_fim_semana`).
4. **Modelagem & Validação Temporal (`treinar_e_avaliar`):** Treinamento com **Regressão Linear** respeitando rigorosamente a ordem cronológica (75% treino / 25% teste), eliminando *data leakage*.

---

## 📊 Resultados Obtidos

| Métrica | Descrição | Resultado (Teste) |
| :--- | :--- | :--- |
| **MAE** | Erro Médio Absoluto | **~7,46 passageiros** |
| **RMSE** | Raiz do Erro Quadrático Médio | **~7,46 passageiros** |

O erro médio inferior a 8 passageiros representa uma variação estatisticamente desprezível frente à capacidade de uma composição (~1.250 passageiros), viabilizando a tomada de decisão operacional em tempo real com baixíssimo custo computacional.

---

## 📁 Estrutura do Repositório

```text
mobilidade-machine-learning/
├── notebooks/                # Notebooks exploratórios (Google Colab / Jupyter)
├── src/
│   ├── __init__.py
│   └── pipeline.py           # Módulo principal de simulação, limpeza IQR, features e treino
├── app.py                    # Dashboard interativo em Streamlit
├── requirements.txt          # Dependências do projeto
├── .gitignore
└── README.md
```

---

## 🚀 Como Executar Localmente

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/helderedias/mobilidade-machine-learning.git
   cd mobilidade-machine-learning
   ```

2. **Crie um ambiente virtual e instale as dependências:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/Mac:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Execute o pipeline via terminal:**
   ```bash
   python -m src.pipeline
   ```

4. **Inicie o dashboard interativo (Streamlit):**
   ```bash
   streamlit run app.py
   ```

---

## 👤 Autor

**Hélder Estevão Dias**  
Analista e Engenheiro de Dados | Tecnologia em Gestão de Dados (UFPI) | Técnico em Inteligência Artificial (IFNMG)
