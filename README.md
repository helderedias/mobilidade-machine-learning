# 🚉 Previsão de Demanda de Passageiros — Ramal Japeri (Machine Learning)

Este projeto propõe um modelo preditivo preliminar para estimar a demanda horária de passageiros no **Eixo Baixada Fluminense - Central do Brasil**, com foco no fluxo pendular do **Ramal Japeri** a partir de Nova Iguaçu. O objetivo principal é fornecer inteligência de dados para otimização operacional e planejamento logístico de transportes.

## 🛠️ O Problema de Engenharia de Transportes
Sistemas ferroviários possuem uma capacidade de transporte massiva: uma única composição cheia do Ramal Japeri acomoda cerca de **1.200 a 1.300 passageiros** nos horários de pico. Em contrapartida, um ônibus urbano convencional comporta aproximadamente **80 pessoas** (menos de 10% da capacidade do trem).

Qualquer flutuação ou atraso não planejado na malha ferroviária exige o deslocamento imediato de até **15 ônibus extras** no asfalto para suprir a demanda pendular, gerando um severo impacto viário nas artérias de trânsito. A previsão exata da curva de passageiros permite:
* Dimensionamento cirúrgico das frotas de ônibus alimentadores/integração.
* Alocação estratégica de trens vazios reguladores partindo de estações intermediárias.
* Mitigação do impacto no tráfego urbano da Baixada Fluminense e acessos à capital.

## 🏗️ Arquitetura do Pipeline de Dados

1. **Modelagem e Simulação Temporal:** Geração de um dataset sintético baseado no comportamento cíclico real (função senoidal) do fluxo diário de passageiros, parametrizando picos explícitos nos horários de tráfego pendular crítico (07h-08h e 17h-18h).
2. **Engenharia de Resiliência (Stress Test):** Injeção de anomalias severas e dados impossíveis (outliers espúrios de volume e valores negativos) para testar o comportamento do pipeline sob falhas reais de sensores de bilhetagem.
3. **Data Cleaning:** Tratamento estatístico via **IQR (Intervalo Interquartil)** para remoção de picos falsos e aplicação de regras de negócio estritas para eliminação de registros fisicamente incoerentes.
4. **Engenharia de Recursos (Feature Engineering):** Criação de variáveis preditoras contextuais e de memória temporal, transformando o pipeline em um modelo autorregressivo de alta fidelidade:
   * `lag_1` e `lag_2`: Defasagem temporal (Janela de tráfego das horas anteriores).
   * `media_3h`: Média móvel suavizada para capturar tendências de aceleração do fluxo.
   * Recursos de calendário (identificação de fins de semana e horários de pico).

## 📊 Modelagem e Resultados

Para evitar vazamento de dados (*data leakage*), o conjunto foi ordenado e dividido de forma **estritamente temporal** (sem embaralhamento), separando o histórico final para validação. Observe que o erro médio é de apenas aproximadamente 7 passageiros.

### Algoritmo Utilizado
Foi implementado um modelo de **Regressão Linear Autorregressiva** utilizando os recursos de memória temporal criados na etapa de engenharia de recursos. 

### Métricas de Desempenho
* **Erro Médio Absoluto (MAE):** `7.46e+00` (~7.46 passageiros)
* **Raiz do Erro Quadrático Médio (RMSE):** Desempenho estável e alinhado ao MAE, indicando ausência de erros discrepantes no conjunto de teste.

Graças à combinação dos *lags*, o modelo atingiu uma precisão cirúrgica (com erro estatisticamente insignificante diante do volume total transportado), tornando-o viável para tomadas de decisão operacionais em tempo real.

## 🚀 Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone [https://github.com/helderedias/mobilidade.git](https://github.com/helderedias/mobilidade.git)
