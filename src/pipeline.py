import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def gerar_dados_simulados(dias: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Simula o fluxo horário de passageiros no Ramal Japeri (Nova Iguaçu -> Central do Brasil)
    combinando padrão senoidal diário, acréscimos nos horários de pico e redução nos fins de semana.
    """
    rng = np.random.default_rng(seed)
    horas_totais = dias * 24
    datas = pd.date_range(start="2026-01-01 00:00:00", periods=horas_totais, freq="h")

    hora = datas.hour.to_numpy()
    dia_semana = datas.dayofweek.to_numpy()
    is_fim_semana = (dia_semana >= 5).astype(int)

    # Curva base senoidal (mínimo na madrugada, máximo durante o dia)
    base = 450 + 350 * np.sin((hora - 6) * np.pi / 12)
    base = np.clip(base, 40, None)

    # Acréscimo de pico pendular (07h, 08h, 17h, 18h) em dias úteis
    is_pico = np.isin(hora, [7, 8, 17, 18]).astype(int)
    acrescimo_pico = np.where((is_pico == 1) & (is_fim_semana == 0), 150.0, 0.0)

    # Redução de fluxo aos sábados e domingos
    fator_fds = np.where(is_fim_semana == 1, 0.55, 1.0)

    # Ruído gaussiano operacional
    ruido = rng.normal(loc=0.0, scale=8.5, size=horas_totais)

    passageiros = (base + acrescimo_pico) * fator_fds + ruido
    # Período fechado (01h às 03h)
    passageiros = np.where(np.isin(hora, [1, 2, 3]), 0.0, passageiros)
    passageiros = np.round(passageiros, 1)

    df = pd.DataFrame({
        "timestamp": datas,
        "hora": hora,
        "dia_semana": dia_semana,
        "is_fim_semana": is_fim_semana,
        "is_pico": is_pico,
        "passageiros_bruto": passageiros,
    })
    return df


def aplicar_stress_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Injeta anomalias extremas para validar a resiliência do pipeline:
    - Pico falso de 5.000 passageiros às 02:00 (fora de operação)
    - Incoerência física de -100 passageiros às 00:00
    """
    df_stress = df.copy()
    idx_02h = df_stress[df_stress["hora"] == 2].index[0]
    idx_00h = df_stress[df_stress["hora"] == 0].index[1]

    df_stress.loc[idx_02h, "passageiros_bruto"] = 5000.0
    df_stress.loc[idx_00h, "passageiros_bruto"] = -100.0
    return df_stress


def limpar_anomalias(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpeza em duas camadas:
    1. Regra de negócio: descarta leituras fisicamente impossíveis (passageiros < 0).
    2. Filtro estatístico IQR (Intervalo Interquartil) para remoção de outliers extremos.
    """
    df_clean = df.copy()
    serie = df_clean["passageiros_bruto"].copy()

    negativos_removidos = int((serie < 0).sum())
    serie[serie < 0] = np.nan

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers_mask = (serie < limite_inferior) | (serie > limite_superior)
    outliers_removidos = int(outliers_mask.sum())
    serie[outliers_mask] = np.nan

    # Interpolação linear temporal para recompor falhas pontuais de sensor
    df_clean["passageiros"] = serie.interpolate(method="linear").bfill().ffill()

    stats = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "limite_superior": float(limite_superior),
        "negativos_removidos": negativos_removidos,
        "outliers_removidos": outliers_removidos,
    }
    return df_clean, stats


def criar_features_temporais(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engenharia de Recursos autorregressivos de memória de curto prazo:
    - lag_1: demanda na hora anterior (t-1)
    - lag_2: demanda duas horas antes (t-2)
    - media_3h: média móvel das últimas 3 horas (t-1 a t-3)
    """
    df_feat = df.copy()
    df_feat["lag_1"] = df_feat["passageiros"].shift(1)
    df_feat["lag_2"] = df_feat["passageiros"].shift(2)
    df_feat["media_3h"] = df_feat["passageiros"].shift(1).rolling(window=3).mean()
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat


def treinar_e_avaliar(df_feat: pd.DataFrame, test_size: float = 0.25) -> dict:
    """
    Treina Regressão Linear respeitando estritamente a ordem cronológica (sem data leakage).
    """
    features = ["lag_1", "lag_2", "media_3h", "is_pico", "is_fim_semana"]
    target = "passageiros"

    split_idx = int(len(df_feat) * (1 - test_size))
    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    test_df["predicao"] = y_pred
    test_df["erro_absoluto"] = np.abs(test_df[target] - test_df["predicao"])

    return {
        "modelo": modelo,
        "features": features,
        "mae": mae,
        "rmse": rmse,
        "train_df": train_df,
        "test_df": test_df,
    }


if __name__ == "__main__":
    df_sim = gerar_dados_simulados(dias=60)
    df_stress = aplicar_stress_test(df_sim)
    df_clean, stats = limpar_anomalias(df_stress)
    df_feat = criar_features_temporais(df_clean)
    resultados = treinar_e_avaliar(df_feat)

    print("=== Pipeline Mobilidade Carioca (Ramal Japeri) ===")
    print(f"Anomalias tratadas -> Negativos: {stats['negativos_removidos']} | Outliers IQR: {stats['outliers_removidos']}")
    print(f"MAE (Teste):  {resultados['mae']:.2f} passageiros")
    print(f"RMSE (Teste): {resultados['rmse']:.2f} passageiros")
