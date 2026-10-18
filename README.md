# Classificação de Doença Hepática com Machine Learning e Explicabilidade (SHAP)

Pipeline completa em **R** para predição de doença hepática a partir de exames laboratoriais, usando o dataset **Indian Liver Patient Dataset (ILPD)**. O projeto compara quatro algoritmos de classificação com validação cruzada repetida, avalia-os em um conjunto de teste independente e explica as decisões do modelo com **valores SHAP**.

---

## Sumário

1. [Visão geral](#1-visão-geral)
2. [O problema](#2-o-problema)
3. [O dataset](#3-o-dataset)
4. [Estrutura do projeto](#4-estrutura-do-projeto)
5. [Como executar](#5-como-executar)
6. [Metodologia — etapa por etapa](#6-metodologia--etapa-por-etapa)
7. [Resultados obtidos](#7-resultados-obtidos)
8. [Explicabilidade — como interpretar os gráficos SHAP](#8-explicabilidade--como-interpretar-os-gráficos-shap)
9. [Decisões metodológicas e justificativas](#9-decisões-metodológicas-e-justificativas)
10. [Limitações e trabalhos futuros](#10-limitações-e-trabalhos-futuros)
11. [Referências](#11-referências)

---

## 1. Visão geral

```
CSV bruto ──► EDA ──► Pré-processamento (split 80/20 ──► imputação/normalização só no treino)
                                                │
                     Treino 4 modelos: CV 5x3 com SMOTE DENTRO de cada fold + Random Search
                                                │
   SHAP (XGB, LightGBM e RF)              ◄── Avaliação no teste (métricas, limiar, por sexo) ◄──┘
```

| Item | Detalhe |
|---|---|
| Linguagem | R 4.5 |
| Framework de modelagem | `caret` (validação cruzada, tuning, comparação) |
| Modelos | Regressão Logística, Random Forest, XGBoost, LightGBM |
| Balanceamento | SMOTE **dentro de cada fold** da CV (via `trainControl(sampling = ...)`) |
| Busca de hiperparâmetros | **Random Search** (30 combinações por modelo) |
| Métrica de seleção | AUC-ROC; Recall analisado via limiar clínico (Sens ≥ 90 %) |
| Explicabilidade | SHAP: **TreeSHAP exato** (XGBoost/LightGBM) e `fastshap` (Random Forest), visualização com `shapviz` |
| Auditoria de viés | Métricas, ROC e SHAP **estratificados por sexo** |
| Reprodutibilidade | Seed global `42` em todas as etapas |
| Tempo total de execução | ≈ 7–8 min (dominado pelo tuning do XGBoost, ≈ 4,5 min) |

---

## 2. O problema

Doenças hepáticas frequentemente são silenciosas nos estágios iniciais. Exames de sangue de rotina (bilirrubina, enzimas hepáticas, proteínas) carregam sinais que podem indicar comprometimento do fígado antes de sintomas clínicos.

**Objetivo:** dado um conjunto de exames laboratoriais + idade + sexo, prever se o paciente é **doente hepático (classe positiva)** ou **saudável**.

**Contexto clínico da métrica:** em triagem, o erro mais grave é o **falso negativo** (dizer que um doente está saudável). Por isso o projeto dá ênfase a **sensibilidade (recall)** e **AUC-ROC**, e não apenas à acurácia.

---

## 3. O dataset

**Indian Liver Patient Dataset (ILPD)** — UCI Machine Learning Repository.
583 pacientes coletados no nordeste de Andhra Pradesh, Índia.

| Variável | Descrição | Tipo |
|---|---|---|
| `Age` | Idade do paciente | numérica |
| `Gender` | Sexo (Male / Female) | categórica |
| `Total_Bilirubin` | Bilirrubina total (mg/dL) | numérica |
| `Direct_Bilirubin` | Bilirrubina direta (mg/dL) | numérica |
| `Alkaline_Phosphotase` | Fosfatase alcalina (IU/L) | numérica |
| `Alamine_Aminotransferase` | ALT / TGP (IU/L) | numérica |
| `Aspartate_Aminotransferase` | AST / TGO (IU/L) | numérica |
| `Total_Proteins` | Proteínas totais (g/dL) | numérica |
| `Albumin` | Albumina (g/dL) | numérica |
| `AG_Ratio` | Razão albumina/globulina | numérica |
| `Diagnostico` | **Alvo**: 1 = doente, 2 = saudável (chamada `Selector` na documentação da UCI) | binária |

Características relevantes descobertas na EDA:

- **Desbalanceamento**: ~71% doentes vs ~29% saudáveis (razão ≈ 2,5 : 1).
- **4 valores ausentes** em `AG_Ratio`.
- **13 linhas duplicadas**.
- **Forte assimetria** (skewness > 2) nas bilirrubinas e nas enzimas — típico de exames laboratoriais, onde poucos pacientes têm valores extremamente altos.
- **Alta correlação** (|r| > 0,7) entre `Total_Bilirubin` × `Direct_Bilirubin`, `ALT` × `AST` e `Total_Proteins` × `Albumin` (esperado: medem processos fisiológicos relacionados).

---

## 4. Estrutura do projeto

```
tcc-ilpd-interpretabilidade/
├── ExecutarTudo.R              # Orquestrador: roda tudo, cronometra e grava log
├── UtilsPipeline.R             # Log, persistência, SMOTE p/ caret (por fold) e wrapper LightGBM
├── 0.InstalarPacotes.R         # Verifica/instala dependências (versões fixadas)
├── 1.CarregamentoDeDados.R     # Leitura + análise exploratória (EDA)
├── 2.PreProcessamento.R        # Limpeza, encoding, log-transform, split, imputação e normalização (só treino)
├── 3.Balanceamento.R           # SMOTE no treino completo (diagnóstico e gráfico)
├── 4.Modelos.R                 # Treino dos 4 modelos: CV repetida + SMOTE por fold + Random Search
├── 5.Avaliacao.R               # Teste: métricas, ROC, PR, DeLong, limiar (todos), estratificação por sexo
├── 6.SHAP.R                    # Explicabilidade global, local e por sexo (XGB, LightGBM e RF)
│
├── data/
│   ├── Indian Liver Patient Dataset (ILPD).csv   # dados brutos
│   ├── df_raw.rds              # dados carregados
│   ├── df_proc.rds             # treino pré-processado (distribuição real; entrada da etapa 4)
│   ├── df_balanceado.rds       # treino após SMOTE (apenas diagnóstico — etapa 3)
│   ├── teste.rds               # teste (nunca tocado pelo SMOTE)
│   ├── teste_genero.rds        # sexo dos pacientes do teste (rótulo legível, p/ estratificação)
│   └── preproc.rds             # parâmetros de normalização (média/DP do treino)
│
├── models/                     # modelos treinados (.rds, objetos caret)
│   ├── modelo_lr.rds
│   ├── modelo_rf.rds
│   ├── modelo_xgb.rds
│   └── modelo_lgb.rds
│
├── results/                    # tabelas (CSV) e citações (BibTeX) prontas para o texto do TCC
│   ├── citacoes.bib                        # BibTeX do R e dos pacotes tidyverse e corrplot
│   ├── estatisticas_descritivas.csv
│   ├── comparacao_medias_por_classe.csv
│   ├── pares_alta_correlacao.csv           # pares de variáveis com |r| > 0,7
│   ├── coeficientes_regressao_logistica.csv
│   ├── cv_auc_por_modelo.csv               # AUC, Sens e Spec médias na CV
│   ├── metricas_teste.csv                  # métricas no limiar padrão e no limiar clínico (Sens ≥ 90 %)
│   ├── delong_pareado.csv
│   ├── metricas_por_sexo.csv               # AUC/Sens/Spec/FN por sexo e modelo
│   ├── shap_importancia_<m>.csv            # m = xgb | rf | lgb
│   ├── shap_direcao_efeito_<m>.csv
│   ├── shap_importancia_por_sexo_<m>.csv
│   ├── comparacao_importancia_modelos.csv  # rankings SHAP (XGB, LightGBM, RF) × |coef. RL|
│   └── concordancia_rankings_spearman.csv
│
├── plots/                      # figuras (PNG, 150 dpi)
│   ├── eda_*.png               # exploração
│   ├── smote_antes_depois.png
│   ├── cv_comparacao_modelos.png
│   ├── roc_teste.png / precision_recall_teste.png / metricas_teste.png
│   ├── metricas_por_sexo.png
│   ├── shap_importancia_<m>.png / shap_beeswarm_<m>.png
│   ├── shap_dependencia_<m>_<var>.png
│   ├── shap_waterfall_<m>_<caso>.png
│   └── shap_importancia_por_sexo_<m>.png / shap_gender_por_sexo_<m>.png
│
├── logs/
│   └── pipeline_<timestamp>.log   # cópia integral do console de cada execução
│                                  # (execução de referência: pipeline_20261011_003612.log)
│
├── latex/                      # monografia do TCC (abnTeX2): monografia.tex, textuais/, postextuais/bibliografia3.bib
└── documents/                  # proposta do TC
```

---

## 5. Como executar

### Pré-requisitos

- R ≥ 4.3 (testado em 4.5.0)
- Acesso à internet na primeira execução (instalação de pacotes)

### Execução completa

No RStudio / DataSpell, com o diretório de trabalho na raiz do projeto:

```r
source("ExecutarTudo.R")
```

Ou pelo terminal:

```bash
Rscript ExecutarTudo.R
```

O script:

1. Verifica e instala pacotes faltantes (`0.InstalarPacotes.R`).
2. Executa as etapas 1 → 6 em sequência.
3. Cronometra cada etapa.
4. Espelha **todo** o console em `logs/pipeline_<timestamp>.log`.
5. Interrompe com mensagem clara se alguma etapa falhar.

### Execução por etapa

Cada script é independente e pode ser rodado isoladamente, desde que os artefatos da etapa anterior existam em `data/` ou `models/`:

```r
source("5.Avaliacao.R")   # reavalia os modelos já treinados sem retreinar
source("6.SHAP.R")        # recalcula só a explicabilidade
```

### Versões fixadas (por quê)

| Pacote | Versão | Motivo |
|---|---|---|
| `xgboost` | 1.7.8.1 | Versões 2.x quebram a integração com `caret::train(method = "xgbTree")` |
| `fastshap` | 0.1.1 | Pacote arquivado no CRAN; instalado a partir do archive |

---

## 6. Metodologia — etapa por etapa

### Etapa 0 — Instalação de pacotes

Verifica cada dependência e instala apenas o que falta. Registra no log a versão do R, a plataforma e a seed global. Fixa `xgboost` e `fastshap` nas versões compatíveis.

### Etapa 1 — Carregamento e análise exploratória

- Gera `results/citacoes.bib` com as entradas BibTeX oficiais do R e dos pacotes `tidyverse` e `corrplot` (já incorporadas, com chaves, em `latex/postextuais/bibliografia3.bib`).
- Leitura do CSV com nomes de colunas explícitos.
- Tipos das variáveis, contagem de `NA` e duplicatas.
- Distribuição da variável alvo e razão de desbalanceamento.
- Tabela cruzada sexo × classe.
- Estatísticas descritivas (média, DP, mín, mediana, máx, **assimetria**) → `results/estatisticas_descritivas.csv`.
- Média de cada exame entre doentes e saudáveis (e a razão entre elas) → `results/comparacao_medias_por_classe.csv`; comparação apenas descritiva.
- Matriz de correlação de Pearson com alerta para pares com |r| > 0,7 → `results/pares_alta_correlacao.csv`.
- Gráficos: distribuição do alvo, boxplots por classe, heatmap de correlação.

### Etapa 2 — Pré-processamento

| Passo | O que faz | Por quê |
|---|---|---|
| Remoção de duplicatas | 583 → 570 linhas | Duplicatas inflam artificialmente a confiança e podem vazar entre treino e teste |
| Encoding | `Gender`: Male=1, Female=0; `Diagnostico`: `"doente"` / `"saudavel"` | `caret` exige rótulos textuais válidos para calcular probabilidades |
| Transformação `log1p` | Bilirrubinas, fosfatase alcalina, ALT, AST | Reduz assimetria extrema; melhora modelos lineares e a estabilidade do SMOTE (transformação determinística, sem parâmetros estimados) |
| **Split estratificado 80/20** | 457 treino / 113 teste | Estratificação garante a mesma proporção de classes nos dois conjuntos (71,1% vs 71,7%) |
| Imputação | `AG_Ratio` ausente ← **mediana do treino** (1,00), aplicada a treino e teste | Feita **após** o split: o teste não influencia o valor imputado. Mediana é robusta a outliers; apenas 4 casos |
| Normalização (center + scale) | Ajustada **somente no treino**, aplicada ao teste | Evita **data leakage**: o teste não pode influenciar os parâmetros de escala |
| Sexo do teste | Salvo em `teste_genero.rds` com rótulo legível | Necessário para a análise estratificada por sexo (etapas 5 e 6) |

> **Ponto crítico:** o split é feito **antes** de qualquer passo que estime parâmetros a partir dos dados (imputação, normalização, SMOTE). Se fosse depois, informação dos pacientes do teste contaminaria o treino e as métricas ficariam otimistas.

### Etapa 3 — Balanceamento com SMOTE (diagnóstico)

- SMOTE (Synthetic Minority Over-sampling Technique) com `K = 5` vizinhos, via a mesma função `aplicar_smote()` usada na etapa 4.
- **`Gender` é binária:** o SMOTE interpola todas as variáveis, o que geraria "sexos intermediários" quando os dois pacientes de origem são de sexos diferentes. `aplicar_smote()` devolve o `Gender` de cada amostra sintética para a **categoria válida mais próxima** (a amostra herda o sexo do paciente de origem mais próximo no segmento de interpolação). Os pacientes reais e as variáveis contínuas não mudam. É uma versão simplificada da ideia do SMOTE-NC (Chawla et al., 2002).
- Aplicado ao treino completo apenas para **diagnóstico e visualização**: 457 → 589 observações (132 sintéticas).
- Verificação de integridade: compara as médias das amostras sintéticas com as da classe minoritária original, registra quantas sintéticas tinham `Gender` intermediário e precisaram de correção (são raras, porque a distância usada para achar os vizinhos já inclui o `Gender` padronizado e aproxima pacientes do mesmo sexo) e confere que o `Gender` sintético só assume os dois valores válidos.
- O arquivo `df_balanceado.rds` **não** é usado para treinar — ver etapa 4.

### Etapa 4 — Treinamento dos modelos

**Validação cruzada repetida: 5 folds × 3 repetições = 15 avaliações por combinação de hiperparâmetros.** A métrica de seleção é a **AUC-ROC**.

**SMOTE dentro de cada fold.** O `trainControl` recebe `sampling = smote_caret` (definido em `UtilsPipeline.R`). Em cada iteração da CV o SMOTE é aplicado **apenas à partição de treino do fold**; a partição de validação contém somente pacientes reais. Sem isso (SMOTE antes da CV), amostras sintéticas — interpolações de pacientes que também estão na validação — vazam entre as partições e inflam a AUC da CV. O ajuste final de cada modelo também aplica SMOTE ao treino completo. Em todos esses casos, o `Gender` das amostras sintéticas volta à categoria válida mais próxima (ver etapa 3).

**Random Search.** `trainControl(search = "random")` sorteia 30 combinações de hiperparâmetros por modelo (`tuneLength = 30`); o wrapper do LightGBM implementa o mesmo comportamento na sua função `grid`.

| Modelo | Tipo | Hiperparâmetros sorteados |
|---|---|---|
| **Regressão Logística** | Linear, interpretável (baseline) | — |
| **Random Forest** | Bagging de 500 árvores | `mtry` (1–10; apenas 10 valores distintos possíveis) |
| **XGBoost** | Gradient boosting | `nrounds`, `max_depth`, `eta`, `gamma`, `colsample_bytree`, `min_child_weight`, `subsample` |
| **LightGBM** | Gradient boosting (histogram-based) | `num_leaves`, `learning_rate` (log-uniforme), `nrounds`, `feature_fraction`, `min_data_in_leaf` |

O `caret` não possui método nativo para LightGBM, então `UtilsPipeline.R` define um **wrapper customizado** (`lightgbm_caret`) implementando as funções `fit`, `predict`, `prob`, `varImp` e `grid`. Isso permite que o LightGBM participe da mesma validação cruzada e da comparação via `resamples()`, em pé de igualdade com os demais.

Saídas adicionais:

- **Coeficientes e odds ratios** da regressão logística com p-valores → interpretação direta do efeito de cada exame.
- AUC, **sensibilidade** e especificidade médias na CV por modelo → `results/cv_auc_por_modelo.csv`.
- Boxplot da AUC nos 15 folds para cada modelo.

### Etapa 5 — Avaliação no conjunto de teste

Para cada modelo:

- Matriz de confusão com VP / FP / FN / VN.
- Acurácia, sensibilidade, especificidade, precisão e AUC.
- **Intervalo de confiança de 95% da AUC** pelo método de **DeLong**.
- **Limiar clínico (foco em Recall)**: o maior limiar que ainda garante **Sens ≥ 90 %** — quantifica o custo, em especificidade e precisão, de priorizar os falsos negativos em cada modelo.
- **O limiar clínico é escolhido sem olhar o teste**: saem das predições *out-of-fold* da validação cruzada (`modelo$pred`, guardadas com `savePredictions = "final"`; 457 pacientes × 3 repetições = 1.371 predições, cada uma feita por um modelo que não viu aquele paciente). Depois é aplicado, fixo, às probabilidades do teste, onde se medem sensibilidade, especificidade e precisão (`Sens_Recall90`, `Spec_Recall90`, `Precisao_Recall90`). Escolher o limiar no próprio teste tornaria essas métricas otimistas (Leeflang et al., 2008); por isso a sensibilidade no teste com o limiar clínico pode ficar um pouco abaixo de 90 %.

Comparação entre modelos:

- Tabela consolidada → `results/metricas_teste.csv`.
- **Teste de DeLong pareado** para todas as combinações de modelos, com **correção de Bonferroni** para as 6 comparações (`p_valor` original e `p_ajustado_Bonferroni`; a coluna `Significativo_5pct` usa o ajustado) → `results/delong_pareado.csv`.
- Curvas ROC sobrepostas e gráfico de métricas.

**Análise estratificada por sexo (auditoria de viés — Straw & Wu, 2022):**

- Prevalência de doença por sexo no teste.
- Para cada modelo e sexo: AUC com IC DeLong, sensibilidade, especificidade, precisão, **taxa de falsos negativos** (limiar 0,5 e limiar clínico escolhido na CV) → `results/metricas_por_sexo.csv`.
- Gráfico: `metricas_por_sexo.png`.

### Etapa 6 — Explicabilidade com SHAP

Aplicada aos três modelos de *ensemble* — **XGBoost**, **LightGBM** e **Random Forest** —, sobre os 113 pacientes do teste:

| Modelo | Método | Escala dos valores SHAP |
|---|---|---|
| XGBoost | **TreeSHAP exato** (`predict(..., predcontrib = TRUE)`) | log-odds |
| LightGBM | **TreeSHAP exato** (`predcontrib` nativo) | log-odds |
| Random Forest | `fastshap` (Monte Carlo, `nsim = 100`) — RF não possui TreeSHAP nativo em R | probabilidade |

> O `caret::xgbTree` codifica internamente o **primeiro** nível (`saudavel`) como 1; a etapa 6 inverte o sinal das contribuições para que, em todos os modelos, **SHAP > 0 signifique "empurra para doente"**.

| Saída (`<m>` = `xgb`, `rf` ou `lgb`) | O que responde |
|---|---|
| `shap_importancia_<m>.png` + `.csv` | Quais variáveis mais pesam nas decisões, em média? |
| `shap_beeswarm_<m>.png` | Em qual direção cada variável empurra a previsão? |
| `shap_direcao_efeito_<m>.csv` | Correlação valor-da-variável × SHAP, com interpretação textual |
| `shap_dependencia_<m>_<var>.png` | Como o efeito muda conforme o valor da variável (a mais importante de cada modelo) |
| `shap_waterfall_<m>_*.png` | Por que o modelo decidiu assim para **um paciente específico**? Dois casos: o verdadeiro positivo mais confiante e o **pior falso negativo** |
| `shap_importancia_por_sexo_<m>.png` + `.csv` | O modelo usa as variáveis de forma diferente para homens e mulheres? (ranking por sexo + Spearman) |
| `shap_gender_por_sexo_<m>.png` | Quanto a variável `Gender`, isoladamente, desloca a previsão de risco em cada sexo? |
| `comparacao_importancia_modelos.csv` | Rankings SHAP (XGB, LightGBM, RF) × \|coeficiente\| da regressão logística |
| `concordancia_rankings_spearman.csv` | Correlação de Spearman entre os quatro rankings |

> **Empates nos rankings:** em todos os rankings (SHAP global, SHAP por sexo e coeficientes), variáveis com importância exatamente igual — tipicamente as que um modelo não usa — recebem o **posto médio**, calculado antes do arredondamento (ex.: as três variáveis que o XGBoost não usa empatam nas posições 8 a 10 e recebem posto 9). Assim o empate não é desfeito pela ordem das colunas, e o Spearman é calculado sobre os postos médios.

---

## 7. Resultados obtidos

> Valores da execução de referência (`logs/pipeline_20261011_003612.log`, de 11/10/2026). Podem variar levemente entre máquinas por diferenças de paralelismo, mas a seed fixa (`42`) garante reprodutibilidade na mesma máquina. Todas as tabelas estão em `results/` e todas as figuras em `plots/`.

### 7.1 Análise exploratória

#### Desbalanceamento das classes

![Distribuição da variável alvo](plots/eda_distribuicao_alvo.png)

Dos 583 registros originais, **416 são doentes (71,4%)** e **167 saudáveis (28,6%)** — razão de 2,5 : 1. Isso tem duas consequências diretas:

- Um classificador trivial que rotulasse todo mundo como "doente" já atingiria 71% de acurácia. Por isso a **acurácia simples não é a métrica principal** deste projeto.
- Sem tratamento, os modelos tenderiam a favorecer a classe majoritária. Daí o uso de SMOTE (Seção 7.2).

#### Distribuição dos exames por classe

![Boxplots por classe](plots/eda_boxplots_por_classe.png)

Os boxplots já revelam, antes de qualquer modelo, quais exames separam os grupos:

| Exame | Média doentes | Média saudáveis | Leitura |
|---|---|---|---|
| Total_Bilirubin | 4,16 | 1,14 | ~3,6× maior nos doentes |
| Direct_Bilirubin | 1,92 | 0,40 | ~4,9× maior nos doentes |
| Alamine_Aminotransferase (ALT) | 99,6 | 33,7 | ~3× maior |
| Aspartate_Aminotransferase (AST) | 137,7 | 40,7 | ~3,4× maior |
| Alkaline_Phosphotase | 319,0 | 219,8 | ~1,5× maior |
| Albumin | 3,06 | 3,34 | **menor** nos doentes |
| AG_Ratio | 0,91 | 1,03 | **menor** nos doentes |
| Age | 46,2 | 41,2 | doentes ~5 anos mais velhos |
| Total_Proteins | 6,46 | 6,54 | **praticamente igual** |

Quase todos os exames diferem entre as classes; `Total_Proteins` é a exceção — o que torna interessante o fato de ela aparecer entre as variáveis mais importantes no SHAP do XGBoost e com o maior odds ratio da regressão logística (Seções 7.3 e 7.5): os modelos a usam em **interação** com `Albumin`, algo que uma comparação de uma variável por vez não captura.

Os boxplots também expõem o problema de **assimetria extrema**: em ALT, AST e bilirrubinas a caixa fica achatada na base do gráfico e os outliers se estendem até dezenas de vezes a mediana. A transformação `log1p` reduziu a assimetria de AST de **10,5 → 1,24** e de ALT de **6,66 → 1,46**.

#### Correlação entre exames

![Matriz de correlação](plots/eda_correlacao.png)

Três pares ultrapassam |r| > 0,7:

- `Total_Bilirubin` × `Direct_Bilirubin` = **0,874** — a bilirrubina direta é um componente da total.
- `Alamine_Aminotransferase` × `Aspartate_Aminotransferase` = **0,792** — ambas são transaminases liberadas na lesão do hepatócito.
- `Total_Proteins` × `Albumin` = **0,783** — albumina é a principal proteína plasmática.

Essa multicolinearidade explica por que, na regressão logística (Seção 7.3), `Total_Bilirubin` e `AST` **não** aparecem como significativas apesar de discriminarem fortemente as classes: seus efeitos são "absorvidos" pelas variáveis correlacionadas `Direct_Bilirubin` e `ALT`. Modelos de árvore são imunes a esse problema, mas o SHAP também redistribui a importância entre variáveis correlacionadas.

### 7.2 Balanceamento com SMOTE

![SMOTE antes e depois](plots/smote_antes_depois.png)

Após remover 13 duplicatas e separar o teste, o treino tem 457 observações (**325 doentes / 132 saudáveis**, razão 2,46 : 1). Aplicado ao treino completo (etapa 3, diagnóstico), o SMOTE com `dup_size = 0` gera **132 amostras sintéticas** — dobrando a classe minoritária — e resulta em **589 observações (325 doentes / 264 saudáveis, razão 1,23 : 1)**.

Note que o balanceamento **não é perfeito** (55% / 45%). Essa foi uma escolha deliberada: com `dup_size = 0` o `smotefamily` gera o menor número inteiro de cópias sintéticas que aproxima as classes, evitando inflar demais o treino com dados artificiais.

Verificação de integridade das 132 sintéticas: suas médias ficam a no máximo 0,05 DP das médias da classe saudável original em todas as variáveis (ex.: `Age` −0,26 vs −0,23; `Albumin` 0,31 vs 0,27, em escala padronizada). Só **1** sintética tinha `Gender` intermediário (pacientes de origem de sexos diferentes) e foi devolvida à categoria mais próxima; ao final, o `Gender` sintético tem apenas valores válidos (33 mulheres / 99 homens).

> **Nota metodológica:** este gráfico é apenas ilustrativo. No treino real (etapa 4) o SMOTE é reaplicado **dentro de cada um dos 15 folds**, somente à partição de treino do fold. As partições de validação e o conjunto de teste (113 pacientes, 71,7% doentes) contêm apenas pacientes reais.

### 7.3 Treinamento e validação cruzada

#### Desempenho na CV (5 folds × 3 repetições = 15 estimativas por modelo, SMOTE por fold)

![Comparação na validação cruzada](plots/cv_comparacao_modelos.png)

| Modelo | AUC média | DP | AUC mín | AUC máx | Sens. média | Spec. média |
|---|---|---|---|---|---|---|
| **Regressão Logística** | **0,753** | 0,054 | 0,654 | 0,836 | 0,667 | **0,694** |
| XGBoost | 0,725 | 0,064 | 0,616 | 0,827 | 0,701 | 0,639 |
| Random Forest | 0,723 | 0,074 | 0,581 | 0,810 | **0,751** | 0,515 |
| LightGBM | 0,720 | 0,064 | 0,607 | 0,792 | 0,685 | 0,659 |

> Sensibilidade e especificidade referem-se à classe positiva `doente`. O `twoClassSummary` do caret toma o primeiro nível do desfecho (`saudavel`) como evento, por isso o pipeline usa um *wrapper* (`resumo_doente`) que troca as duas métricas; a AUC não é afetada.

Com o SMOTE aplicado corretamente dentro de cada fold, o quadro muda em relação ao que se obtém com SMOTE *antes* da CV (onde os modelos de árvore atingiam AUC ≈ 0,85–0,87 e a regressão logística ≈ 0,77):

- **A regressão logística passa a ser o melhor modelo na CV**, e os três modelos de árvore formam um bloco em torno de 0,72, com caixas amplamente sobrepostas.
- A queda de ~0,13 de AUC nos modelos de árvore confirma que **o SMOTE antes da CV vaza informação**: as amostras sintéticas são interpolações de pacientes que também estão na partição de validação, e modelos flexíveis aprendem essa estrutura. A regressão logística, mais rígida, praticamente não se beneficiava do vazamento (0,774 → 0,753).
- O Random Forest tem a **maior sensibilidade (0,74) e a menor especificidade (0,51) na CV** no limiar 0,5: com `mtry = 1` suas probabilidades ficam concentradas perto de 0,5 e o limiar padrão separa mal as classes — reforçando a necessidade da análise de limiar (Seção 7.4).

**Conclusão da CV:** a regressão logística tem a maior AUC média (cerca de 0,03 acima dos três modelos de árvore, que ficam praticamente empatados), mas a diferença é bem menor que a variação entre as partições. A comparação estatística entre os modelos é feita no conjunto de teste (DeLong, Seção 7.4).

#### Hiperparâmetros selecionados (Random Search, 30 combinações)

| Modelo | Melhor configuração | Comentário |
|---|---|---|
| Random Forest | `mtry = 1`, 500 árvores | Uma única variável candidata por split → árvores máximamente descorrelacionadas; coerente com a forte multicolinearidade dos exames |
| XGBoost | `nrounds = 24`, `max_depth = 4`, `eta = 0,23`, `gamma = 7,3`, `colsample_bytree = 0,35`, `min_child_weight = 16`, `subsample = 0,49` | Configuração **fortemente regularizada** (poucas rodadas, `gamma` alto, 35 % das variáveis por árvore). Consequência: três variáveis nunca são usadas (ver Seção 7.5) |
| LightGBM | `num_leaves = 4`, `learning_rate = 0,012`, `nrounds = 200`, `feature_fraction = 0,57`, `min_data_in_leaf = 40` | Árvores rasas com aprendizado lento e folhas grandes — também regularizado |

O padrão é consistente: **a busca escolheu, para os três ensembles, as configurações mais simples do espaço**. É o comportamento esperado quando a CV é honesta e o sinal nos dados é essencialmente aditivo.

#### Coeficientes da regressão logística

Como as variáveis estão padronizadas, cada coeficiente representa o efeito de **+1 desvio-padrão** na variável sobre o log-odds de doença (modelo final ajustado no treino com SMOTE):

| Variável | Coeficiente | Odds Ratio | p-valor | Interpretação |
|---|---|---|---|---|
| Total_Proteins | +2,06 | **7,85** | < 0,001 | +1 DP multiplica a chance de doença por ~8 |
| AG_Ratio | +1,34 | 3,81 | 0,002 | efeito positivo (surpreendente — ver abaixo) |
| Alamine_Aminotransferase | +0,58 | 1,79 | 0,009 | ALT alta ≈ +79 % na chance |
| Alkaline_Phosphotase | +0,43 | 1,53 | 0,004 | |
| Age | +0,32 | 1,37 | 0,002 | cada DP (~16 anos) aumenta 37 % |
| Gender (masculino) | −0,21 | 0,81 | 0,037 | homens têm chance ~19 % menor, **ajustado pelos exames** |
| **Albumin** | **−2,85** | **0,058** | < 0,001 | +1 DP reduz a chance em ~94 % — o efeito protetor mais forte |
| Direct_Bilirubin | +0,38 | 1,46 | 0,525 | não significativo (colinear com Total_Bilirubin) |
| Total_Bilirubin | +0,28 | 1,32 | 0,646 | não significativo |
| Aspartate_Aminotransferase | +0,28 | 1,32 | 0,202 | não significativo (colinear com ALT) |

Três pontos merecem discussão no texto do TCC:

1. **`Total_Proteins` e `Albumin` com sinais opostos e magnitudes enormes.** Como as duas têm r = 0,78, o modelo está efetivamente usando a *diferença* entre elas — que é aproximadamente a **globulina**. Globulina alta com albumina baixa é o padrão clássico de doença hepática crônica. A regressão logística "descobriu" isso sozinha.
2. **`AG_Ratio` com OR > 1** contradiz o teste univariado (doentes têm AG_Ratio *menor*). É um efeito de supressão: uma vez controlado por albumina e proteínas totais, o coeficiente muda de sinal. Isso ilustra por que coeficientes de regressão múltipla **não** devem ser interpretados isoladamente quando há multicolinearidade.
3. **`Gender` é significativo (p = 0,037) e protetor para homens.** Isso é o oposto da prevalência bruta (79 % dos homens do teste são doentes vs 53 % das mulheres): o modelo aprende que, *para o mesmo perfil laboratorial*, uma mulher tem risco ligeiramente maior. Esse é exatamente o tipo de efeito que a análise estratificada por sexo (Seção 7.4) precisa auditar.

### 7.4 Avaliação no conjunto de teste

#### Curvas ROC

![Curvas ROC no teste](plots/roc_teste.png)

| Modelo | AUC | IC 95% (DeLong) | Acurácia | Sensibilidade | Especificidade | Precisão |
|---|---|---|---|---|---|---|
| **Random Forest** | **0,787** | [0,703 ; 0,871] | **0,726** | **0,765** | 0,625 | 0,838 |
| LightGBM | 0,761 | [0,671 ; 0,851] | 0,681 | 0,691 | 0,656 | 0,836 |
| XGBoost | 0,745 | [0,650 ; 0,840] | 0,717 | 0,716 | **0,719** | **0,866** |
| Regressão Logística | 0,736 | [0,644 ; 0,829] | 0,681 | 0,728 | 0,563 | 0,808 |

Lendo o gráfico:

- As quatro curvas estão **claramente acima da diagonal**, mas **entrelaçadas** — nenhuma domina as outras em toda a extensão. As curvas são "serrilhadas" porque há apenas 113 pacientes no teste.
- **No teste, o ranking se inverte em relação à CV**: o Random Forest, terceiro na CV, tem a maior AUC; a regressão logística, primeira na CV, tem a menor. As diferenças, porém, estão todas dentro dos intervalos de confiança (largura ≈ 0,18) — o que a Seção seguinte formaliza.
- A comparação CV → teste é agora **coerente**: RL 0,753 → 0,736 (−0,017), XGBoost 0,725 → 0,745 (+0,020), RF 0,723 → 0,787 (+0,064), LightGBM 0,720 → 0,761 (+0,041). Nenhum modelo "despenca" no teste, sinal de que a CV com SMOTE por fold é uma estimativa honesta.

#### Matrizes de confusão (limiar 0,5)

| | Random Forest | XGBoost | LightGBM | Reg. Logística |
|---|---|---|---|---|
| **Verdadeiros positivos** (doente → doente) | 62 | 58 | 56 | 59 |
| **Falsos negativos** (doente → saudável) ⚠️ | 19 | 23 | 25 | 22 |
| **Falsos positivos** (saudável → doente) | 12 | 9 | 11 | 14 |
| **Verdadeiros negativos** (saudável → saudável) | 20 | 23 | 21 | 18 |

Dos 81 doentes no teste, o Random Forest deixa passar **19 (23,5 %)** no limiar padrão; o LightGBM, 25 (30,9 %). O XGBoost é o mais equilibrado (Sens ≈ Spec ≈ 0,72), refletindo sua forte regularização.

#### Comparação estatística das AUCs (teste de DeLong)

| Comparação | AUC A | AUC B | p-valor | p ajustado (Bonferroni) |
|---|---|---|---|---|
| Reg. Logística vs Random Forest | 0,736 | 0,787 | 0,090 | 0,541 |
| Random Forest vs XGBoost | 0,787 | 0,745 | 0,159 | 0,956 |
| Random Forest vs LightGBM | 0,787 | 0,761 | 0,230 | 1,000 |
| Reg. Logística vs LightGBM | 0,736 | 0,761 | 0,418 | 1,000 |
| XGBoost vs LightGBM | 0,745 | 0,761 | 0,556 | 1,000 |
| Reg. Logística vs XGBoost | 0,736 | 0,745 | 0,764 | 1,000 |

**Nenhuma diferença é significativa no teste** — nem mesmo antes da correção de Bonferroni (menor p-valor bruto = 0,09). Combinado com a CV (onde a regressão logística teve a maior AUC média), o resultado do trabalho é claro: **neste dataset, os modelos de caixa-preta não oferecem ganho preditivo demonstrável sobre o modelo linear interpretável.** Esse achado é central para o objetivo 6 da proposta (trade-off desempenho × interpretabilidade).

#### Visão consolidada das métricas

![Métricas no teste por modelo](plots/metricas_teste.png)

#### Foco em Recall: limiar clínico (Sens ≥ 90 %)

A proposta pede atenção especial ao Recall. Em vez de otimizar Recall diretamente (o que leva ao modelo trivial "todos doentes"), cada modelo foi avaliado no **maior limiar que garante Sens ≥ 90 % nas predições *out-of-fold* da CV** (1.371 predições por modelo). O limiar é escolhido sem olhar o teste e depois aplicado, fixo, aos 113 pacientes do teste; por isso a sensibilidade no teste pode ficar abaixo de 90 % (no teste, 90 % significa no máximo 8 dos 81 doentes não detectados):

| Modelo | Limiar clínico (CV) | Spec na CV | Sens no teste | Spec no teste | Precisão no teste | FN / FP no teste |
|---|---|---|---|---|---|---|
| **Random Forest** | 0,375 | 0,255 | 0,901 | **0,312** | **0,768** | **8 / 22** |
| LightGBM | 0,360 | 0,220 | 0,901 | 0,281 | 0,760 | 8 / 23 |
| XGBoost | 0,361 | 0,215 | **0,840** ⚠️ | **0,312** | 0,756 | 13 / 22 |
| Regressão Logística | 0,320 | 0,318 | 0,901 | 0,188 | 0,737 | 8 / 26 |

Leitura:

- **Garantir 90 % de sensibilidade custa muita especificidade** em todos os modelos: o melhor (RF) ainda rotula 69 % dos saudáveis como doentes (22 de 32). Isso caracteriza os modelos como ferramentas de **triagem** (alta sensibilidade, seguida de exame confirmatório), não de diagnóstico.
- **A meta de 90 % se manteve no teste para RF, LightGBM e RL** (8 FN). **No XGBoost, não:** o limiar 0,361, que dava Sens 0,902 na CV, deu 0,840 no teste (13 FN). Escolher o limiar fora do teste deixa essa variação visível, em vez de escondê-la com um limiar ajustado ao próprio teste (Leeflang et al., 2008).
- No regime de alta sensibilidade o **Random Forest é o melhor**: mesma sensibilidade de RL e LightGBM com menos falsos positivos (22 vs 26 e 23) e mesma especificidade do XGBoost com 5 falsos negativos a menos (8 vs 13).

#### Análise estratificada por sexo (auditoria de viés)

![Métricas por sexo](plots/metricas_por_sexo.png)

O teste tem **32 mulheres (17 doentes, 53 %)** e **81 homens (64 doentes, 79 %)**. Como a prevalência difere muito, uma métrica agregada pode esconder desempenho desigual.

| Modelo | AUC ♀ | AUC ♂ | Δ AUC (♂−♀) | Sens ♀ | Sens ♂ | Spec ♀ | Spec ♂ | Taxa FN ♀ | Taxa FN ♂ |
|---|---|---|---|---|---|---|---|---|---|
| Regressão Logística | 0,729 | 0,748 | +0,019 | 0,765 | 0,719 | 0,467 | 0,647 | 0,235 | 0,281 |
| **Random Forest** | 0,790 | 0,781 | −0,009 | 0,824 | 0,766 | 0,667 | 0,588 | 0,176 | 0,234 |
| XGBoost | 0,733 | 0,743 | +0,009 | **0,647** | 0,734 | 0,733 | 0,706 | **0,353** | 0,266 |
| LightGBM | 0,780 | 0,737 | −0,043 | 0,706 | 0,688 | 0,800 | 0,529 | 0,294 | 0,313 |

Sensibilidade, especificidade e taxa de FN no limiar 0,5.

Leitura:

- **As AUCs por sexo estão sempre a menos de 0,05 uma da outra.** Não há sinal de que os modelos discriminem pior em um dos grupos (com a ressalva de que o grupo feminino é pequeno).
- **O Random Forest perde menos mulheres doentes** do que homens doentes (taxa de FN 17,6 % vs 23,4 %).
- **O XGBoost é o menos equânime em sensibilidade**: perde **35 % das mulheres doentes** contra 27 % dos homens doentes (Δ Sens = +0,087 a favor dos homens). Com 17 mulheres doentes, isso corresponde a 6 vs ~4,5 casos — a diferença não é significativa, mas é exatamente o padrão que Straw & Wu (2022) alertam: modelos treinados em populações majoritariamente masculinas (75 % dos pacientes do ILPD) podem ter falsos negativos concentrados nas mulheres.
- A **regressão logística e o LightGBM têm especificidade muito diferente entre sexos** (Δ = +0,18 e −0,27), ou seja, o mesmo limiar produz taxas de falso positivo distintas para homens e mulheres. Em um sistema real, isso sugere **calibrar o limiar por sexo**.
- **No limiar clínico** (escolhido na CV), a sensibilidade das mulheres fica acima da dos homens na RL, no RF e no LightGBM (0,94–1,00 vs 0,88–0,89), e o gap do XGBoost praticamente desaparece (0,82 vs 0,84). O custo recai na especificidade, baixa nos dois sexos (0,13–0,41): na RL, só 2 das 15 mulheres saudáveis são reconhecidas como saudáveis (`results/metricas_por_sexo.csv`).

> **Cautela:** o grupo feminino tem apenas 32 pacientes; os ICs das AUCs femininas têm largura ≈ 0,35. Os gaps aqui são indicativos, não conclusivos.

### 7.5 Explicabilidade (SHAP)

Foram explicados os três *ensembles*: **XGBoost** e **LightGBM** (TreeSHAP exato, escala log-odds) e **Random Forest** (maior AUC no teste; `fastshap`, escala de probabilidade). Como as escalas diferem, comparam-se **rankings e direções**, não magnitudes.

#### Importância global

| XGBoost | LightGBM | Random Forest |
|---|---|---|
| ![](plots/shap_importancia_xgb.png) | ![](plots/shap_importancia_lgb.png) | ![](plots/shap_importancia_rf.png) |

| Variável | XGB (%) | LGB (%) | RF (%) |
|---|---|---|---|
| Total_Bilirubin | **37,6** | 16,5 | 13,8 |
| Alkaline_Phosphotase | 26,4 | **23,0** | **18,4** |
| Total_Proteins | 8,5 | 0,7 | 4,9 |
| Aspartate_Aminotransferase | 8,1 | 16,8 | 11,8 |
| Age | 7,9 | 7,3 | 8,1 |
| AG_Ratio | 5,8 | 5,4 | 7,3 |
| Alamine_Aminotransferase | 5,6 | 9,1 | 12,0 |
| Direct_Bilirubin | 0,0 † | 19,2 | 14,7 |
| Albumin | 0,0 † | 2,1 | 6,0 |
| Gender | 0,0 † | 0,0 | 3,1 |

† Não usadas pelo XGBoost (média |SHAP| = 0); nos rankings, recebem o posto médio 9.

- **XGBoost é concentrado**: bilirrubina total e fosfatase alcalina somam 64 %; 4 variáveis acumulam 80 %. A regularização forte (24 árvores, `colsample_bytree = 0,35`, `gamma = 7,3`) fez o modelo usar só uma variável de cada par redundante.
- **LightGBM e Random Forest são distribuídos** (5 e 7 variáveis para 80 %): usam as duas bilirrubinas e as duas transaminases. O LightGBM quase não usa as proteínas (`Total_Proteins` 0,7 %, `Albumin` 2,1 %).
- Os três concordam que **`Gender` é (quase) irrelevante** e que **fosfatase alcalina, bilirrubinas e transaminases** dominam.

#### Direção dos efeitos — beeswarm

| XGBoost | LightGBM | Random Forest |
|---|---|---|
| ![](plots/shap_beeswarm_xgb.png) | ![](plots/shap_beeswarm_lgb.png) | ![](plots/shap_beeswarm_rf.png) |

Correlação entre o valor da variável e seu SHAP (`results/shap_direcao_efeito_<m>.csv`):

| Variável | XGB | LGB | RF |
|---|---|---|---|
| Age | +0,86 | +0,87 | +0,89 |
| Total_Bilirubin | +0,81 | +0,86 | +0,86 |
| Direct_Bilirubin | — | +0,82 | +0,82 |
| Alkaline_Phosphotase | +0,81 | +0,82 | +0,75 |
| Alamine_Aminotransferase | +0,53 | +0,79 | +0,84 |
| Aspartate_Aminotransferase | +0,72 | +0,87 | +0,82 |
| Total_Proteins | +0,36 | +0,77 | +0,43 |
| Albumin | — | −0,05 | **−0,74** |
| AG_Ratio | **−0,67** | **−0,67** | **−0,57** |
| Gender | — | — | −0,14 |

**Coerência clínica:** bilirrubinas, fosfatase alcalina e transaminases elevadas aumentam o risco; albumina e razão A/G altas o reduzem. O SHAP recupera a **direção correta de `AG_Ratio`** (protetora), ao contrário do coeficiente da regressão logística (OR = 3,8), distorcido pela multicolinearidade.

#### Gráficos de dependência (variável mais importante de cada modelo)

| XGBoost | LightGBM | Random Forest |
|---|---|---|
| ![](plots/shap_dependencia_xgb_Total_Bilirubin.png) | ![](plots/shap_dependencia_lgb_Alkaline_Phosphotase.png) | ![](plots/shap_dependencia_rf_Alkaline_Phosphotase.png) |

#### Explicações locais — piores falsos negativos

| XGBoost | LightGBM | Random Forest |
|---|---|---|
| ![](plots/shap_waterfall_xgb_FN_pior_erro.png) | ![](plots/shap_waterfall_lgb_FN_pior_erro.png) | ![](plots/shap_waterfall_rf_FN_pior_erro.png) |

- **XGBoost — paciente #78** (homem, 35 anos, prob. 0,22): bilirrubina total de 1,6 mg/dL, logo abaixo do corte de ~1,8 aprendido (−0,46 log-odds), e albumina baixa (2,5 g/dL), que o modelo não usa.
- **LightGBM — paciente #50** (homem, 30 anos, prob. 0,33): fosfatase alcalina abaixo da mediana (−0,30 log-odds), mas albumina (2,3 g/dL) e proteínas totais (4,6 g/dL) baixas, que o modelo quase não usa.
- **Random Forest — paciente #113** (homem, 31 anos, prob. 0,25): exames compatíveis com pessoa saudável; a fosfatase alcalina (−0,10) domina.
- O SHAP **distingue um erro do modelo** (XGBoost e LightGBM ignoram o sinal das proteínas) **de uma limitação dos dados** (no RF, o paciente tem exames normais).

#### SHAP estratificado por sexo

| XGBoost | LightGBM | Random Forest |
|---|---|---|
| ![](plots/shap_importancia_por_sexo_xgb.png) | ![](plots/shap_importancia_por_sexo_lgb.png) | ![](plots/shap_importancia_por_sexo_rf.png) |

- **Concordância do ranking entre mulheres e homens: Spearman = 0,988** nos três modelos.
- **Contribuição média de `Gender`:** 0 no XGBoost e no LightGBM (variável não usada); no Random Forest, **+0,0030 para mulheres e −0,0015 para homens** (probabilidade) — menos de meio ponto percentual.

#### Comparação dos rankings de importância entre métodos

| Variável | SHAP XGB | SHAP LGB | SHAP RF | \|coef.\| RL |
|---|---|---|---|---|
| Total_Bilirubin | **1** | 4 | 3 | 9 |
| Alkaline_Phosphotase | 2 | **1** | **1** | 5 |
| Total_Proteins | 3 | 9 | 9 | 2 |
| Aspartate_Aminotransferase | 4 | 3 | 5 | 8 |
| Age | 5 | 6 | 6 | 7 |
| AG_Ratio | 6 | 7 | 7 | 3 |
| Alamine_Aminotransferase | 7 | 5 | 4 | 4 |
| Gender | 9 (empate) | 10 | 10 | 10 |
| Direct_Bilirubin | 9 (empate) | 2 | 2 | 6 |
| Albumin | 9 (empate) | 8 | 8 | **1** |

Spearman (`results/concordancia_rankings_spearman.csv`): LGB × RF = **0,96**; XGB × LGB = XGB × RF = 0,36; RL × (XGB, LGB, RF) = −0,14, −0,20, −0,16.

1. **LightGBM e Random Forest concordam quase totalmente**, apesar de um usar *boosting* e o outro *bagging*: a forma como a importância se distribui depende mais da configuração (muitas árvores pequenas / `mtry = 1`) do que da estratégia de *ensemble*.
2. **O XGBoost diverge** porque sua regularização eliminou variáveis redundantes e concentrou o crédito em `Total_Bilirubin`.
3. **A regressão logística tem ranking negativamente correlacionado com todos.** Os coeficientes gigantes e opostos de `Albumin`/`Total_Proteins` são artefato da multicolinearidade: **coeficientes de um modelo linear com preditores correlacionados não são uma medida confiável de importância**.

### 7.6 Síntese dos resultados

1. **Todos os modelos superam o acaso** (AUC 0,74–0,79 no teste; nenhum IC 95% inclui 0,5), mas nenhum é excelente. O dataset é pequeno e o problema é genuinamente difícil.
2. **Não há evidência de que os modelos de caixa-preta superem a regressão logística.** Na CV honesta (SMOTE por fold) a RL tem a maior AUC média (0,753, cerca de 0,03 acima dos ensembles); no teste o RF lidera (0,787) mas nenhuma diferença é significativa (DeLong, todos p > 0,09 antes e ≥ 0,54 após Bonferroni). Para o objetivo 6 da proposta, o trade-off desempenho × interpretabilidade **favorece o modelo interpretável** neste dataset.
3. **O SMOTE antes da CV inflava a AUC dos ensembles em ~0,13** e produzia um ranking enganoso (árvores ≫ RL). Aplicá-lo dentro de cada fold é indispensável para uma comparação justa.
4. **O Random Search escolheu as configurações mais regularizadas** para os três ensembles — coerente com o sinal essencialmente aditivo dos dados.
5. **Perfil operacional é de triagem.** Buscar Sens ≥ 90 % (limiar escolhido na CV) custa Spec ≤ 0,31 no teste em todos os modelos; o RF é o melhor nesse regime (22 FP contra 23–26 do LightGBM e da RL, e 5 FN a menos que o XGBoost, que ficou abaixo da meta no teste: Sens 0,84). O limiar deve ser escolhido pelo custo relativo dos erros, não fixado em 0,5.
6. **Não há sinal de viés discriminatório por sexo na AUC** (diferenças < 0,05), mas há **assimetrias de sensibilidade** — no limiar 0,5, o XGBoost perde 35 % das mulheres doentes vs 27 % dos homens (no limiar clínico a diferença praticamente some: Sens 0,82 vs 0,84). O SHAP mostra que isso não vem do uso da variável `Gender` (que o XGBoost nem usa e que fica em último ou penúltimo lugar de importância em todos os modelos), e sim das distribuições laboratoriais distintas entre sexos, sugerindo limiares específicos por sexo.
7. **O SHAP confirma a plausibilidade clínica** dos modelos (bilirrubinas, fosfatase alcalina e transaminases ↑ risco; albumina e razão A/G ↓ risco) e recupera a direção correta de `AG_Ratio`, que o coeficiente da RL inverte por multicolinearidade.
8. **Os piores falsos negativos têm causas diferentes**: no XGBoost e no LightGBM, o modelo ignorou proteínas baixas; no Random Forest, o paciente tinha exames normais (limitação dos dados). O SHAP permite fazer essa distinção caso a caso.

---

## 8. Explicabilidade — como interpretar os gráficos SHAP

### A ideia em uma frase

O SHAP decompõe a previsão de **cada paciente** em uma soma de contribuições, uma por variável:

```
previsão (log-odds ou probabilidade) = valor base (média) + SHAP(Age) + SHAP(Albumin) + ... + SHAP(Gender)
```

- **SHAP > 0** → aquela variável, para aquele paciente, empurrou a previsão na direção de **"doente"**.
- **SHAP < 0** → empurrou para **"saudável"**.
- Quanto maior o módulo, maior o impacto.
- **Escala:** para XGBoost/LightGBM (TreeSHAP) os valores estão em **log-odds** (somam ao logit da probabilidade); para o Random Forest (`fastshap`) estão em **probabilidade**. Rankings e direções são comparáveis entre modelos; magnitudes absolutas não.

### `shap_importancia_<m>.png` — gráfico de barras

Média do |SHAP| de cada variável em todos os pacientes do teste. Barra maior = variável mais usada pelo modelo. **Não indica direção**, só magnitude.

### `shap_beeswarm_<m>.png` — "enxame de abelhas"

O gráfico mais informativo:

- **Cada linha** = uma variável (ordenada da mais para a menos importante).
- **Cada ponto** = um paciente do teste.
- **Posição horizontal** = valor SHAP (direita → mais risco de doença; esquerda → menos).
- **Cor** = valor da variável para aquele paciente (escala padronizada; cores quentes = alto, frias = baixo).

Como ler: se na linha de `Alkaline_Phosphotase` os pontos de cor quente (valor alto) estão à direita, então **fosfatase alcalina alta aumenta a probabilidade de doença**. Se em `Albumin` os pontos de cor quente estão à esquerda, **albumina alta protege**.

### `shap_dependencia_<m>_<variável>.png`

Eixo X = valor da variável; eixo Y = seu SHAP. Mostra a **forma** da relação: linear, com limiar, saturação, etc. A cor indica a variável com maior interação.

### `shap_waterfall_<m>_*.png` — explicação de um paciente

Começa no valor base (previsão média) e, barra a barra, mostra como cada variável somou ou subtraiu até chegar à previsão final daquele paciente. Dois casos são gerados:

- `VP_maior_confianca`: doente que o modelo acertou com mais certeza — mostra o "perfil típico" de doença aprendido.
- `FN_pior_erro`: **doente que o modelo classificou como saudável com mais convicção** — o caso mais importante para análise crítica: quais exames "enganaram" o modelo?

### `shap_importancia_por_sexo_<m>.png` e `shap_gender_por_sexo_<m>.png` — auditoria de viés

- O primeiro compara a média de |SHAP| de cada variável calculada **só nas mulheres** e **só nos homens** do teste. Barras parecidas e ranking concordante (Spearman alto) indicam que o modelo aplica a mesma "lógica" aos dois grupos.
- O segundo mostra a contribuição da variável `Gender` para cada paciente, separada por sexo. Se os pontos de um sexo estão sistematicamente acima de zero e os do outro abaixo, o modelo está usando o sexo **por si só** para deslocar o risco — algo que deve ser confrontado com a prevalência real de doença em cada grupo (ver `results/metricas_por_sexo.csv`).

---

## 9. Decisões metodológicas e justificativas

| Decisão | Alternativa descartada | Justificativa |
|---|---|---|
| Split **antes** da imputação, da normalização e do SMOTE | Imputar/balancear/normalizar tudo e depois dividir | Evita vazamento de informação do teste para o treino |
| **SMOTE dentro de cada fold da CV** | SMOTE uma vez, antes da CV | Amostras sintéticas são interpolações de pacientes reais; se esses pacientes caem na partição de validação, a CV fica otimista e superestima os modelos flexíveis |
| SMOTE em vez de undersampling | Remover pacientes da classe majoritária | Dataset pequeno (570); descartar dados piora a variância |
| `log1p` nas variáveis assimétricas | Manter escala original | Reduz influência de outliers extremos e melhora modelos lineares e o cálculo de vizinhos do SMOTE |
| CV repetida 5×3 | CV simples 10 folds | Repetição reduz a variância da estimativa; com 457 obs, 5 folds mantém folds de tamanho razoável |
| **Random Search** (30 combinações) | Grade exaustiva | Cobre espaços contínuos (`eta`, `learning_rate`, `subsample`) com custo fixo; a literatura mostra eficiência igual ou superior à grade para o mesmo orçamento (Bergstra & Bengio, 2012) |
| AUC como métrica de seleção + **limiar clínico** (Sens ≥ 90 %) | Acurácia; ou otimizar Recall diretamente | Acurácia é enganosa em dados desbalanceados; otimizar Recall puro leva ao modelo trivial "todos doentes". A AUC escolhe o modelo com melhor ordenação e o limiar clínico traduz isso em Recall alvo |
| Teste de DeLong no conjunto de teste | Comparar apenas médias | Diferenças de AUC de poucos centésimos podem ser ruído; testes estatísticos deixam isso explícito |
| **Bonferroni** nos testes de DeLong (6 pares de modelos) | p-valores brutos | Com várias comparações simultâneas, a chance de ao menos um "significativo" por acaso passa de 5 %; Bonferroni limita essa chance a 5 % |
| **Limiar clínico escolhido nas predições *out-of-fold* da CV** e só aplicado ao teste | Escolher o limiar nas previsões do próprio teste | Escolher no teste infla sensibilidade/especificidade, sobretudo em amostras pequenas (Leeflang et al., 2008); assim o teste continua independente |
| `Gender` sintético devolvido à **categoria válida mais próxima** | Interpolar `Gender` como se fosse contínua | Evita "sexos intermediários" sem significado; mesma ideia do SMOTE-NC (Chawla et al., 2002) |
| **Postos médios** nos empates dos rankings de importância | Desempatar pela ordem das colunas | A ordem das colunas é arbitrária e alteraria a correlação de Spearman entre rankings |
| **Estratificação por sexo** (métricas e SHAP) | Avaliar só no agregado | Prevalência e perfil laboratorial diferem entre sexos; um modelo pode ter FN concentrados em um grupo sem que a métrica global revele (Straw & Wu, 2022) |
| SHAP em vez de só importância nativa | `varImp()` do caret | SHAP fornece direção do efeito, explicações locais e é consistente entre modelos |
| **TreeSHAP exato** para XGBoost e LightGBM | Monte Carlo (`fastshap`) para todos | Exato, determinístico e instantâneo; `fastshap` fica reservado ao Random Forest, que não tem TreeSHAP nativo em R |
| SHAP nos três *ensembles* (XGB, LightGBM, RF) | Explicar só um modelo | Compara se modelos de *boosting* e de *bagging* "olham" para os mesmos exames |
| LightGBM via wrapper customizado no caret | Treinar fora do caret | Garante que o LightGBM use exatamente os mesmos folds e a mesma métrica dos outros modelos |
| Limiar clínico para todos os modelos | Fixar 0,5 | Em triagem, o custo de FN ≠ custo de FP; o limiar é uma decisão clínica, não estatística |

---

## 10. Limitações e trabalhos futuros

**Limitações**

- Amostra pequena (113 pacientes de teste; 32 mulheres) → intervalos de confiança largos; no teste, nenhuma diferença entre modelos, nem entre sexos, foi significativa.
- Dataset de uma única região geográfica — generalização para outras populações não é garantida.
- O rótulo "doente" agrupa diferentes patologias hepáticas.
- SMOTE gera amostras por interpolação linear, o que pode criar pacientes fisiologicamente implausíveis (o `Gender` é devolvido a uma categoria válida, mas as variáveis contínuas continuam interpoladas).
- Os limiares escolhidos na CV vêm de modelos ajustados com ~80 % do treino; aplicá-los ao modelo final (ajustado no treino completo) é uma aproximação, e a sensibilidade no teste pode ficar abaixo da meta de 90 % — foi o que ocorreu com o XGBoost (0,84).
- Random Search com 30 combinações explora o espaço de hiperparâmetros de forma limitada; a seleção do modelo ainda é feita na mesma CV usada para o tuning (sem CV aninhada), o que pode manter leve otimismo.
- Os valores SHAP do Random Forest são aproximados (Monte Carlo) e estão em escala diferente (probabilidade) dos de XGBoost/LightGBM (log-odds).
- Especificidade baixa em todos os modelos no limiar 0,5, e muito baixa no limiar clínico de Sens ≥ 90 %.

**Possíveis extensões**

- Validação cruzada aninhada (nested CV) para estimativa não enviesada do desempenho após tuning.
- Otimização bayesiana de hiperparâmetros (ex.: `tune`/`mlr3` ou `rBayesianOptimization`) em substituição ao Random Search.
- Calibração de probabilidades (Platt scaling / isotônica) e curva de calibração, inclusive por sexo.
- Modelos treinados separadamente por sexo ou com reponderação, para testar se reduzem o gap de FN (Straw & Wu, 2022).
- Engenharia de atributos com razões clínicas conhecidas (ex.: AST/ALT — razão de De Ritis).
- Ensemble (stacking) dos quatro modelos.
- Curvas de decisão (decision curve analysis) para avaliar utilidade clínica em vez de só desempenho estatístico.
- Validação externa em outro dataset hepático.

---

## 11. Referências

- Ramana, B. V., Babu, M. S. P., & Venkateswarlu, N. B. (2012). *ILPD (Indian Liver Patient Dataset)*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/225
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 30*.
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 30*.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated ROC curves. *Biometrics*, 44(3), 837–845.
- Kuhn, M. (2008). Building Predictive Models in R Using the caret Package. *Journal of Statistical Software*, 28(5).
- Leeflang, M. M. G., Moons, K. G. M., Reitsma, J. B., & Zwinderman, A. H. (2008). Bias in sensitivity and specificity caused by data-driven selection of optimal cutoff values: mechanisms, magnitude, and solutions. *Clinical Chemistry*, 54(4), 729–737.
- Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13, 281–305.
- Lundberg, S. M., Erion, G. G., & Lee, S.-I. (2018). Consistent Individualized Feature Attribution for Tree Ensembles. *arXiv:1802.03888* (TreeSHAP).
- Straw, I., & Wu, H. (2022). Investigating for bias in healthcare algorithms: a sex-stratified analysis of supervised machine learning models in liver disease prediction. *BMJ Health & Care Informatics*, 29(1), e100457.
- Santos, M. S., Soares, J. P., Abreu, P. H., Araújo, H., & Santos, J. (2018). Cross-Validation for Imbalanced Datasets: Avoiding Overoptimistic and Overfitting Approaches. *IEEE Computational Intelligence Magazine*, 13(4), 59–76.
