# =============================================================================
# 2.PreProcessamento.R
# Encoding, transformacao log das variaveis assimetricas, divisao treino/teste
# ESTRATIFICADA e, SOMENTE DEPOIS, imputacao e normalizacao com parametros
# estimados no treino (evita qualquer vazamento de informacao do teste).
# =============================================================================
# [Fundamentação Teórica - 2.3 Preparacao dos Dados (sec:fund_preparacao)] Etapa 2 da Figura 2.1.
# A ordem das operacoes segue a secao 2.1: o que nao depende de parametros
# estimados (duplicatas, log) vem antes da divisao; o que depende (imputacao,
# padronizacao, SMOTE) e ajustado somente no treino (2.3.2, vazamento de dados).
# [Metodologia - 3.3 Pre-processamento (sec:preprocessamento)] este script implementa as
#   secoes 3.3.1 a 3.3.3 da Metodologia; o SMOTE (3.3.4) esta em 3.Balanceamento.R,
#   UtilsPipeline.R e 4.Modelos.R.
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
}))

log_section("ETAPA 2 - PRE-PROCESSAMENTO")
t0 <- timer_start()

df <- readRDS("data/df_raw.rds")
log_info("Dados brutos carregados: %d x %d", nrow(df), ncol(df))

# --- Duplicatas --------------------------------------------------------------
# [Metodologia - 3.3.1 Remocao de Duplicatas e Transformacao Logaritmica (sec:preproc_duplicatas)]
#   remocao dos 13 registros duplicados (583 -> 570).
# [Fundamentação Teórica - 2.3.2 Registros Duplicados e Vazamento de Dados (sec:fund_vazamento)]
#   primeira origem de vazamento: um registro no treino e sua copia no teste.
log_subsection("Remocao de duplicatas")
n_antes <- nrow(df)
df <- distinct(df)
log_info("Removidas %d linhas duplicadas (%d -> %d).", n_antes - nrow(df), n_antes, nrow(df))

# --- Encoding ----------------------------------------------------------------
# [Metodologia - 3.3.3 Imputacao e Padronizacao (sec:padronizacao)] Gender codificada
#   como masculino = 1, feminino = 0; Diagnostico -> "doente" / "saudavel".
# [Fundamentação Teórica - 2.2.2 Conceitos Basicos (sec:fund_conceitos)] variavel binaria (Gender)
#   e classe positiva ('doente') x negativa ('saudavel').
log_subsection("Codificacao de variaveis categoricas")
df$Gender  <- ifelse(df$Gender == "Male", 1, 0)
log_info("Gender  -> Male = 1, Female = 0  (%d homens, %d mulheres)", sum(df$Gender == 1), sum(df$Gender == 0))
# Rotulos textuais sao exigidos pelo caret quando classProbs = TRUE
df$Diagnostico <- factor(ifelse(df$Diagnostico == 1, "doente", "saudavel"),
                     levels = c("saudavel", "doente"))
log_info("Diagnostico -> 'doente' (classe positiva) / 'saudavel'")
log_kv(paste0("Classe ", levels(df$Diagnostico)), as.vector(table(df$Diagnostico)))

# --- Transformacao log das variaveis assimetricas ---------------------------
# [Metodologia - 3.3.1 (sec:preproc_duplicatas)] ln(1 + x) nas cinco variaveis assimetricas;
#   a assimetria antes/depois impressa aqui e a da Tabela tab:assimetria_log.
# [Fundamentação Teórica - 2.3.3 Transformacao Logaritmica e Padronizacao (sec:fund_log)]
#   log1p(x) = ln(1 + x), Eq. 2.2 (eq:log); o efeito e conferido pelo coeficiente
#   de assimetria antes e depois. Nao usa parametros estimados dos dados,
#   por isso pode vir antes da divisao treino/teste.
log_subsection("Transformacao log1p em variaveis com forte assimetria")
vars_log <- c("Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
              "Alamine_Aminotransferase", "Aspartate_Aminotransferase")
skew <- function(x) mean(((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))^3, na.rm = TRUE)
for (v in vars_log) {
  antes <- skew(df[[v]])
  df[[v]] <- log1p(df[[v]])
  log_info("%-28s assimetria %6.2f -> %5.2f", v, antes, skew(df[[v]]))
}

# --- Divisao treino/teste (estratificada) -----------------------------------
# [Metodologia - 3.3.2 Divisao Treino/Teste (sec:divisao_treino_teste)] 80/20 estratificada,
#   semente 42 fixada imediatamente antes do sorteio.
# [Fundamentação Teórica - 2.2.2 Conceitos Basicos (sec:fund_conceitos)] conjunto de treino e de teste.
# [Fundamentação Teórica - 2.5.1 Validacao Cruzada (sec:fund_cv)] estratificacao: mantem a proporcao
#   de classes. [Fundamentação Teórica - 2.1] semente fixada antes do sorteio.
log_subsection("Divisao treino / teste (80/20, estratificada pela classe)")
set.seed(PIPELINE_SEED)
idx    <- createDataPartition(df$Diagnostico, p = 0.8, list = FALSE)
treino <- df[idx, ]
teste  <- df[-idx, ]
log_info("Treino: %d obs  |  Teste: %d obs", nrow(treino), nrow(teste))
log_info("Proporcao 'doente' - treino: %.1f%%  |  teste: %.1f%%",
         100 * mean(treino$Diagnostico == "doente"), 100 * mean(teste$Diagnostico == "doente"))

# --- Imputacao (mediana estimada SOMENTE no treino) -------------------------
# [Metodologia - 3.3.3 Imputacao e Padronizacao (sec:padronizacao)] mediana do treino.
# [Fundamentação Teórica - 2.3.1 Valores Ausentes e Imputacao (sec:fund_ausentes)] mediana, robusta
#   a assimetria; [Fundamentação Teórica - 2.3.2] estimada so no treino (segunda origem de vazamento).
log_subsection("Imputacao de valores ausentes (mediana do treino aplicada a treino e teste)")
mediana_ag <- median(treino$AG_Ratio, na.rm = TRUE)
na_treino  <- sum(is.na(treino$AG_Ratio)); na_teste <- sum(is.na(teste$AG_Ratio))
treino$AG_Ratio[is.na(treino$AG_Ratio)] <- mediana_ag
teste$AG_Ratio[is.na(teste$AG_Ratio)]   <- mediana_ag
log_info("AG_Ratio: mediana do treino = %.2f | imputados: %d no treino, %d no teste.",
         mediana_ag, na_treino, na_teste)
log_ok("Total de NAs restantes: treino = %d | teste = %d", sum(is.na(treino)), sum(is.na(teste)))

# --- Normalizacao (ajustada SOMENTE no treino) ------------------------------
# [Metodologia - 3.3.3 Imputacao e Padronizacao (sec:padronizacao)] z-score com media e
#   desvio-padrao do treino, aplicados ao treino e ao teste.
# [Fundamentação Teórica - 2.3.3 Transformacao Logaritmica e Padronizacao (sec:fund_padronizacao)]
#   z-score z = (x - media) / desvio-padrao, com media e desvio estimados so no treino.
log_subsection("Normalizacao (center + scale) ajustada no conjunto de treino")
X_treino <- treino %>% select(-Diagnostico)
X_teste  <- teste  %>% select(-Diagnostico)

preproc  <- preProcess(X_treino, method = c("center", "scale"))
X_treino_s <- predict(preproc, X_treino)
X_teste_s  <- predict(preproc, X_teste)
log_info("Parametros (media / desvio) estimados em %d variaveis do treino e aplicados ao teste.", ncol(X_treino))

verif <- data.frame(Variavel = names(X_treino_s),
                    Media_Treino = round(colMeans(X_treino_s), 3),
                    DP_Treino    = round(apply(X_treino_s, 2, sd), 3),
                    Media_Teste  = round(colMeans(X_teste_s), 3),
                    row.names = NULL)
log_table(verif, digits = 3)

treino_proc <- cbind(X_treino_s, Diagnostico = treino$Diagnostico)
teste_proc  <- cbind(X_teste_s,  Diagnostico = teste$Diagnostico)

# --- Persistencia -------------------------------------------------------------
log_subsection("Salvando artefatos")
salvar_rds(treino_proc, "data/df_proc.rds")   # treino pre-processado (entrada do SMOTE e da CV)
salvar_rds(teste_proc,  "data/teste.rds")     # teste intocado pelo SMOTE
salvar_rds(preproc,     "data/preproc.rds")
# [Metodologia - 3.6.4 Avaliacao Estratificada por Sexo (sec:avaliacao_sexo)] sexo dos
#   pacientes de teste guardado para as analises por subgrupo.
# Genero em rotulo legivel, para a analise estratificada por sexo (etapas 5 e 6)
# [Fundamentação Teórica - 2.7 Vies Algoritmico e Equidade em Saude (sec:fund_vies)]
genero_teste <- factor(ifelse(teste$Gender == 1, "Masculino", "Feminino"),
                       levels = c("Feminino", "Masculino"))
salvar_rds(genero_teste, "data/teste_genero.rds")
log_kv(paste0("Teste - ", levels(genero_teste)), as.vector(table(genero_teste)))

timer_end(t0, "Etapa 2")
