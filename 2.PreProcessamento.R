# =============================================================================
# 2.PreProcessamento.R
# Encoding, transformacao log das variaveis assimetricas, divisao treino/teste
# ESTRATIFICADA e, SOMENTE DEPOIS, imputacao e normalizacao com parametros
# estimados no treino (evita qualquer vazamento de informacao do teste).
# =============================================================================
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
log_subsection("Remocao de duplicatas")
n_antes <- nrow(df)
df <- distinct(df)
log_info("Removidas %d linhas duplicadas (%d -> %d).", n_antes - nrow(df), n_antes, nrow(df))

# --- Encoding ----------------------------------------------------------------
log_subsection("Codificacao de variaveis categoricas")
df$Gender  <- ifelse(df$Gender == "Male", 1, 0)
log_info("Gender  -> Male = 1, Female = 0  (%d homens, %d mulheres)", sum(df$Gender == 1), sum(df$Gender == 0))
# Rotulos textuais sao exigidos pelo caret quando classProbs = TRUE
df$Dataset <- factor(ifelse(df$Dataset == 1, "doente", "saudavel"),
                     levels = c("saudavel", "doente"))
log_info("Dataset -> 'doente' (classe positiva) / 'saudavel'")
log_kv(paste0("Classe ", levels(df$Dataset)), as.vector(table(df$Dataset)))

# --- Transformacao log das variaveis assimetricas ---------------------------
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
log_subsection("Divisao treino / teste (80/20, estratificada pela classe)")
set.seed(PIPELINE_SEED)
idx    <- createDataPartition(df$Dataset, p = 0.8, list = FALSE)
treino <- df[idx, ]
teste  <- df[-idx, ]
log_info("Treino: %d obs  |  Teste: %d obs", nrow(treino), nrow(teste))
log_info("Proporcao 'doente' - treino: %.1f%%  |  teste: %.1f%%",
         100 * mean(treino$Dataset == "doente"), 100 * mean(teste$Dataset == "doente"))

# --- Imputacao (mediana estimada SOMENTE no treino) -------------------------
log_subsection("Imputacao de valores ausentes (mediana do treino aplicada a treino e teste)")
mediana_ag <- median(treino$AG_Ratio, na.rm = TRUE)
na_treino  <- sum(is.na(treino$AG_Ratio)); na_teste <- sum(is.na(teste$AG_Ratio))
treino$AG_Ratio[is.na(treino$AG_Ratio)] <- mediana_ag
teste$AG_Ratio[is.na(teste$AG_Ratio)]   <- mediana_ag
log_info("AG_Ratio: mediana do treino = %.2f | imputados: %d no treino, %d no teste.",
         mediana_ag, na_treino, na_teste)
log_ok("Total de NAs restantes: treino = %d | teste = %d", sum(is.na(treino)), sum(is.na(teste)))

# --- Normalizacao (ajustada SOMENTE no treino) ------------------------------
log_subsection("Normalizacao (center + scale) ajustada no conjunto de treino")
X_treino <- treino %>% select(-Dataset)
X_teste  <- teste  %>% select(-Dataset)

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

treino_proc <- cbind(X_treino_s, Dataset = treino$Dataset)
teste_proc  <- cbind(X_teste_s,  Dataset = teste$Dataset)

# --- Persistencia -------------------------------------------------------------
log_subsection("Salvando artefatos")
salvar_rds(treino_proc, "data/df_proc.rds")   # treino pre-processado (entrada do SMOTE e da CV)
salvar_rds(teste_proc,  "data/teste.rds")     # teste intocado pelo SMOTE
salvar_rds(preproc,     "data/preproc.rds")
# Genero em rotulo legivel, para a analise estratificada por sexo (etapas 5 e 6)
genero_teste <- factor(ifelse(teste$Gender == 1, "Masculino", "Feminino"),
                       levels = c("Feminino", "Masculino"))
salvar_rds(genero_teste, "data/teste_genero.rds")
log_kv(paste0("Teste - ", levels(genero_teste)), as.vector(table(genero_teste)))

timer_end(t0, "Etapa 2")
