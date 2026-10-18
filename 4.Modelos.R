# =============================================================================
# 4.Modelos.R
# Treina quatro classificadores com validacao cruzada repetida (5 folds x 3),
# aplicando SMOTE dentro de cada fold (apenas na particao de treino) e
# otimizando hiperparametros por Random Search com selecao pela AUC-ROC:
#   - Regressao Logistica (baseline linear e interpretavel)
#   - Random Forest (bagging de arvores)
#   - XGBoost (gradient boosting)
#   - LightGBM (gradient boosting, wrapper customizado para o caret)
# =============================================================================
# [Fundamentação Teórica - 2.4 Modelagem e Classificacao (sec:fund_modelos)] Etapa 3 da Figura 2.1.
# [Fundamentação Teórica - 2.5 Treinamento, Validacao e Avaliacao (sec:fund_validacao)] Etapa 4,
#   parte de treinamento e validacao (a avaliacao no teste esta em 5.Avaliacao.R).
# [Metodologia - 3.4 Modelos Preditivos (sec:modelos_preditivos)] e
# [Metodologia - 3.5 Treinamento e Validacao Cruzada (sec:treinamento_validacao)].
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(smotefamily)
}))

log_section("ETAPA 4 - TREINAMENTO DOS MODELOS")
t0 <- timer_start()
ensure_dirs(c("models", "results"))

treino <- readRDS("data/df_proc.rds")
log_info("Treino (distribuicao real) carregado: %d obs x %d preditores", nrow(treino), ncol(treino) - 1)
log_kv(paste0("Classe ", levels(treino$Diagnostico)), as.vector(table(treino$Diagnostico)))
log_info("O SMOTE sera aplicado dentro de cada fold da CV e no ajuste final (caret::trainControl(sampling)).")

# --- Configuracao da validacao cruzada --------------------------------------
# [Metodologia - 3.5.1 Validacao Cruzada Repetida (sec:validacao_cruzada)] 5 folds x 3
#   repeticoes, SMOTE dentro de cada fold (Metodologia - 3.3.4).
# [Metodologia - 3.5.2 Otimizacao de Hiperparametros (sec:otimizacao_hiperparametros)]
#   busca aleatoria com N_RANDOM = 30 sorteios por modelo (Tabela tab:espaco_busca).
# [Metodologia - 3.5.3 Selecao e Comparacao dos Modelos (sec:metrica_selecao)] selecao pela AUC-ROC.
log_subsection("Configuracao da validacao cruzada")
# [Fundamentação Teórica - 2.5.2 Hiperparametros e Sua Otimizacao (sec:fund_hiperparametros)]
#   busca aleatoria (Random Search) com numero fixo de combinacoes.
N_RANDOM <- 30   # combinacoes de hiperparametros sorteadas por modelo (Random Search)

# twoClassSummary toma o primeiro nivel ("saudavel") como evento; troca Sens/Spec
# para que ambas se refiram a classe positiva "doente" (a AUC nao muda).
# [Fundamentação Teórica - 2.5.3 Metricas de Desempenho (sec:fund_metricas)] sensibilidade e
#   especificidade referidas a classe positiva 'doente'.
resumo_doente <- function(data, lev = NULL, model = NULL) {
  out <- twoClassSummary(data, lev = lev, model = model)
  setNames(out[c("ROC", "Spec", "Sens")], c("ROC", "Sens", "Spec"))
}

# [Fundamentação Teórica - 2.5.1 Validacao Cruzada (sec:fund_cv)] k-fold repetida (k = 5, 3 repeticoes);
#   savePredictions guarda as previsoes fora da particao (out-of-fold).
# [Fundamentação Teórica - 2.5.4 Curva ROC e AUC (sec:fund_roc)] selecao pela AUC-ROC (metric = "ROC").
# [Fundamentação Teórica - 2.3.4 SMOTE] sampling = smote_caret: SMOTE dentro de cada fold.
ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = resumo_doente,     # ROC, Sens, Spec (classe positiva "doente")
  sampling        = smote_caret,       # SMOTE por fold (sem vazamento para a validacao)
  search          = "random",          # Random Search de hiperparametros
  savePredictions = "final",
  verboseIter     = FALSE
)
log_kv(c("Metodo", "Folds", "Repeticoes", "Balanceamento", "Busca de hiperparametros", "Metrica de selecao", "Classe positiva"),
       c("repeatedcv", "5", "3", "SMOTE dentro de cada fold", sprintf("Random Search (%d combinacoes)", N_RANDOM), "ROC (AUC)", "doente"))

# Silencia o aviso deprecated `ntree_limit` do xgboost 1.7 (interno do caret)
xgboost::xgb.set.config(verbosity = 0)

# [Fundamentação Teórica - 2.1] semente redefinida antes de cada train(): mesmas particoes para
#   todos os modelos. [Fundamentação Teórica - 2.4] sobreajuste e a razao da validacao cruzada.
# [Metodologia - 3.5.1 (sec:validacao_cruzada)] semente 42 redefinida antes de cada modelo,
#   para que todos usem as mesmas 15 particoes.
treinar <- function(nome, ...) {
  log_subsection(sprintf("Treinando: %s", nome))
  ti <- timer_start()
  set.seed(PIPELINE_SEED)
  # glm gera avisos de separacao quase perfeita apos SMOTE; sao esperados e inofensivos
  modelo <- suppressWarnings(train(Diagnostico ~ ., data = treino, trControl = ctrl, metric = "ROC", ...))

  melhor <- modelo$results[rownames(modelo$bestTune), , drop = FALSE]
  if (ncol(modelo$bestTune) > 0 && !identical(names(modelo$bestTune), "parameter")) {
    log_info("Combinacoes testadas: %d", nrow(modelo$results))
    log_info("Melhores hiperparametros:")
    log_kv(names(modelo$bestTune), as.character(unlist(modelo$bestTune)))
  } else {
    log_info("Modelo sem hiperparametros a otimizar.")
  }
  log_info("Desempenho na validacao cruzada (media +- desvio):")
  log_kv(c("AUC-ROC", "Sensibilidade", "Especificidade"),
         sprintf("%.4f +- %.4f", c(melhor$ROC, melhor$Sens, melhor$Spec),
                 c(melhor$ROCSD, melhor$SensSD, melhor$SpecSD)))
  timer_end(ti, nome)
  modelo
}

# [Fundamentação Teórica - 2.4.1 Regressao Logistica (sec:fund_lr)] Eq. 2.3 (eq:logit).
# [Fundamentação Teórica - 2.4.3 Random Forest (sec:fund_rf)] bagging, bootstrap, mtry, ntree = 500.
# [Fundamentação Teórica - 2.4.4 XGBoost (sec:fund_xgb)] gradient boosting, Eq. 2.4 (eq:boosting);
#   eta, gamma etc. sao os ajustes contra sobreajuste descritos no texto.
# [Fundamentação Teórica - 2.4.5 LightGBM (sec:fund_lgbm)] wrapper em UtilsPipeline.R.
# [Fundamentação Teórica - 2.4.2 Arvores de Decisao e Metodos de Ensemble (sec:fund_arvores)]
# [Metodologia - 3.4.1 Regressao Logistica (sec:regressao_logistica)] glm sem penalizacao.
# [Metodologia - 3.4.2 Random Forest (sec:random_forest)] ntree = 500; mtry otimizado.
# [Metodologia - 3.4.3 XGBoost (sec:xgboost)] sete hiperparametros otimizados (tab:espaco_busca).
# [Metodologia - 3.4.4 LightGBM (sec:lightgbm)] wrapper lightgbm_caret em UtilsPipeline.R.
modelo_lr  <- treinar("Regressao Logistica", method = "glm", family = "binomial")
modelo_rf  <- treinar("Random Forest",       method = "rf", tuneLength = N_RANDOM, ntree = 500)
modelo_xgb <- treinar("XGBoost",             method = "xgbTree", tuneLength = N_RANDOM, verbosity = 0)
modelo_lgb <- treinar("LightGBM",            method = lightgbm_caret, tuneLength = N_RANDOM)

# --- Coeficientes da regressao logistica (interpretacao direta) -------------
# [Metodologia - 3.4.1 Regressao Logistica (sec:regressao_logistica)] coeficientes, razoes de
#   chances e valores-p de Wald (tambem usados no ranking da Metodologia - 3.7.4).
# [Fundamentação Teórica - 2.4.1 Regressao Logistica (sec:fund_lr)] razao de chances OR = exp(beta)
#   e valor-p do teste de Wald; instabilidade sob multicolinearidade (2.2.4).
log_subsection("Coeficientes da Regressao Logistica (odds ratio)")
coefs <- summary(modelo_lr$finalModel)$coefficients
coef_df <- data.frame(Variavel   = rownames(coefs),
                      Coeficiente = round(coefs[, "Estimate"], 4),
                      Odds_Ratio  = round(exp(coefs[, "Estimate"]), 4),
                      p_valor     = round(coefs[, "Pr(>|z|)"], 4),
                      row.names = NULL) %>%
  mutate(Significativo = ifelse(p_valor < 0.05, "sim", "nao"))
log_table(coef_df)
log_info("OR > 1: aumento de 1 DP na variavel eleva a chance de 'doente'; OR < 1: reduz.")
salvar_csv(coef_df, "results/coeficientes_regressao_logistica.csv")

# --- Comparacao dos modelos na validacao cruzada -----------------------------
# [Metodologia - 3.5.3 Selecao e Comparacao dos Modelos (sec:metrica_selecao, sec:comparacao_cv)]
#   AUC-ROC media e distribuicao nas 15 particoes (as mesmas para os quatro modelos).
log_subsection("Comparacao dos modelos na validacao cruzada (resamples)")
modelos <- list(
  "Regressao Logistica" = modelo_lr,
  "Random Forest"       = modelo_rf,
  "XGBoost"             = modelo_xgb,
  "LightGBM"            = modelo_lgb
)
rs <- resamples(modelos)
resumo_cv <- summary(rs)$statistics$ROC[, c("Min.", "Mean", "Max.")]
log_info("AUC-ROC por modelo ao longo dos %d folds:", nrow(rs$values))
log_table(resumo_cv, digits = 4)
resumo_sens <- summary(rs)$statistics$Sens[, c("Min.", "Mean", "Max.")]
log_info("Sensibilidade (Recall, limiar 0.5) por modelo na CV:")
log_table(resumo_sens, digits = 4)

cv_df <- as.data.frame(resumo_cv) %>% rownames_to_column("Modelo") %>%
  rename(AUC_Min = Min., AUC_Media = Mean, AUC_Max = Max.) %>%
  mutate(Sens_Media = as.data.frame(resumo_sens)$Mean,
         Spec_Media = as.data.frame(summary(rs)$statistics$Spec)$Mean)
salvar_csv(cv_df, "results/cv_auc_por_modelo.csv")

# Boxplot da AUC-ROC nas 15 particoes (5 folds x 3 repeticoes), em ggplot2 para
# manter o mesmo estilo dos demais graficos. Pontos = particoes; losango = media.
cv_long <- rs$values %>%
  select(ends_with("~ROC")) %>%
  pivot_longer(everything(), names_to = "Modelo", values_to = "AUC") %>%
  mutate(Modelo = rotulo_modelo(sub("~ROC$", "", Modelo)),
         Modelo = reorder(Modelo, AUC, FUN = mean))
p_cv <- ggplot(cv_long, aes(AUC, Modelo)) +
  geom_boxplot(width = 0.5, outlier.shape = NA, fill = "grey92", linewidth = 0.3) +
  geom_point(position = position_jitter(height = 0.12, width = 0, seed = PIPELINE_SEED),
             size = 1, alpha = 0.6) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 2.2, fill = "white") +
  labs(x = "AUC-ROC na valida\u00e7\u00e3o cruzada (15 parti\u00e7\u00f5es)", y = NULL) +
  tema_tcc()
salvar_plot(p_cv, "plots/cv_comparacao_modelos.png", width = 13.5, height = 7)

# --- Persistencia -------------------------------------------------------------
log_subsection("Salvando modelos")
salvar_rds(modelo_lr,  "models/modelo_lr.rds")
salvar_rds(modelo_rf,  "models/modelo_rf.rds")
salvar_rds(modelo_xgb, "models/modelo_xgb.rds")
salvar_rds(modelo_lgb, "models/modelo_lgb.rds")

timer_end(t0, "Etapa 4")
