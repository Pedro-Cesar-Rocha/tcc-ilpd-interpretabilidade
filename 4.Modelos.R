# =============================================================================
# 4.Modelos.R
# Treina quatro classificadores com validacao cruzada repetida (5 folds x 3),
# otimizando hiperparametros pela AUC-ROC:
#   - Regressao Logistica (baseline linear e interpretavel)
#   - Random Forest (bagging de arvores)
#   - XGBoost (gradient boosting)
#   - LightGBM (gradient boosting, wrapper customizado para o caret)
# =============================================================================
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
}))

log_section("ETAPA 4 - TREINAMENTO DOS MODELOS")
t0 <- timer_start()
ensure_dirs(c("models", "results"))

treino <- readRDS("data/df_balanceado.rds")
log_info("Treino balanceado carregado: %d obs x %d preditores", nrow(treino), ncol(treino) - 1)
log_kv(paste0("Classe ", levels(treino$Dataset)), as.vector(table(treino$Dataset)))

# --- Configuracao da validacao cruzada --------------------------------------
log_subsection("Configuracao da validacao cruzada")
ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,   # ROC, Sens, Spec
  savePredictions = "final",
  verboseIter     = FALSE
)
log_kv(c("Metodo", "Folds", "Repeticoes", "Metrica de selecao", "Classe positiva"),
       c("repeatedcv", "5", "3", "ROC (AUC)", "doente"))

# Silencia o aviso deprecated `ntree_limit` do xgboost 1.7 (interno do caret)
xgboost::xgb.set.config(verbosity = 0)

treinar <- function(nome, ...) {
  log_subsection(sprintf("Treinando: %s", nome))
  ti <- timer_start()
  set.seed(PIPELINE_SEED)
  # glm gera avisos de separacao quase perfeita apos SMOTE; sao esperados e inofensivos
  modelo <- suppressWarnings(train(Dataset ~ ., data = treino, trControl = ctrl, metric = "ROC", ...))

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

modelo_lr  <- treinar("Regressao Logistica", method = "glm", family = "binomial")
modelo_rf  <- treinar("Random Forest",       method = "rf", tuneLength = 5, ntree = 500, importance = TRUE)
modelo_xgb <- treinar("XGBoost",             method = "xgbTree", tuneLength = 5, verbosity = 0)
modelo_lgb <- treinar("LightGBM",            method = lightgbm_caret)

# --- Coeficientes da regressao logistica (interpretacao direta) -------------
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

cv_df <- as.data.frame(resumo_cv) %>% rownames_to_column("Modelo")
salvar_csv(cv_df, "results/cv_auc_por_modelo.csv")

png("plots/cv_comparacao_modelos.png", width = 1000, height = 600, res = 130)
print(bwplot(rs, metric = "ROC", main = "AUC-ROC na validacao cruzada (5 folds x 3 repeticoes)"))
invisible(dev.off())
log_ok("Grafico salvo: plots/cv_comparacao_modelos.png")

# Teste pareado: diferencas de AUC entre modelos sao estatisticamente significativas?
log_subsection("Diferencas de AUC entre modelos (teste t pareado, Bonferroni)")
dif <- diff(rs, metric = "ROC")
log_table(summary(dif)$table$ROC, digits = 4)
log_info("Triangulo superior: diferenca media de AUC | triangulo inferior: p-valor ajustado.")

# --- Persistencia -------------------------------------------------------------
log_subsection("Salvando modelos")
salvar_rds(modelo_lr,  "models/modelo_lr.rds")
salvar_rds(modelo_rf,  "models/modelo_rf.rds")
salvar_rds(modelo_xgb, "models/modelo_xgb.rds")
salvar_rds(modelo_lgb, "models/modelo_lgb.rds")

timer_end(t0, "Etapa 4")
