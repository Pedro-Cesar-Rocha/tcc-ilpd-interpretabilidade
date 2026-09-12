# =============================================================================
# 5.Avaliacao.R
# Avalia os modelos no conjunto de teste (distribuicao real, sem SMOTE):
# matriz de confusao, metricas, IC 95% da AUC (DeLong), curvas ROC e
# Precision-Recall, e analise do limiar de decisao.
# =============================================================================
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(pROC)
}))

log_section("ETAPA 5 - AVALIACAO NO CONJUNTO DE TESTE")
t0 <- timer_start()

modelos <- list(
  "Regressao Logistica" = readRDS("models/modelo_lr.rds"),
  "Random Forest"       = readRDS("models/modelo_rf.rds"),
  "XGBoost"             = readRDS("models/modelo_xgb.rds"),
  "LightGBM"            = readRDS("models/modelo_lgb.rds")
)
teste <- readRDS("data/teste.rds")
xgboost::xgb.set.config(verbosity = 0)

log_info("Conjunto de teste: %d observacoes (nunca vistas no treino nem no SMOTE)", nrow(teste))
log_kv(paste0("Classe ", levels(teste$Dataset)),
       sprintf("%d (%.1f%%)", table(teste$Dataset), 100 * prop.table(table(teste$Dataset))))

# --- Avaliacao individual -----------------------------------------------------
avaliar <- function(modelo, nome) {
  log_subsection(sprintf("Modelo: %s", nome))

  prob   <- predict(modelo, teste, type = "prob")[, "doente"]
  classe <- predict(modelo, teste)
  cm     <- confusionMatrix(classe, teste$Dataset, positive = "doente")
  roc_o  <- roc(teste$Dataset, prob, levels = c("saudavel", "doente"), direction = "<", quiet = TRUE)
  ic     <- ci.auc(roc_o, method = "delong")

  log_info("Matriz de confusao (linhas = predito, colunas = real):")
  log_table(cm$table)

  tn <- cm$table["saudavel", "saudavel"]; fp <- cm$table["doente", "saudavel"]
  fn <- cm$table["saudavel", "doente"];   tp <- cm$table["doente", "doente"]
  log_info("VP = %d | FP = %d | FN = %d | VN = %d", tp, fp, fn, tn)
  log_info("Falsos negativos (doentes classificados como saudaveis) sao o erro mais critico no contexto clinico.")

  metricas <- c(
    Acuracia       = unname(cm$overall["Accuracy"]),
    Acuracia_Bal   = unname(cm$byClass["Balanced Accuracy"]),
    Sensibilidade  = unname(cm$byClass["Sensitivity"]),
    Especificidade = unname(cm$byClass["Specificity"]),
    Precisao       = unname(cm$byClass["Precision"]),
    F1             = unname(cm$byClass["F1"]),
    Kappa          = unname(cm$overall["Kappa"]),
    AUC            = as.numeric(auc(roc_o))
  )
  log_info("Metricas (limiar = 0.5):")
  log_kv(names(metricas), metricas, digits = 4)
  log_info("IC 95%% da AUC (DeLong): [%.4f ; %.4f]", ic[1], ic[3])

  # Limiar otimo pelo indice de Youden (Sens + Spec - 1)
  yj <- coords(roc_o, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"))
  log_info("Limiar otimo (Youden): %.3f -> Sens = %.3f | Spec = %.3f",
           yj$threshold[1], yj$sensitivity[1], yj$specificity[1])

  list(nome = nome, prob = prob, roc = roc_o,
       metricas = c(Modelo = nome, round(metricas, 4), AUC_IC_inf = round(ic[1], 4), AUC_IC_sup = round(ic[3], 4),
                    Limiar_Youden = round(yj$threshold[1], 3)))
}

resultados <- imap(modelos, ~avaliar(.x, .y))

# --- Tabela comparativa -------------------------------------------------------
log_subsection("Tabela comparativa no conjunto de teste")
tabela <- map_dfr(resultados, ~as.data.frame(t(.x$metricas), stringsAsFactors = FALSE)) %>%
  mutate(across(-Modelo, as.numeric)) %>%
  arrange(desc(AUC))
log_table(tabela %>% select(Modelo, AUC, Sensibilidade, Especificidade, F1, Acuracia_Bal, Kappa), digits = 4)
salvar_csv(tabela, "results/metricas_teste.csv")

melhor <- tabela$Modelo[1]
log_ok("Melhor modelo pela AUC no teste: %s (AUC = %.4f)", melhor, tabela$AUC[1])

# --- Comparacao estatistica das curvas ROC (DeLong) --------------------------
log_subsection("Comparacao pareada das AUCs (teste de DeLong)")
nomes <- names(resultados)
pares <- combn(nomes, 2, simplify = FALSE)
delong <- map_dfr(pares, function(p) {
  tt <- roc.test(resultados[[p[1]]]$roc, resultados[[p[2]]]$roc, method = "delong")
  data.frame(Modelo_A = p[1], Modelo_B = p[2],
             AUC_A = round(as.numeric(auc(resultados[[p[1]]]$roc)), 4),
             AUC_B = round(as.numeric(auc(resultados[[p[2]]]$roc)), 4),
             p_valor = round(tt$p.value, 4))
})
delong$Significativo_5pct <- ifelse(delong$p_valor < 0.05, "sim", "nao")
log_table(delong)
salvar_csv(delong, "results/delong_pareado.csv")

# --- Curvas ROC ---------------------------------------------------------------
log_subsection("Gerando graficos")
roc_df <- map_dfr(resultados, function(r) {
  data.frame(Modelo = sprintf("%s (AUC = %.3f)", r$nome, as.numeric(auc(r$roc))),
             FPR = 1 - r$roc$specificities, TPR = r$roc$sensitivities)
})
p_roc <- ggplot(roc_df, aes(FPR, TPR, colour = Modelo)) +
  geom_abline(linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.9) +
  coord_equal() +
  labs(title = "Curvas ROC no conjunto de teste",
       x = "1 - Especificidade (FPR)", y = "Sensibilidade (TPR)") +
  theme_minimal() + theme(legend.position = "bottom", legend.text = element_text(size = 8)) +
  guides(colour = guide_legend(ncol = 2))
salvar_plot(p_roc, "plots/roc_teste.png", width = 7, height = 7.5)

# --- Curvas Precision-Recall --------------------------------------------------
pr_df <- map_dfr(resultados, function(r) {
  co <- coords(r$roc, "all", ret = c("recall", "precision"))
  data.frame(Modelo = r$nome, Recall = co$recall, Precision = co$precision) %>% filter(!is.na(Precision))
})
p_pr <- ggplot(pr_df, aes(Recall, Precision, colour = Modelo)) +
  geom_hline(yintercept = mean(teste$Dataset == "doente"), linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.9) +
  labs(title = "Curvas Precision-Recall no conjunto de teste",
       subtitle = "Linha tracejada: prevalencia da classe positiva") +
  theme_minimal()
salvar_plot(p_pr, "plots/precision_recall_teste.png", width = 7, height = 5)

# --- Grafico comparativo de metricas -----------------------------------------
p_met <- tabela %>%
  select(Modelo, AUC, Sensibilidade, Especificidade, F1, Acuracia_Bal) %>%
  pivot_longer(-Modelo, names_to = "Metrica", values_to = "Valor") %>%
  ggplot(aes(Metrica, Valor, fill = Modelo)) +
  geom_col(position = position_dodge(0.85), width = 0.8) +
  ylim(0, 1.05) +
  labs(title = "Metricas no conjunto de teste por modelo", x = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
salvar_plot(p_met, "plots/metricas_teste.png", width = 11, height = 6)

# --- Sensibilidade ao limiar (melhor modelo) ---------------------------------
log_subsection(sprintf("Analise de limiar de decisao - %s", melhor))
r_melhor <- resultados[[melhor]]
limiares <- seq(0.1, 0.9, by = 0.1)
sens_lim <- map_dfr(limiares, function(l) {
  pred <- factor(ifelse(r_melhor$prob >= l, "doente", "saudavel"), levels = c("saudavel", "doente"))
  cm <- confusionMatrix(pred, teste$Dataset, positive = "doente")
  data.frame(Limiar = l,
             Sensibilidade = round(cm$byClass["Sensitivity"], 3),
             Especificidade = round(cm$byClass["Specificity"], 3),
             F1 = round(cm$byClass["F1"], 3),
             row.names = NULL)
})
log_table(sens_lim, digits = 3)
log_info("Em triagem clinica, limiares menores priorizam sensibilidade (menos falsos negativos).")
salvar_csv(sens_lim, "results/analise_limiar_melhor_modelo.csv")

timer_end(t0, "Etapa 5")
