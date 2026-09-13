# =============================================================================
# 5.Avaliacao.R
# Avalia os modelos no conjunto de teste (distribuicao real, sem SMOTE):
# matriz de confusao, metricas, IC 95% da AUC (DeLong), curvas ROC e
# Precision-Recall, analise do limiar de decisao com foco em Recall (para
# todos os modelos) e analise de desempenho estratificada por sexo (vies).
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
teste  <- readRDS("data/teste.rds")
genero <- readRDS("data/teste_genero.rds")
xgboost::xgb.set.config(verbosity = 0)

RECALL_ALVO <- 0.90   # sensibilidade minima desejada em cenario de triagem clinica

log_info("Conjunto de teste: %d observacoes (nunca vistas no treino nem no SMOTE)", nrow(teste))
log_kv(paste0("Classe ", levels(teste$Dataset)),
       sprintf("%d (%.1f%%)", table(teste$Dataset), 100 * prop.table(table(teste$Dataset))))
log_kv(paste0("Sexo ", levels(genero)),
       sprintf("%d (%.1f%%)", table(genero), 100 * prop.table(table(genero))))

# Menor limiar que garante sensibilidade >= alvo (prioriza reduzir falsos negativos)
limiar_recall <- function(roc_o, alvo) {
  co <- coords(roc_o, "all", ret = c("threshold", "sensitivity", "specificity", "precision"))
  co <- co[is.finite(co$threshold) & co$sensitivity >= alvo, ]
  co[which.max(co$threshold), ]
}

metricas_no_limiar <- function(prob, real, limiar) {
  pred <- factor(ifelse(prob >= limiar, "doente", "saudavel"), levels = c("saudavel", "doente"))
  cm <- confusionMatrix(pred, real, positive = "doente")
  c(Sensibilidade = unname(cm$byClass["Sensitivity"]), Especificidade = unname(cm$byClass["Specificity"]),
    Precisao = unname(cm$byClass["Precision"]), F1 = unname(cm$byClass["F1"]),
    Acuracia_Bal = unname(cm$byClass["Balanced Accuracy"]),
    FN = unname(cm$table["saudavel", "doente"]), FP = unname(cm$table["doente", "saudavel"]))
}

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

  # Limiar clinico: menor sacrificio de especificidade que garante Recall >= alvo
  lr <- limiar_recall(roc_o, RECALL_ALVO)
  log_info("Limiar clinico (Recall >= %.0f%%): %.3f -> Sens = %.3f | Spec = %.3f | Precisao = %.3f",
           100 * RECALL_ALVO, lr$threshold, lr$sensitivity, lr$specificity, lr$precision)

  list(nome = nome, prob = prob, roc = roc_o,
       metricas = c(Modelo = nome, round(metricas, 4), AUC_IC_inf = round(ic[1], 4), AUC_IC_sup = round(ic[3], 4),
                    Limiar_Youden = round(yj$threshold[1], 3),
                    Limiar_Recall90 = round(lr$threshold, 3),
                    Spec_Recall90 = round(lr$specificity, 4),
                    Precisao_Recall90 = round(lr$precision, 4)))
}

resultados <- imap(modelos, ~avaliar(.x, .y))

# --- Tabela comparativa -------------------------------------------------------
log_subsection("Tabela comparativa no conjunto de teste")
tabela <- map_dfr(resultados, ~as.data.frame(t(.x$metricas), stringsAsFactors = FALSE)) %>%
  mutate(across(-Modelo, as.numeric)) %>%
  arrange(desc(AUC))
log_table(tabela %>% select(Modelo, AUC, Sensibilidade, Especificidade, F1, Acuracia_Bal, Kappa), digits = 4)
log_info("Foco em Recall: especificidade obtida por cada modelo quando o limiar garante Sens >= %.0f%%:", 100 * RECALL_ALVO)
log_table(tabela %>% select(Modelo, Limiar_Recall90, Spec_Recall90, Precisao_Recall90), digits = 4)
salvar_csv(tabela, "results/metricas_teste.csv")

melhor <- tabela$Modelo[1]
log_ok("Melhor modelo pela AUC no teste: %s (AUC = %.4f)", melhor, tabela$AUC[1])
salvar_rds(melhor, "results/melhor_modelo.rds")

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

# --- Sensibilidade ao limiar (todos os modelos) -------------------------------
log_subsection("Analise de limiar de decisao - todos os modelos")
limiares <- seq(0.1, 0.9, by = 0.1)
sens_lim <- map_dfr(resultados, function(r) {
  map_dfr(limiares, function(l) {
    m <- metricas_no_limiar(r$prob, teste$Dataset, l)
    data.frame(Modelo = r$nome, Limiar = l, Sensibilidade = round(m["Sensibilidade"], 3),
               Especificidade = round(m["Especificidade"], 3), F1 = round(m["F1"], 3),
               FN = m["FN"], FP = m["FP"], row.names = NULL)
  })
})
for (nm in names(resultados)) {
  log_info("%s:", nm)
  log_table(sens_lim %>% filter(Modelo == nm) %>% select(-Modelo), digits = 3)
}
log_info("Em triagem clinica, limiares menores priorizam sensibilidade (menos falsos negativos).")
salvar_csv(sens_lim, "results/analise_limiar_modelos.csv")

p_lim <- sens_lim %>%
  pivot_longer(c(Sensibilidade, Especificidade), names_to = "Metrica", values_to = "Valor") %>%
  ggplot(aes(Limiar, Valor, colour = Metrica)) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.5) +
  geom_vline(xintercept = 0.5, linetype = "dotted", colour = "grey50") +
  facet_wrap(~Modelo, ncol = 2) +
  labs(title = "Trade-off Sensibilidade x Especificidade por limiar de decisao",
       subtitle = "Conjunto de teste; linha pontilhada = limiar padrao 0.5", x = "Limiar", y = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
salvar_plot(p_lim, "plots/limiar_sens_spec_modelos.png", width = 9, height = 6)

# --- Analise estratificada por sexo (vies diagnostico) ------------------------
log_subsection("Desempenho estratificado por sexo (Straw & Wu, 2022)")
log_info("Prevalencia de 'doente' por sexo no teste:")
log_table(prop.table(table(Sexo = genero, Classe = teste$Dataset), 1), digits = 3)

por_sexo <- map_dfr(resultados, function(r) {
  map_dfr(levels(genero), function(g) {
    idx  <- genero == g
    real <- teste$Dataset[idx]; prob <- r$prob[idx]
    roc_g <- roc(real, prob, levels = c("saudavel", "doente"), direction = "<", quiet = TRUE)
    ic_g  <- ci.auc(roc_g, method = "delong")
    m05   <- metricas_no_limiar(prob, real, 0.5)
    m90   <- metricas_no_limiar(prob, real, tabela$Limiar_Recall90[tabela$Modelo == r$nome])
    data.frame(Modelo = r$nome, Sexo = g, n = sum(idx), n_doentes = sum(real == "doente"),
               AUC = round(as.numeric(auc(roc_g)), 4),
               AUC_IC_inf = round(ic_g[1], 4), AUC_IC_sup = round(ic_g[3], 4),
               Sensibilidade = round(m05["Sensibilidade"], 4),
               Especificidade = round(m05["Especificidade"], 4),
               F1 = round(m05["F1"], 4),
               Taxa_FN = round(1 - m05["Sensibilidade"], 4),
               Sens_LimiarRecall90 = round(m90["Sensibilidade"], 4),
               Spec_LimiarRecall90 = round(m90["Especificidade"], 4),
               row.names = NULL)
  })
})
log_table(por_sexo %>% select(Modelo, Sexo, n, n_doentes, AUC, Sensibilidade, Especificidade, Taxa_FN), digits = 4)
salvar_csv(por_sexo, "results/metricas_por_sexo.csv")

# Diferenca de AUC entre sexos (DeLong, amostras independentes) e gap de metricas
gap_sexo <- map_dfr(resultados, function(r) {
  idx_f <- genero == "Feminino"; idx_m <- genero == "Masculino"
  roc_f <- roc(teste$Dataset[idx_f], r$prob[idx_f], levels = c("saudavel", "doente"), direction = "<", quiet = TRUE)
  roc_m <- roc(teste$Dataset[idx_m], r$prob[idx_m], levels = c("saudavel", "doente"), direction = "<", quiet = TRUE)
  tt <- roc.test(roc_f, roc_m, method = "delong", paired = FALSE)
  ps <- por_sexo %>% filter(Modelo == r$nome)
  data.frame(Modelo = r$nome,
             AUC_Feminino = round(as.numeric(auc(roc_f)), 4), AUC_Masculino = round(as.numeric(auc(roc_m)), 4),
             Dif_AUC = round(as.numeric(auc(roc_m)) - as.numeric(auc(roc_f)), 4),
             p_valor_DeLong = round(tt$p.value, 4),
             Dif_Sensibilidade = round(diff(ps$Sensibilidade), 4),   # Masculino - Feminino
             Dif_Especificidade = round(diff(ps$Especificidade), 4),
             row.names = NULL)
})
gap_sexo$Significativo_5pct <- ifelse(gap_sexo$p_valor_DeLong < 0.05, "sim", "nao")
log_table(gap_sexo, digits = 4)
log_info("Dif_* = Masculino - Feminino. Diferencas sistematicas de Sensibilidade indicam risco desigual de falso negativo.")
log_warn("Amostras por sexo sao pequenas (especialmente feminino); interpretar gaps com cautela (ver ICs).")
salvar_csv(gap_sexo, "results/gap_desempenho_sexo.csv")

p_sexo <- por_sexo %>%
  select(Modelo, Sexo, AUC, Sensibilidade, Especificidade) %>%
  pivot_longer(c(AUC, Sensibilidade, Especificidade), names_to = "Metrica", values_to = "Valor") %>%
  ggplot(aes(Modelo, Valor, fill = Sexo)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  facet_wrap(~Metrica, ncol = 1) +
  ylim(0, 1) +
  labs(title = "Desempenho no teste estratificado por sexo", x = NULL, y = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
salvar_plot(p_sexo, "plots/metricas_por_sexo.png", width = 8, height = 8)

p_roc_sexo <- map_dfr(resultados, function(r) {
  map_dfr(levels(genero), function(g) {
    idx <- genero == g
    roc_g <- roc(teste$Dataset[idx], r$prob[idx], levels = c("saudavel", "doente"), direction = "<", quiet = TRUE)
    data.frame(Modelo = r$nome, Sexo = sprintf("%s (AUC = %.3f)", g, as.numeric(auc(roc_g))),
               FPR = 1 - roc_g$specificities, TPR = roc_g$sensitivities)
  })
}) %>%
  ggplot(aes(FPR, TPR, colour = Sexo)) +
  geom_abline(linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.9) +
  facet_wrap(~Modelo, ncol = 2) + coord_equal() +
  labs(title = "Curvas ROC por sexo no conjunto de teste", x = "1 - Especificidade", y = "Sensibilidade") +
  theme_minimal() + theme(legend.position = "bottom", legend.text = element_text(size = 7)) +
  guides(colour = guide_legend(ncol = 2))
salvar_plot(p_roc_sexo, "plots/roc_por_sexo.png", width = 8, height = 9)

timer_end(t0, "Etapa 5")
