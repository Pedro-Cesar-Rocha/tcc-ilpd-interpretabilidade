# =============================================================================
# 5.Avaliacao.R
# Avalia os modelos no conjunto de teste (distribuicao real, sem SMOTE):
# matriz de confusao, metricas, IC 95% da AUC (DeLong), curvas ROC, limiar
# clinico (sensibilidade >= 90%) escolhido nas predicoes out-of-fold da
# validacao cruzada e apenas aplicado ao teste, comparacao pareada das AUCs
# (DeLong com correcao de Bonferroni) e desempenho estratificado por sexo.
# =============================================================================
# [Fundamentação Teórica - 2.5 Treinamento, Validacao e Avaliacao (sec:fund_validacao)] Etapa 4 da
# Figura 2.1, parte de avaliacao no conjunto de teste (2.5.3 a 2.5.5), e
# [Fundamentação Teórica - 2.7 Vies Algoritmico (sec:fund_vies)] na analise por sexo.
# [Metodologia - 3.6 Avaliacao de Desempenho (sec:avaliacao_desempenho)] este script
#   implementa as secoes 3.6.1 a 3.6.4 da Metodologia.
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

# [Fundamentação Teórica - 2.5.5 Limiar de Decisao (sec:fund_limiar)] meta minima de sensibilidade
#   (triagem clinica: falso negativo e o erro mais custoso).
# [Metodologia - 3.6.2 Analise do Limiar de Decisao (sec:limiar_decisao)] meta de 90% de sensibilidade.
RECALL_ALVO <- 0.90   # sensibilidade minima desejada em cenario de triagem clinica

log_info("Conjunto de teste: %d observacoes (nunca vistas no treino nem no SMOTE)", nrow(teste))
log_kv(paste0("Classe ", levels(teste$Diagnostico)),
       sprintf("%d (%.1f%%)", table(teste$Diagnostico), 100 * prop.table(table(teste$Diagnostico))))
log_kv(paste0("Sexo ", levels(genero)),
       sprintf("%d (%.1f%%)", table(genero), 100 * prop.table(table(genero))))

# Maior limiar que ainda garante sensibilidade >= alvo (menor custo em especificidade)
limiar_recall <- function(roc_o, alvo) {
  co <- coords(roc_o, "all", ret = c("threshold", "sensitivity", "specificity", "precision"))
  co <- co[is.finite(co$threshold) & co$sensitivity >= alvo, ]
  co[which.max(co$threshold), ]
}

metricas_no_limiar <- function(prob, real, limiar) {
  pred <- factor(ifelse(prob >= limiar, "doente", "saudavel"), levels = c("saudavel", "doente"))
  cm <- confusionMatrix(pred, real, positive = "doente")
  c(Sensibilidade = unname(cm$byClass["Sensitivity"]), Especificidade = unname(cm$byClass["Specificity"]),
    Precisao = unname(cm$byClass["Precision"]),
    FN = unname(cm$table["saudavel", "doente"]), FP = unname(cm$table["doente", "saudavel"]))
}

# Curva ROC das predicoes out-of-fold da validacao cruzada (somente treino).
# Com savePredictions = "final", o caret guarda, para os hiperparametros finais,
# a probabilidade prevista para cada paciente do treino pelo modelo do fold em
# que ele ficou de fora (uma vez por repeticao: 5 folds x 3 repeticoes).
# [Fundamentação Teórica - 2.5.1 Validacao Cruzada (sec:fund_cv)] previsoes fora da particao.
# [Fundamentação Teórica - 2.5.5] limiar escolhido fora das previsoes de avaliacao (Leeflang et al.).
# [Metodologia - 3.6.2 (sec:limiar_decisao)] limiares escolhidos nas previsoes out-of-fold
#   da validacao cruzada e apenas aplicados ao teste.
roc_oof <- function(modelo) {
  oof <- modelo$pred
  roc(oof$obs, oof$doente, levels = c("saudavel", "doente"), direction = "<", quiet = TRUE)
}

# --- Avaliacao individual -----------------------------------------------------
# [Metodologia - 3.6.1 Metricas e Curvas de Desempenho (sec:metricas)] matriz de confusao,
#   metricas no limiar 0,5, AUC-ROC e IC 95% (DeLong) de cada modelo;
# [Metodologia - 3.6.2 (sec:limiar_decisao)] limiar clinico.
avaliar <- function(modelo, nome) {
  log_subsection(sprintf("Modelo: %s", nome))

  prob   <- predict(modelo, teste, type = "prob")[, "doente"]
  classe <- predict(modelo, teste)
  # [Fundamentação Teórica - 2.5.3 Metricas de Desempenho (sec:fund_matriz)] matriz de confusao
  #   (Tabela tab:matriz_confusao): VP, FN, VN, FP.
  cm     <- confusionMatrix(classe, teste$Diagnostico, positive = "doente")
  # [Fundamentação Teórica - 2.5.4 Curva ROC, AUC e Teste de DeLong (sec:fund_roc, sec:fund_delong)]
  #   curva ROC, AUC-ROC e intervalo de confianca de 95% pelo metodo de DeLong.
  roc_o  <- roc(teste$Diagnostico, prob, levels = c("saudavel", "doente"), direction = "<", quiet = TRUE)
  ic     <- ci.auc(roc_o, method = "delong")

  log_info("Matriz de confusao (linhas = predito, colunas = real):")
  log_table(cm$table)

  tn <- cm$table["saudavel", "saudavel"]; fp <- cm$table["doente", "saudavel"]
  fn <- cm$table["saudavel", "doente"];   tp <- cm$table["doente", "doente"]
  log_info("VP = %d | FP = %d | FN = %d | VN = %d", tp, fp, fn, tn)
  log_info("Falsos negativos (doentes classificados como saudaveis) sao o erro mais critico no contexto clinico.")

  # [Fundamentação Teórica - 2.5.3] acuracia, acuracia balanceada, sensibilidade, especificidade,
  #   e precisao (mesmas definicoes da Tabela tab:matriz_confusao).
  metricas <- c(
    Acuracia       = unname(cm$overall["Accuracy"]),
    Sensibilidade  = unname(cm$byClass["Sensitivity"]),
    Especificidade = unname(cm$byClass["Specificity"]),
    Precisao       = unname(cm$byClass["Precision"]),
    AUC            = as.numeric(auc(roc_o))
  )
  log_info("Metricas (limiar = 0.5):")
  log_kv(names(metricas), metricas, digits = 4)
  log_info("IC 95%% da AUC (DeLong): [%.4f ; %.4f]", ic[1], ic[3])

  # Limiares escolhidos SOMENTE com dados de treino (predicoes out-of-fold da CV)
  # e apenas aplicados ao teste; escolhe-los nas proprias previsoes do teste
  # tornaria otimistas as metricas reportadas nesses limiares.
  roc_cv <- roc_oof(modelo)
  log_info("ROC out-of-fold da CV: %d predicoes | AUC = %.4f", length(roc_cv$response), as.numeric(auc(roc_cv)))

  # Limiar clinico: menor sacrificio de especificidade que garante Recall >= alvo na CV
  # [Fundamentação Teórica - 2.5.5] maior limiar que atinge a meta minima de sensibilidade.
  lr   <- limiar_recall(roc_cv, RECALL_ALVO)
  m_lr <- metricas_no_limiar(prob, teste$Diagnostico, lr$threshold)
  log_info("Limiar clinico (Recall >= %.0f%% na CV): %.3f -> CV: Sens = %.3f | Spec = %.3f || teste: Sens = %.3f | Spec = %.3f | Precisao = %.3f",
           100 * RECALL_ALVO, lr$threshold, lr$sensitivity, lr$specificity,
           m_lr["Sensibilidade"], m_lr["Especificidade"], m_lr["Precisao"])

  list(nome = nome, prob = prob, roc = roc_o, limiar_recall = lr$threshold,
       metricas = c(Modelo = nome, round(metricas, 4), AUC_IC_inf = round(ic[1], 4), AUC_IC_sup = round(ic[3], 4),
                    Limiar_Recall90 = round(lr$threshold, 3),
                    Sens_Recall90 = round(m_lr[["Sensibilidade"]], 4),
                    Spec_Recall90 = round(m_lr[["Especificidade"]], 4),
                    Precisao_Recall90 = round(m_lr[["Precisao"]], 4)))
}

resultados <- imap(modelos, ~avaliar(.x, .y))

# --- Tabela comparativa -------------------------------------------------------
# [Metodologia - 3.6.1 Metricas e Curvas de Desempenho (sec:metricas)] tabela de metricas no teste.
log_subsection("Tabela comparativa no conjunto de teste")
tabela <- map_dfr(resultados, ~as.data.frame(t(.x$metricas), stringsAsFactors = FALSE)) %>%
  mutate(across(-Modelo, as.numeric)) %>%
  arrange(desc(AUC))
log_table(tabela %>% select(Modelo, AUC, Acuracia, Sensibilidade, Especificidade, Precisao), digits = 4)
log_info("Foco em Recall: desempenho no teste com o limiar que garante Sens >= %.0f%% na CV:", 100 * RECALL_ALVO)
log_table(tabela %>% select(Modelo, Limiar_Recall90, Sens_Recall90, Spec_Recall90, Precisao_Recall90), digits = 4)
salvar_csv(tabela, "results/metricas_teste.csv")

melhor <- tabela$Modelo[1]
log_ok("Melhor modelo pela AUC no teste: %s (AUC = %.4f)", melhor, tabela$AUC[1])
salvar_rds(melhor, "results/melhor_modelo.rds")

# --- Comparacao estatistica das curvas ROC (DeLong) --------------------------
# [Metodologia - 3.6.3 Comparacao Estatistica das AUCs (sec:comparacao_auc)] DeLong pareado
#   nos seis pares de modelos, com correcao de Bonferroni.
# Seis comparacoes simultaneas (todos os pares de modelos): os p-valores sao
# ajustados por Bonferroni para controlar a taxa de erro do tipo I da familia.
# [Fundamentação Teórica - 2.5.4 (sec:fund_delong)] teste de DeLong para curvas ROC nos mesmos pacientes;
# [Fundamentação Teórica - 2.2.5 (sec:fund_bonferroni)] correcao de Bonferroni.
log_subsection("Comparacao pareada das AUCs (teste de DeLong, correcao de Bonferroni)")
nomes <- names(resultados)
pares <- combn(nomes, 2, simplify = FALSE)
delong <- map_dfr(pares, function(p) {
  tt <- roc.test(resultados[[p[1]]]$roc, resultados[[p[2]]]$roc, method = "delong")
  data.frame(Modelo_A = p[1], Modelo_B = p[2],
             AUC_A = round(as.numeric(auc(resultados[[p[1]]]$roc)), 4),
             AUC_B = round(as.numeric(auc(resultados[[p[2]]]$roc)), 4),
             p_valor = tt$p.value)
})
delong$p_ajustado_Bonferroni <- round(p.adjust(delong$p_valor, method = "bonferroni"), 4)
delong$p_valor <- round(delong$p_valor, 4)
delong$Significativo_5pct <- ifelse(delong$p_ajustado_Bonferroni < 0.05, "sim", "nao")
log_table(delong)
log_info("Significancia avaliada no p-valor ajustado (Bonferroni, %d comparacoes).", nrow(delong))
salvar_csv(delong, "results/delong_pareado.csv")

# --- Curvas ROC ---------------------------------------------------------------
# [Metodologia - 3.6.1 (sec:metricas)] curvas ROC no teste.
log_subsection("Gerando graficos")
roc_df <- map_dfr(resultados, function(r) {
  data.frame(Modelo = sprintf("%s (AUC = %.3f)", rotulo_modelo(r$nome), as.numeric(auc(r$roc))),
             FPR = 1 - r$roc$specificities, TPR = r$roc$sensitivities)
})
p_roc <- ggplot(roc_df, aes(FPR, TPR, colour = Modelo)) +
  geom_abline(linetype = "dashed", colour = "grey60") +
  geom_line(linewidth = 0.9) +
  coord_equal() +
  scale_x_continuous(breaks = seq(0, 1, 0.25)) + scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  labs(x = "1 - Especificidade", y = "Sensibilidade", colour = NULL) +
  tema_tcc() + theme(legend.text = element_text(size = 7)) +
  guides(colour = guide_legend(ncol = 2))
salvar_plot(p_roc, "plots/roc_teste.png", width = 11, height = 12.5)

# --- Grafico comparativo de metricas -----------------------------------------
# Pontos em vez de barras: as metricas ficam entre ~0,5 e ~0,8, e barras a partir
# do zero escondem as diferencas; com pontos, o eixo pode comecar perto do menor valor.
met_long <- tabela %>%
  select(Modelo, AUC, Acuracia, Sensibilidade, Especificidade, Precisao) %>%
  mutate(across(-Modelo, as.numeric), Modelo = rotulo_modelo(Modelo)) %>%
  pivot_longer(-Modelo, names_to = "Metrica", values_to = "Valor") %>%
  mutate(Metrica = recode(Metrica, Acuracia = "Acur\u00e1cia", Precisao = "Precis\u00e3o", AUC = "AUC-ROC"))
p_met <- ggplot(met_long, aes(Metrica, Valor, colour = Modelo, shape = Modelo)) +
  geom_point(position = position_dodge(width = 0.6), size = 2.2) +
  coord_cartesian(ylim = c(floor(min(met_long$Valor) * 10) / 10, 1)) +
  scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  labs(x = NULL, y = NULL, colour = NULL, shape = NULL) +
  tema_tcc() + guides(colour = guide_legend(nrow = 2))
salvar_plot(p_met, "plots/metricas_teste.png", width = 16, height = 8)

# --- Analise estratificada por sexo (vies diagnostico) ------------------------
# [Metodologia - 3.6.4 Avaliacao Estratificada por Sexo (sec:avaliacao_sexo)] metricas por sexo
#   no limiar padrao e no limiar clinico.
# [Fundamentação Teórica - 2.7 Vies Algoritmico e Equidade em Saude (sec:fund_vies)] auditoria por
#   subgrupo com sensibilidade, especificidade e AUC-ROC (independentes da prevalencia).
log_subsection("Desempenho estratificado por sexo (Straw & Wu, 2022)")
log_info("Prevalencia de 'doente' por sexo no teste:")
log_table(prop.table(table(Sexo = genero, Classe = teste$Diagnostico), 1), digits = 3)

por_sexo <- map_dfr(resultados, function(r) {
  map_dfr(levels(genero), function(g) {
    idx  <- genero == g
    real <- teste$Diagnostico[idx]; prob <- r$prob[idx]
    roc_g <- roc(real, prob, levels = c("saudavel", "doente"), direction = "<", quiet = TRUE)
    ic_g  <- ci.auc(roc_g, method = "delong")
    m05   <- metricas_no_limiar(prob, real, 0.5)
    m90   <- metricas_no_limiar(prob, real, r$limiar_recall)   # limiar clinico escolhido na CV
    data.frame(Modelo = r$nome, Sexo = g, n = sum(idx), n_doentes = sum(real == "doente"),
               AUC = round(as.numeric(auc(roc_g)), 4),
               AUC_IC_inf = round(ic_g[1], 4), AUC_IC_sup = round(ic_g[3], 4),
               Sensibilidade = round(m05["Sensibilidade"], 4),
               Especificidade = round(m05["Especificidade"], 4),
               Precisao = round(m05["Precisao"], 4),
               Taxa_FN = round(1 - m05["Sensibilidade"], 4),
               Sens_LimiarRecall90 = round(m90["Sensibilidade"], 4),
               Spec_LimiarRecall90 = round(m90["Especificidade"], 4),
               row.names = NULL)
  })
})
log_table(por_sexo %>% select(Modelo, Sexo, n, n_doentes, AUC, Sensibilidade, Especificidade, Taxa_FN), digits = 4)
salvar_csv(por_sexo, "results/metricas_por_sexo.csv")

# Pontos ligados por segmento (grafico de halteres): o comprimento do segmento e a
# diferenca entre os sexos, e o eixo nao precisa comecar no zero.
sexo_long <- por_sexo %>%
  select(Modelo, Sexo, AUC, Sensibilidade, Especificidade) %>%
  pivot_longer(c(AUC, Sensibilidade, Especificidade), names_to = "Metrica", values_to = "Valor") %>%
  mutate(Modelo = rotulo_modelo(Modelo), Metrica = recode(Metrica, AUC = "AUC-ROC"))
p_sexo <- ggplot(sexo_long, aes(Valor, Modelo)) +
  geom_line(aes(group = Modelo), colour = "grey70", linewidth = 0.8) +
  geom_point(aes(colour = Sexo, shape = Sexo), size = 2.2) +
  facet_wrap(~Metrica, ncol = 1) +
  coord_cartesian(xlim = c(floor(min(sexo_long$Valor) * 10) / 10, 1)) +
  scale_x_continuous(breaks = seq(0, 1, 0.1)) +
  labs(x = NULL, y = NULL, colour = NULL, shape = NULL) +
  tema_tcc()
salvar_plot(p_sexo, "plots/metricas_por_sexo.png", width = 13, height = 11)

timer_end(t0, "Etapa 5")
