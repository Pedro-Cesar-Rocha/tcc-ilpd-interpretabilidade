# =============================================================================
# 6.SHAP.R
# Explicabilidade do modelo XGBoost com valores SHAP (fastshap + shapviz):
#   - importancia global (bar) e distribuicao dos efeitos (beeswarm)
#   - graficos de dependencia para as variaveis mais importantes
#   - explicacao local (waterfall) de pacientes individuais
#   - tabela textual interpretavel em results/
# =============================================================================
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(shapviz)
}))

log_section("ETAPA 6 - EXPLICABILIDADE (SHAP)")
t0 <- timer_start()

if (!requireNamespace("fastshap", quietly = TRUE)) {
  log_warn("fastshap nao instalado; etapa 6 sera ignorada.")
} else {
  suppressWarnings(suppressPackageStartupMessages(library(fastshap)))
  xgboost::xgb.set.config(verbosity = 0)

  modelo_xgb <- readRDS("models/modelo_xgb.rds")
  teste      <- readRDS("data/teste.rds")
  X_teste    <- teste %>% select(-Dataset)
  log_info("Modelo explicado: XGBoost | %d observacoes de teste | %d preditores", nrow(X_teste), ncol(X_teste))

  # --- Calculo dos valores SHAP -----------------------------------------------
  log_subsection("Calculando valores SHAP (Monte Carlo, nsim = 100)")
  log_info("SHAP decompoe cada previsao em contribuicoes aditivas de cada variavel.")
  log_info("Valor > 0 empurra a previsao para 'doente'; valor < 0 empurra para 'saudavel'.")
  pfun <- function(object, newdata) predict(object, newdata, type = "prob")[, "doente"]

  set.seed(PIPELINE_SEED)
  ti <- timer_start()
  shap_values <- explain(object = modelo_xgb, X = X_teste, pred_wrapper = pfun, nsim = 100)
  timer_end(ti, "Calculo SHAP")

  prob_media <- mean(pfun(modelo_xgb, X_teste))
  log_info("Probabilidade media prevista de 'doente' (valor base): %.4f", prob_media)

  sv <- shapviz(shap_values, X = X_teste, baseline = prob_media)

  # --- Importancia global ----------------------------------------------------
  log_subsection("Importancia global das variaveis (media de |SHAP|)")
  imp <- colMeans(abs(as.matrix(shap_values)))
  imp_df <- data.frame(Variavel = names(imp), Importancia_SHAP = round(imp, 4), row.names = NULL) %>%
    arrange(desc(Importancia_SHAP)) %>%
    mutate(Percentual = round(100 * Importancia_SHAP / sum(Importancia_SHAP), 1),
           Acumulado  = cumsum(Percentual))
  log_table(imp_df, digits = 4)
  n_80 <- which(imp_df$Acumulado >= 80)[1]
  log_info("As %d variaveis mais importantes respondem por ~80%% do impacto total nas previsoes.", n_80)
  salvar_csv(imp_df, "results/shap_importancia.csv")

  # --- Direcao do efeito ------------------------------------------------------
  log_subsection("Direcao do efeito (correlacao entre valor da variavel e SHAP)")
  direcao <- map_dfr(names(imp), function(v) {
    r <- suppressWarnings(cor(X_teste[[v]], shap_values[, v]))
    data.frame(Variavel = v, Correlacao = round(r, 3),
               Interpretacao = case_when(
                 is.na(r)  ~ "sem variacao",
                 r >  0.3  ~ "valores altos -> maior risco de doenca",
                 r < -0.3  ~ "valores altos -> menor risco de doenca",
                 TRUE      ~ "efeito nao monotonico / fraco"))
  }) %>% arrange(desc(abs(Correlacao)))
  log_table(direcao)
  salvar_csv(direcao, "results/shap_direcao_efeito.csv")

  # --- Graficos globais -------------------------------------------------------
  log_subsection("Gerando graficos globais")
  p_bar <- sv_importance(sv, kind = "bar") +
    labs(title = "Importancia global (media de |SHAP|) - XGBoost")
  p_bee <- sv_importance(sv, kind = "beeswarm") +
    labs(title = "Distribuicao dos valores SHAP por variavel - XGBoost",
         subtitle = "Cada ponto e um paciente; cor = valor da variavel (padronizado)")
  salvar_plot(p_bar, "plots/shap_importancia.png")
  salvar_plot(p_bee, "plots/shap_beeswarm.png")

  # --- Graficos de dependencia (top 4) ---------------------------------------
  log_subsection("Graficos de dependencia para as 4 variaveis mais importantes")
  top4 <- head(imp_df$Variavel, 4)
  for (v in top4) {
    p_dep <- sv_dependence(sv, v = v, color_var = "auto") +
      labs(title = sprintf("Dependencia SHAP - %s", v))
    salvar_plot(p_dep, sprintf("plots/shap_dependencia_%s.png", v), width = 6, height = 4.5)
  }

  # --- Explicacoes locais -----------------------------------------------------
  log_subsection("Explicacoes locais (waterfall) de pacientes individuais")
  probs <- pfun(modelo_xgb, X_teste)
  real  <- teste$Dataset
  casos <- c(
    "VP_maior_confianca" = which(real == "doente")[which.max(probs[real == "doente"])],
    "VN_maior_confianca" = which(real == "saudavel")[which.min(probs[real == "saudavel"])],
    "FN_pior_erro"       = which(real == "doente")[which.min(probs[real == "doente"])]
  )
  for (nm in names(casos)) {
    i <- casos[[nm]]
    log_info("%-20s paciente #%d | real = %-8s | prob(doente) = %.3f", nm, i, real[i], probs[i])
    contrib <- sort(shap_values[i, ], decreasing = TRUE)
    log_info("    maior contribuicao p/ doente: %s (%+.3f) | p/ saudavel: %s (%+.3f)",
             names(contrib)[1], contrib[1], names(contrib)[length(contrib)], contrib[length(contrib)])
    p_wf <- sv_waterfall(sv, row_id = i) +
      labs(title = sprintf("Explicacao local - paciente #%d (%s)", i, nm),
           subtitle = sprintf("Real: %s | Prob. prevista de doente: %.3f", real[i], probs[i]))
    salvar_plot(p_wf, sprintf("plots/shap_waterfall_%s.png", nm), width = 7, height = 5)
  }

  # --- Comparacao com importancia nativa dos outros modelos -------------------
  log_subsection("Comparacao: ranking SHAP (XGBoost) vs importancia nativa (RF e LightGBM)")
  rank_nativo <- function(caminho, sufixo) {
    m   <- readRDS(caminho)
    imp <- varImp(m, scale = TRUE)$importance
    data.frame(Variavel = rownames(imp), Imp = round(imp[, 1], 2), row.names = NULL) %>%
      arrange(desc(Imp)) %>% mutate(Rank = row_number()) %>%
      rename_with(~paste0(.x, "_", sufixo), c(Imp, Rank))
  }
  comp <- imp_df %>% select(Variavel, Importancia_SHAP) %>%
    mutate(Rank_SHAP = row_number()) %>%
    left_join(rank_nativo("models/modelo_rf.rds",  "RF"),  by = "Variavel") %>%
    left_join(rank_nativo("models/modelo_lgb.rds", "LGB"), by = "Variavel")
  comp[is.na(comp)] <- 0   # variavel nao usada em nenhuma arvore do LightGBM
  log_table(comp)
  log_info("Concordancia de rankings (Spearman) - SHAP vs RF: %.3f | SHAP vs LightGBM: %.3f",
           cor(comp$Rank_SHAP, comp$Rank_RF, method = "spearman"),
           cor(comp$Rank_SHAP, comp$Rank_LGB, method = "spearman"))
  salvar_csv(comp, "results/comparacao_importancia_shap_rf_lgb.csv")

  log_subsection("Como ler os resultados")
  log_info("shap_importancia.png : quais variaveis mais pesam nas decisoes do modelo, em media.")
  log_info("shap_beeswarm.png    : direcao do efeito - pontos a direita aumentam a chance de 'doente'.")
  log_info("shap_dependencia_*   : como o SHAP muda conforme o valor de cada variavel.")
  log_info("shap_waterfall_*     : porque o modelo deu determinada previsao para UM paciente.")
}

timer_end(t0, "Etapa 6")
