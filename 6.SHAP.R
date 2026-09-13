# =============================================================================
# 6.SHAP.R
# Explicabilidade com valores SHAP para os modelos de caixa-preta:
#   - XGBoost  : TreeSHAP exato (predcontrib nativo, escala log-odds)
#   - LightGBM : TreeSHAP exato (quando for o melhor modelo no teste)
#   - Random Forest : SHAP amostral (fastshap, Monte Carlo, escala de probabilidade)
# Sempre sao explicados o XGBoost, o Random Forest e o melhor modelo pela AUC
# no teste (etapa 5). Para cada um:
#   - importancia global (bar) e distribuicao dos efeitos (beeswarm)
#   - graficos de dependencia para as variaveis mais importantes
#   - explicacao local (waterfall) de pacientes individuais
#   - importancia SHAP estratificada por sexo (auditoria de vies)
# Ao final, compara os rankings de importancia entre modelos.
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

  teste   <- readRDS("data/teste.rds")
  genero  <- readRDS("data/teste_genero.rds")
  X_teste <- teste %>% select(-Dataset)
  real    <- teste$Dataset
  melhor  <- if (file.exists("results/melhor_modelo.rds")) readRDS("results/melhor_modelo.rds") else "XGBoost"

  arquivos <- c("Regressao Logistica" = "models/modelo_lr.rds", "Random Forest" = "models/modelo_rf.rds",
                "XGBoost" = "models/modelo_xgb.rds", "LightGBM" = "models/modelo_lgb.rds")
  slugs    <- c("Regressao Logistica" = "lr", "Random Forest" = "rf", "XGBoost" = "xgb", "LightGBM" = "lgb")

  a_explicar <- unique(c("XGBoost", "Random Forest", melhor))
  log_info("Melhor modelo no teste (etapa 5): %s", melhor)
  log_info("Modelos a explicar: %s | %d observacoes de teste | %d preditores",
           paste(a_explicar, collapse = ", "), nrow(X_teste), ncol(X_teste))
  log_info("SHAP decompoe cada previsao em contribuicoes aditivas de cada variavel.")
  log_info("Valor > 0 empurra a previsao para 'doente'; valor < 0 empurra para 'saudavel'.")

  pfun <- function(object, newdata) predict(object, newdata, type = "prob")[, "doente"]

  # --- Calculo dos valores SHAP conforme o tipo de modelo ---------------------
  calcular_shap <- function(nome, modelo) {
    if (nome == "XGBoost") {
      log_info("Metodo: TreeSHAP exato (xgboost predcontrib) - escala log-odds.")
      booster <- modelo$finalModel
      M <- predict(booster, as.matrix(X_teste[, booster$xNames]), predcontrib = TRUE)
      # caret::xgbTree codifica o PRIMEIRO nivel ('saudavel') como 1; inverte-se o sinal
      # para que SHAP > 0 signifique "empurra para 'doente'", como nos demais modelos.
      S <- -M[, booster$xNames, drop = FALSE]
      sv <- shapviz(S, X = X_teste, baseline = -mean(M[, "BIAS"]))
      list(sv = sv, escala = "log-odds")
    } else if (nome == "LightGBM") {
      log_info("Metodo: TreeSHAP exato (lightgbm predcontrib) - escala log-odds.")
      sv <- shapviz(modelo$finalModel$booster, X_pred = as.matrix(X_teste), X = X_teste)
      list(sv = sv, escala = "log-odds")
    } else {
      log_info("Metodo: SHAP amostral (fastshap, Monte Carlo, nsim = 100) - escala de probabilidade.")
      set.seed(PIPELINE_SEED)
      shap_values <- explain(object = modelo, X = X_teste, pred_wrapper = pfun, nsim = 100)
      sv <- shapviz(shap_values, X = X_teste, baseline = mean(pfun(modelo, X_teste)))
      list(sv = sv, escala = "probabilidade")
    }
  }

  # --- Explicacao completa de um modelo ---------------------------------------
  explicar <- function(nome) {
    slug   <- slugs[[nome]]
    modelo <- readRDS(arquivos[[nome]])
    log_subsection(sprintf("SHAP - %s", nome))
    ti  <- timer_start()
    res <- calcular_shap(nome, modelo)
    sv  <- res$sv
    S   <- get_shap_values(sv)
    timer_end(ti, sprintf("Calculo SHAP (%s)", nome))
    log_info("Valor base (media das previsoes na escala %s): %.4f", res$escala, get_baseline(sv))

    # Importancia global
    imp <- colMeans(abs(S))
    imp_df <- data.frame(Variavel = names(imp), Importancia_SHAP = round(imp, 4), row.names = NULL) %>%
      arrange(desc(Importancia_SHAP)) %>%
      mutate(Percentual = round(100 * Importancia_SHAP / sum(Importancia_SHAP), 1),
             Acumulado  = cumsum(Percentual))
    log_info("Importancia global (media de |SHAP|):")
    log_table(imp_df, digits = 4)
    n_80 <- which(imp_df$Acumulado >= 80)[1]
    log_info("As %d variaveis mais importantes respondem por ~80%% do impacto total nas previsoes.", n_80)
    salvar_csv(imp_df, sprintf("results/shap_importancia_%s.csv", slug))

    # Direcao do efeito
    direcao <- map_dfr(names(imp), function(v) {
      r <- suppressWarnings(cor(X_teste[[v]], S[, v]))
      data.frame(Variavel = v, Correlacao = round(r, 3),
                 Interpretacao = case_when(
                   is.na(r)  ~ "sem variacao",
                   r >  0.3  ~ "valores altos -> maior risco de doenca",
                   r < -0.3  ~ "valores altos -> menor risco de doenca",
                   TRUE      ~ "efeito nao monotonico / fraco"))
    }) %>% arrange(desc(abs(Correlacao)))
    log_info("Direcao do efeito (correlacao valor da variavel x SHAP):")
    log_table(direcao)
    salvar_csv(direcao, sprintf("results/shap_direcao_efeito_%s.csv", slug))

    # Graficos globais
    p_bar <- sv_importance(sv, kind = "bar") +
      labs(title = sprintf("Importancia global (media de |SHAP|) - %s", nome),
           subtitle = sprintf("Escala: %s", res$escala))
    p_bee <- sv_importance(sv, kind = "beeswarm") +
      labs(title = sprintf("Distribuicao dos valores SHAP por variavel - %s", nome),
           subtitle = "Cada ponto e um paciente; cor = valor da variavel (padronizado)")
    salvar_plot(p_bar, sprintf("plots/shap_importancia_%s.png", slug))
    salvar_plot(p_bee, sprintf("plots/shap_beeswarm_%s.png", slug))

    # Dependencia (top 4)
    top4 <- head(imp_df$Variavel, 4)
    for (v in top4) {
      p_dep <- sv_dependence(sv, v = v, color_var = "auto") +
        labs(title = sprintf("Dependencia SHAP - %s (%s)", v, nome))
      salvar_plot(p_dep, sprintf("plots/shap_dependencia_%s_%s.png", slug, v), width = 6, height = 4.5)
    }

    # Explicacoes locais
    probs <- pfun(modelo, X_teste)
    casos <- c(
      "VP_maior_confianca" = which(real == "doente")[which.max(probs[real == "doente"])],
      "VN_maior_confianca" = which(real == "saudavel")[which.min(probs[real == "saudavel"])],
      "FN_pior_erro"       = which(real == "doente")[which.min(probs[real == "doente"])]
    )
    log_info("Explicacoes locais (waterfall):")
    for (nm in names(casos)) {
      i <- casos[[nm]]
      contrib <- sort(S[i, ], decreasing = TRUE)
      log_info("  %-20s paciente #%d | real = %-8s | sexo = %-9s | prob(doente) = %.3f", nm, i, real[i], genero[i], probs[i])
      log_info("      maior contribuicao p/ doente: %s (%+.3f) | p/ saudavel: %s (%+.3f)",
               names(contrib)[1], contrib[1], names(contrib)[length(contrib)], contrib[length(contrib)])
      p_wf <- sv_waterfall(sv, row_id = i) +
        labs(title = sprintf("Explicacao local - paciente #%d (%s) - %s", i, nm, nome),
             subtitle = sprintf("Real: %s | Sexo: %s | Prob. prevista de doente: %.3f", real[i], genero[i], probs[i]))
      salvar_plot(p_wf, sprintf("plots/shap_waterfall_%s_%s.png", slug, nm), width = 7, height = 5)
    }

    # SHAP estratificado por sexo (auditoria de vies)
    log_info("Importancia SHAP estratificada por sexo:")
    imp_sexo <- map_dfr(levels(genero), function(g) {
      idx <- genero == g
      data.frame(Sexo = g, Variavel = colnames(S),
                 Importancia_SHAP = round(colMeans(abs(S[idx, , drop = FALSE])), 4),
                 SHAP_Medio = round(colMeans(S[idx, , drop = FALSE]), 4), row.names = NULL)
    })
    imp_sexo_w <- imp_sexo %>%
      select(Sexo, Variavel, Importancia_SHAP) %>%
      pivot_wider(names_from = Sexo, values_from = Importancia_SHAP, names_prefix = "Imp_") %>%
      mutate(Rank_Feminino = rank(-Imp_Feminino), Rank_Masculino = rank(-Imp_Masculino)) %>%
      arrange(Rank_Masculino)
    log_table(imp_sexo_w, digits = 4)
    log_info("Concordancia dos rankings entre sexos (Spearman): %.3f",
             cor(imp_sexo_w$Rank_Feminino, imp_sexo_w$Rank_Masculino, method = "spearman"))
    shap_gen <- imp_sexo %>% filter(Variavel == "Gender")
    log_info("Contribuicao media da variavel Gender: Feminino = %+.4f | Masculino = %+.4f (escala %s)",
             shap_gen$SHAP_Medio[shap_gen$Sexo == "Feminino"], shap_gen$SHAP_Medio[shap_gen$Sexo == "Masculino"], res$escala)
    log_info("Valores de sinal oposto indicam que o sexo, por si so, desloca a previsao de risco entre os grupos.")
    salvar_csv(imp_sexo, sprintf("results/shap_importancia_por_sexo_%s.csv", slug))

    p_sexo <- imp_sexo %>%
      mutate(Variavel = factor(Variavel, levels = rev(imp_df$Variavel))) %>%
      ggplot(aes(Variavel, Importancia_SHAP, fill = Sexo)) +
      geom_col(position = position_dodge(0.8), width = 0.7) +
      coord_flip() +
      labs(title = sprintf("Importancia SHAP por sexo - %s", nome),
           subtitle = sprintf("Media de |SHAP| no teste; escala: %s", res$escala), x = NULL, y = "Media |SHAP|") +
      theme_minimal() + theme(legend.position = "bottom")
    salvar_plot(p_sexo, sprintf("plots/shap_importancia_por_sexo_%s.png", slug), width = 7, height = 5)

    p_bee_sexo <- data.frame(SHAP_Gender = S[, "Gender"], Sexo = genero, Classe = real) %>%
      ggplot(aes(Sexo, SHAP_Gender, colour = Classe)) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
      geom_jitter(width = 0.15, alpha = 0.7, size = 1.6) +
      labs(title = sprintf("Contribuicao SHAP da variavel Gender por sexo - %s", nome),
           subtitle = "Valores > 0 aumentam a previsao de 'doente'", y = sprintf("SHAP(Gender) [%s]", res$escala), x = NULL) +
      theme_minimal() + theme(legend.position = "bottom")
    salvar_plot(p_bee_sexo, sprintf("plots/shap_gender_por_sexo_%s.png", slug), width = 6, height = 4.5)

    imp_df %>% transmute(Variavel, Imp = Importancia_SHAP, Rank = row_number()) %>%
      rename_with(~paste0(.x, "_SHAP_", slug), c(Imp, Rank))
  }

  importancias <- map(a_explicar, explicar)
  names(importancias) <- a_explicar

  # --- Comparacao dos rankings de importancia entre modelos -------------------
  log_subsection("Comparacao dos rankings de importancia entre modelos")
  rank_nativo <- function(caminho, sufixo) {
    m   <- readRDS(caminho)
    imp <- varImp(m, scale = TRUE)$importance
    data.frame(Variavel = rownames(imp), Imp = round(imp[, 1], 2), row.names = NULL) %>%
      arrange(desc(Imp)) %>% mutate(Rank = row_number()) %>%
      rename_with(~paste0(.x, "_nativa_", sufixo), c(Imp, Rank))
  }
  comp <- reduce(importancias, full_join, by = "Variavel") %>%
    left_join(rank_nativo("models/modelo_rf.rds",  "rf"),  by = "Variavel") %>%
    left_join(rank_nativo("models/modelo_lgb.rds", "lgb"), by = "Variavel")
  # Regressao logistica: ranking pelo |coeficiente padronizado| (baseline interpretavel)
  coef_lr <- read.csv("results/coeficientes_regressao_logistica.csv") %>%
    filter(Variavel != "(Intercept)") %>%
    arrange(desc(abs(Coeficiente))) %>%
    transmute(Variavel, Coef_abs_lr = abs(Coeficiente), Rank_lr = row_number())
  comp <- comp %>% left_join(coef_lr, by = "Variavel")
  # Variavel nao usada em nenhuma arvore: importancia 0 e ultima posicao no ranking
  comp <- comp %>%
    mutate(across(starts_with("Rank_"), ~replace_na(.x, n())),
           across(-c(Variavel, starts_with("Rank_")), ~replace_na(.x, 0))) %>%
    arrange(Rank_SHAP_xgb)
  log_table(comp)

  ranks <- comp %>% select(starts_with("Rank_")) %>% as.matrix()
  rho   <- round(cor(ranks, method = "spearman"), 3)
  log_info("Concordancia entre rankings (correlacao de Spearman):")
  log_table(rho, digits = 3)
  salvar_csv(comp, "results/comparacao_importancia_modelos.csv")
  salvar_csv(as.data.frame(rho) %>% rownames_to_column("Ranking"), "results/concordancia_rankings_spearman.csv")

  log_subsection("Como ler os resultados")
  log_info("shap_importancia_<m>.png          : quais variaveis mais pesam nas decisoes do modelo <m>, em media.")
  log_info("shap_beeswarm_<m>.png             : direcao do efeito - pontos a direita aumentam a chance de 'doente'.")
  log_info("shap_dependencia_<m>_<var>.png    : como o SHAP muda conforme o valor de cada variavel.")
  log_info("shap_waterfall_<m>_<caso>.png     : porque o modelo deu determinada previsao para UM paciente.")
  log_info("shap_importancia_por_sexo_<m>.png : se o modelo usa as variaveis de forma diferente para homens e mulheres.")
  log_info("shap_gender_por_sexo_<m>.png      : quanto o sexo, isoladamente, desloca a previsao de risco.")
  log_info("<m> = xgb (TreeSHAP exato), rf (fastshap), lgb (TreeSHAP exato).")
}

timer_end(t0, "Etapa 6")
