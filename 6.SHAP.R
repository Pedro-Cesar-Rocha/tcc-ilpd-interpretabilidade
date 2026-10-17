# =============================================================================
# 6.SHAP.R
# Explicabilidade com valores SHAP para os modelos de caixa-preta:
#   - XGBoost  : TreeSHAP exato (predcontrib nativo, escala log-odds)
#   - LightGBM : TreeSHAP exato (somente se for o melhor modelo no teste)
#   - Random Forest : SHAP amostral (fastshap, Monte Carlo, escala de probabilidade)
# Sempre sao explicados o XGBoost, o Random Forest e o melhor modelo pela AUC
# no teste (etapa 5). Para cada um:
#   - importancia global (bar) e distribuicao dos efeitos (beeswarm)
#   - graficos de dependencia para as variaveis mais importantes
#   - explicacao local (waterfall) de pacientes individuais
#   - importancia SHAP estratificada por sexo (auditoria de vies)
# Ao final, compara os rankings de importancia entre modelos.
# Postos: calculados a partir dos valores de importancia SEM arredondar; empates
# reais recebem o posto medio (rank com ties.method = "average"), e o arredondamento
# ocorre apenas na exportacao dos CSV. Variavel usada com importancia nativa minima
# (0 apos a reescala) fica a frente das variaveis nao usadas.
# =============================================================================
# [Fund. 2.6 Interpretabilidade de Modelos (sec:fund_interpretabilidade)] Etapa 5
# da Figura 2.1: explicacao post-hoc dos modelos de ensemble (caixa-preta).
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(shapviz)
}))

log_section("ETAPA 6 - EXPLICABILIDADE (SHAP)")
t0 <- timer_start()

# --- Registro da versao e da citacao do pacote shapviz ----------------------
# Usado para corrigir a entrada @manual{Mayer2026shapviz,...} em
# latex/postextuais/bibliografia3.bib: o campo "year" dessa entrada estava
# provisorio (2026, data futura/placeholder) e deve ser substituido pelo ano
# real de publicacao da versao efetivamente usada no pipeline, obtido abaixo.
log_subsection("Versao e citacao do pacote shapviz (para o .bib)")
shapviz_versao <- as.character(utils::packageVersion("shapviz"))
log_info("shapviz instalado: versao %s", shapviz_versao)
shapviz_citacao <- utils::citation("shapviz")
log_info("Saida de citation(\"shapviz\") (copiar os dados corretos para o .bib):")
print(shapviz_citacao)
log_info("Entrada BibTeX equivalente (toBibtex):")
print(utils::toBibtex(shapviz_citacao))
log_info("Apos conferir ano/autor/url acima, atualizar a entrada 'Mayer2026shapviz' em latex/postextuais/bibliografia3.bib (chave pode ser renomeada para refletir o ano correto, ex.: Mayer2024shapviz).")

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
  # [Fund. 2.6.1 Valores de Shapley (sec:fund_shapley)] e [2.6.2 SHAP (sec:fund_shap)]:
  #   decomposicao aditiva f(x) = phi_0 + soma(phi_j), Eq. 2.7 (eq:shap_aditivo);
  #   o 'baseline' do shapviz e o valor base phi_0.
  # [Fund. 2.6.3 Estimacao dos Valores SHAP (sec:fund_estimacao)]
  #   - XGBoost/LightGBM: TreeSHAP exato, escala do logaritmo da chance (Eq. 2.5);
  #   - Random Forest: Monte Carlo (fastshap, nsim = 100), escala de probabilidade,
  #     com semente fixada porque a estimativa tem erro aleatorio.
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
    # [Fund. 2.6.4 Visualizacoes e Resumos dos Valores SHAP (sec:fund_visualizacoes)]
    #   importancia global = media dos valores SHAP absolutos.
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
    # [Fund. 2.6.4] sentido da associacao entre o valor da variavel e seu valor SHAP
    #   (resumo numerico do beeswarm); correlacao de Pearson (2.2.4).
    direcao <- map_dfr(names(imp), function(v) {
      r <- suppressWarnings(cor(X_teste[[v]], S[, v]))
      data.frame(Variavel = v, Correlacao = round(r, 3),
                 Interpretacao = case_when(
                   is.na(r)  ~ "sem variacao",
                   r >  0.3  ~ "valores altos -> maior risco de doenca",
                   r < -0.3  ~ "valores altos -> menor risco de doenca",
                   TRUE      ~ "efeito nao linear / fraco"))
    }) %>% arrange(desc(abs(Correlacao)))
    log_info("Direcao do efeito (correlacao valor da variavel x SHAP):")
    log_table(direcao)
    salvar_csv(direcao, sprintf("results/shap_direcao_efeito_%s.csv", slug))

    # Graficos globais
    # [Fund. 2.6.4] graficos de barras (importancia global) e beeswarm.
    p_bar <- sv_importance(sv, kind = "bar") +
      labs(title = sprintf("Importancia global (media de |SHAP|) - %s", nome),
           subtitle = sprintf("Escala: %s", res$escala))
    p_bee <- sv_importance(sv, kind = "beeswarm") +
      labs(title = sprintf("Distribuicao dos valores SHAP por variavel - %s", nome),
           subtitle = "Cada ponto e um paciente; cor = valor da variavel (padronizado)")
    salvar_plot(p_bar, sprintf("plots/shap_importancia_%s.png", slug))
    salvar_plot(p_bee, sprintf("plots/shap_beeswarm_%s.png", slug))

    # Dependencia (top 4)
    # [Fund. 2.6.4] graficos de dependencia.
    top4 <- head(imp_df$Variavel, 4)
    for (v in top4) {
      p_dep <- sv_dependence(sv, v = v, color_var = "auto") +
        labs(title = sprintf("Dependencia SHAP - %s (%s)", v, nome))
      salvar_plot(p_dep, sprintf("plots/shap_dependencia_%s_%s.png", slug, v), width = 6, height = 4.5)
    }

    # Explicacoes locais
    # [Fund. 2.6.4] graficos waterfall: explicacoes locais de pacientes individuais.
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
    # [Fund. 2.7 Vies Algoritmico (sec:fund_vies)] verifica se as variaveis que orientam
    #   as previsoes diferem entre os grupos; postos comparados por Spearman (2.2.4).
    log_info("Importancia SHAP estratificada por sexo:")
    imp_sexo <- map_dfr(levels(genero), function(g) {
      idx <- genero == g
      data.frame(Sexo = g, Variavel = colnames(S),
                 Importancia_SHAP = colMeans(abs(S[idx, , drop = FALSE])),
                 SHAP_Medio = colMeans(S[idx, , drop = FALSE]), row.names = NULL)
    }) %>%
      mutate(Importancia_SHAP = ifelse(Importancia_SHAP < 1e-12, 0, Importancia_SHAP))
    imp_sexo_w <- imp_sexo %>%
      select(Sexo, Variavel, Importancia_SHAP) %>%
      pivot_wider(names_from = Sexo, values_from = Importancia_SHAP, names_prefix = "Imp_") %>%
      mutate(Rank_Feminino  = rank(-Imp_Feminino,  ties.method = "average"),
             Rank_Masculino = rank(-Imp_Masculino, ties.method = "average")) %>%
      arrange(Rank_Masculino)
    log_table(imp_sexo_w, digits = 4)
    log_info("Concordancia dos rankings entre sexos (Spearman): %.3f",
             cor(imp_sexo_w$Rank_Feminino, imp_sexo_w$Rank_Masculino, method = "spearman"))
    shap_gen <- imp_sexo %>% filter(Variavel == "Gender")
    log_info("Contribuicao media da variavel Gender: Feminino = %+.4f | Masculino = %+.4f (escala %s)",
             shap_gen$SHAP_Medio[shap_gen$Sexo == "Feminino"], shap_gen$SHAP_Medio[shap_gen$Sexo == "Masculino"], res$escala)
    log_info("Valores de sinal oposto indicam que o sexo, por si so, desloca a previsao de risco entre os grupos.")
    salvar_csv(imp_sexo %>% mutate(across(c(Importancia_SHAP, SHAP_Medio), ~round(.x, 4))),
               sprintf("results/shap_importancia_por_sexo_%s.csv", slug))

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

    # Importancia sem arredondar (postos calculados depois, com posto medio nos empates)
    tibble(Variavel = names(imp), Imp = as.numeric(imp)) %>%
      mutate(Imp = ifelse(Imp < 1e-12, 0, Imp)) %>%
      rename_with(~paste0(.x, "_SHAP_", slug), Imp)
  }

  importancias <- map(a_explicar, explicar)
  names(importancias) <- a_explicar

  # --- Comparacao dos rankings de importancia entre modelos -------------------
  # [Fund. 2.6.3 Estimacao dos Valores SHAP e Efeito da Correlacao (sec:fund_dependencia)]
  #   a correlacao entre variaveis afeta cada medida de importancia de modo distinto:
  #   permutacao do Random Forest (2.4.3), SHAP, coeficientes da Regressao Logistica (2.4.1).
  # [Fund. 2.2.4 Correlacao (sec:fund_correlacao)] Spearman sobre os postos; empates
  #   recebem o posto medio.
  log_subsection("Comparacao dos rankings de importancia entre modelos")
  rank_nativo <- function(caminho, sufixo) {
    m   <- readRDS(caminho)
    imp <- varImp(m, scale = TRUE)$importance
    tibble(Variavel = rownames(imp), Imp = imp[, 1]) %>%
      rename_with(~paste0(.x, "_nativa_", sufixo), Imp)
  }
  comp <- reduce(importancias, full_join, by = "Variavel") %>%
    left_join(rank_nativo("models/modelo_rf.rds",  "rf"),  by = "Variavel") %>%
    left_join(rank_nativo("models/modelo_lgb.rds", "lgb"), by = "Variavel")
  # Regressao logistica: ranking pelo |coeficiente padronizado| (baseline interpretavel).
  # Os coeficientes vem do CSV da etapa 4, ja arredondados a 4 casas (sem empates).
  coef_lr <- read.csv("results/coeficientes_regressao_logistica.csv") %>%
    filter(Variavel != "(Intercept)") %>%
    transmute(Variavel, Coef_abs_lr = abs(Coeficiente))
  comp <- comp %>% left_join(coef_lr, by = "Variavel")

  # Postos: a partir dos valores sem arredondar; empates reais recebem o posto medio.
  # Variavel ausente da medida (nao usada em nenhuma arvore) recebe importancia 0 e fica
  # atras da variavel usada cuja importancia reescalada e 0 (bonus de 1e-6 so para quem foi usada).
  for (col in c(grep("^Imp_", names(comp), value = TRUE), "Coef_abs_lr")) {
    x     <- comp[[col]]
    usada <- !is.na(x)
    x[is.na(x)] <- 0
    comp[[col]] <- x
    nome_rank <- if (col == "Coef_abs_lr") "Rank_lr" else sub("^Imp_", "Rank_", col)
    comp[[nome_rank]] <- rank(-(x + ifelse(usada, 1e-6, 0)), ties.method = "average")
  }
  comp <- comp %>% arrange(Rank_SHAP_xgb, desc(Imp_SHAP_xgb))
  log_table(comp)

  ranks <- comp %>% select(starts_with("Rank_")) %>% as.matrix()
  rho   <- round(cor(ranks, method = "spearman"), 3)
  log_info("Concordancia entre rankings (correlacao de Spearman):")
  log_table(rho, digits = 3)
  salvar_csv(comp %>% mutate(across(c(starts_with("Imp_"), Coef_abs_lr), ~round(.x, 4))),
             "results/comparacao_importancia_modelos.csv")
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