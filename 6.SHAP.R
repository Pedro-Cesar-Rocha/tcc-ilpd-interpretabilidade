# =============================================================================
# 6.SHAP.R
# Explicabilidade com valores SHAP para os dois modelos de caixa-preta explicados:
#   - XGBoost  : TreeSHAP exato (predcontrib nativo, escala log-odds)
#   - Random Forest : SHAP amostral (fastshap, Monte Carlo, escala de probabilidade)
# Para cada um:
#   - importancia global (bar) e distribuicao dos efeitos (beeswarm)
#   - grafico de dependencia da variavel mais importante
#   - explicacao local (waterfall) do verdadeiro positivo de maior confianca e
#     do falso negativo de pior erro
#   - importancia SHAP estratificada por sexo (auditoria de vies)
# Ao final, compara tres rankings de importancia: SHAP do XGBoost, SHAP do
# Random Forest e |coeficiente padronizado| da Regressao Logistica.
# Postos: calculados a partir dos valores SEM arredondar; empates recebem o
# posto medio (rank com ties.method = "average").
# =============================================================================
# [Fundamentação Teórica - 2.6 Interpretabilidade de Modelos (sec:fund_interpretabilidade)] Etapa 5
# da Figura 2.1: explicacao post-hoc dos modelos de ensemble (caixa-preta).
# [Metodologia - 3.7 Interpretabilidade com SHAP (sec:shap)] este script implementa as
#   secoes 3.7.1 a 3.7.4 da Metodologia.
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
  X_teste <- teste %>% select(-Diagnostico)
  real    <- teste$Diagnostico

  arquivos <- c("Random Forest" = "models/modelo_rf.rds", "XGBoost" = "models/modelo_xgb.rds")
  slugs    <- c("Random Forest" = "rf", "XGBoost" = "xgb")

  # [Metodologia - 3.7.1 Modelos Explicados (sec:modelos_explicados)] sempre o XGBoost
  #   (boosting) e o Random Forest (bagging).
  a_explicar <- c("XGBoost", "Random Forest")
  log_info("Modelos a explicar: %s | %d observacoes de teste | %d preditores",
           paste(a_explicar, collapse = ", "), nrow(X_teste), ncol(X_teste))
  log_info("SHAP decompoe cada previsao em contribuicoes aditivas de cada variavel.")
  log_info("Valor > 0 empurra a previsao para 'doente'; valor < 0 empurra para 'saudavel'.")

  pfun <- function(object, newdata) predict(object, newdata, type = "prob")[, "doente"]

  # --- Calculo dos valores SHAP conforme o tipo de modelo ---------------------
  # [Metodologia - 3.7.1 Modelos Explicados (sec:modelos_explicados)] TreeSHAP no XGBoost (sinal
  #   invertido porque o caret codifica "saudavel" como positiva) e Monte Carlo no
  #   Random Forest (nsim = 100, semente 42).
  # [Fundamentação Teórica - 2.6.1 Valores de Shapley (sec:fund_shapley)] e [2.6.2 SHAP (sec:fund_shap)]:
  #   decomposicao aditiva f(x) = phi_0 + soma(phi_j), Eq. 2.5 (eq:shap_aditivo);
  #   o 'baseline' do shapviz e o valor base phi_0.
  # [Fundamentação Teórica - 2.6.3 Estimacao dos Valores SHAP (sec:fund_estimacao)]
  #   - XGBoost: TreeSHAP exato, escala do logaritmo da chance (Eq. 2.4);
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
    } else {
      log_info("Metodo: SHAP amostral (fastshap, Monte Carlo, nsim = 100) - escala de probabilidade.")
      set.seed(PIPELINE_SEED)
      shap_values <- explain(object = modelo, X = X_teste, pred_wrapper = pfun, nsim = 100)
      sv <- shapviz(shap_values, X = X_teste, baseline = mean(pfun(modelo, X_teste)))
      list(sv = sv, escala = "probabilidade")
    }
  }

  # --- Explicacao completa de um modelo ---------------------------------------
  # [Metodologia - 3.7.2 Analises Realizadas (sec:analises_shap)] importancia global, direcao
  #   do efeito, dependencia, waterfall (VP, VN e FN selecionados) e
  # [Metodologia - 3.7.3 Auditoria de Vies por Sexo (sec:shap_sexo)] importancia SHAP por sexo
  #   e contribuicao da variavel Gender.
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
    # Rotulo da escala dos valores SHAP nos eixos dos graficos.
    esc <- if (res$escala == "log-odds") "log da chance" else "probabilidade"

    # Importancia global
    # [Fundamentação Teórica - 2.6.4 Visualizacoes e Resumos dos Valores SHAP (sec:fund_visualizacoes)]
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
    # [Fundamentação Teórica - 2.6.4] sentido da associacao entre o valor da variavel e seu valor SHAP
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
    # [Fundamentação Teórica - 2.6.4] graficos de barras (importancia global) e beeswarm.
    # Estes graficos entram lado a lado no TCC (XGBoost e Random Forest, cada um com
    # 0,49 da largura do texto, ~8 cm). Sao salvos com 10 cm e fonte 8, o que da
    # cerca de 6,5 pt impressos; o eixo informa a escala dos valores SHAP.
    p_bar <- sv_importance(sv, kind = "bar") +
      labs(x = sprintf("M\u00e9dia de |SHAP| (%s)", esc)) +
      tema_tcc(base_size = 8, legenda = "none")
    p_bee <- sv_importance(sv, kind = "beeswarm") +
      labs(x = sprintf("Valor SHAP (%s)", esc), colour = "Valor da\nvari\u00e1vel") +
      tema_tcc(base_size = 8, legenda = "right") +
      # Rotulos da barra de cores em portugues: o shapviz reescala a cor de cada
      # variavel para [0, 1] e rotula os extremos como "Low"/"High"; aqui a mesma
      # paleta (inferno) e refeita com os rotulos "Baixo"/"Alto".
      ggplot2::scale_colour_viridis_c(option = "inferno", begin = 0.25, end = 0.85,
                                      breaks = 0:1, labels = c("Baixo", "Alto"))
    salvar_plot(p_bar, sprintf("plots/shap_importancia_%s.png", slug), width = 10, height = 8)
    salvar_plot(p_bee, sprintf("plots/shap_beeswarm_%s.png", slug), width = 10, height = 8)

    # Dependencia (apenas a variavel mais importante de cada modelo)
    # [Fundamentação Teórica - 2.6.4] graficos de dependencia.
    for (v in head(imp_df$Variavel, 1)) {
      p_dep <- sv_dependence(sv, v = v, color_var = "auto") +
        labs(y = sprintf("SHAP (%s)", esc)) +
        tema_tcc(base_size = 8, legenda = "right")
      salvar_plot(p_dep, sprintf("plots/shap_dependencia_%s_%s.png", slug, v), width = 10, height = 7.5)
    }

    # Explicacoes locais
    # [Fundamentação Teórica - 2.6.4] graficos waterfall: explicacoes locais de pacientes individuais.
    probs <- pfun(modelo, X_teste)
    casos <- c(
      "VP_maior_confianca" = which(real == "doente")[which.max(probs[real == "doente"])],
      "FN_pior_erro"       = which(real == "doente")[which.min(probs[real == "doente"])]
    )
    log_info("Explicacoes locais (waterfall):")
    for (nm in names(casos)) {
      i <- casos[[nm]]
      contrib <- sort(S[i, ], decreasing = TRUE)
      log_info("  %-20s paciente #%d | real = %-8s | sexo = %-9s | prob(doente) = %.3f", nm, i, real[i], genero[i], probs[i])
      log_info("      maior contribuicao p/ doente: %s (%+.3f) | p/ saudavel: %s (%+.3f)",
               names(contrib)[1], contrib[1], names(contrib)[length(contrib)], contrib[length(contrib)])
      p_wf <- sv_waterfall(sv, row_id = i, annotation_size = 2.4) +
        labs(x = sprintf("Valor SHAP (%s)", esc)) +
        tema_tcc(base_size = 8, legenda = "none")
      salvar_plot(p_wf, sprintf("plots/shap_waterfall_%s_%s.png", slug, nm), width = 10, height = 8)
    }

    # SHAP estratificado por sexo (auditoria de vies)
    # [Fundamentação Teórica - 2.7 Vies Algoritmico (sec:fund_vies)] verifica se as variaveis que orientam
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
      labs(x = NULL, y = sprintf("M\u00e9dia de |SHAP| (%s)", esc), fill = NULL) +
      tema_tcc(base_size = 8)
    salvar_plot(p_sexo, sprintf("plots/shap_importancia_por_sexo_%s.png", slug), width = 10, height = 8.5)

    p_bee_sexo <- data.frame(SHAP_Gender = S[, "Gender"], Sexo = genero, Classe = real) %>%
      ggplot(aes(Sexo, SHAP_Gender, colour = Classe)) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
      geom_point(position = position_jitter(width = 0.15, height = 0, seed = PIPELINE_SEED),
                 alpha = 0.7, size = 1.3) +
      labs(y = sprintf("SHAP de Gender (%s)", esc), x = NULL, colour = NULL) +
      tema_tcc()
    salvar_plot(p_bee_sexo, sprintf("plots/shap_gender_por_sexo_%s.png", slug), width = 10.5, height = 7.5)

    # Importancia sem arredondar (postos calculados depois, com posto medio nos empates)
    tibble(Variavel = names(imp), Imp = as.numeric(imp)) %>%
      mutate(Imp = ifelse(Imp < 1e-12, 0, Imp)) %>%
      rename_with(~paste0(.x, "_SHAP_", slug), Imp)
  }

  importancias <- map(a_explicar, explicar)
  names(importancias) <- a_explicar

  # --- Comparacao dos rankings de importancia entre modelos -------------------
  # [Metodologia - 3.7.4 Comparacao das Importancias entre os Modelos (sec:comparacao_importancias)]
  #   tres rankings (SHAP XGB, SHAP RF, |coef| RL) e correlacao de Spearman entre eles.
  # [Fundamentação Teórica - 2.6.3 Estimacao dos Valores SHAP e Efeito da Correlacao (sec:fund_dependencia)]
  #   a correlacao entre variaveis afeta cada medida de importancia de modo distinto:
  #   SHAP e coeficientes da Regressao Logistica (2.4.1).
  # [Fundamentação Teórica - 2.2.4 Correlacao (sec:fund_correlacao)] Spearman sobre os postos; empates
  #   recebem o posto medio.
  log_subsection("Comparacao dos rankings de importancia entre modelos")
  comp <- reduce(importancias, full_join, by = "Variavel")
  # Regressao logistica: ranking pelo |coeficiente padronizado| (baseline interpretavel).
  # Os coeficientes vem do CSV da etapa 4, ja arredondados a 4 casas (sem empates).
  coef_lr <- read.csv("results/coeficientes_regressao_logistica.csv") %>%
    filter(Variavel != "(Intercept)") %>%
    transmute(Variavel, Coef_abs_lr = abs(Coeficiente))
  comp <- comp %>% left_join(coef_lr, by = "Variavel")

  # Postos a partir dos valores sem arredondar; empates (por exemplo, variaveis que
  # o XGBoost nao usa, todas com importancia 0) recebem o posto medio.
  for (col in c(grep("^Imp_", names(comp), value = TRUE), "Coef_abs_lr")) {
    x <- comp[[col]]
    x[is.na(x)] <- 0
    comp[[col]] <- x
    nome_rank <- if (col == "Coef_abs_lr") "Rank_lr" else sub("^Imp_", "Rank_", col)
    comp[[nome_rank]] <- rank(-x, ties.method = "average")
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
  log_info("shap_dependencia_<m>_<var>.png    : como o SHAP muda conforme o valor da variavel mais importante.")
  log_info("shap_waterfall_<m>_<caso>.png     : porque o modelo deu determinada previsao para UM paciente.")
  log_info("shap_importancia_por_sexo_<m>.png : se o modelo usa as variaveis de forma diferente para homens e mulheres.")
  log_info("shap_gender_por_sexo_<m>.png      : quanto o sexo, isoladamente, desloca a previsao de risco.")
  log_info("<m> = xgb (TreeSHAP exato), rf (fastshap).")
}

timer_end(t0, "Etapa 6")