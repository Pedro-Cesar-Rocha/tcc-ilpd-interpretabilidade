# =============================================================================
# UtilsPipeline.R
# Funcoes utilitarias compartilhadas por todas as etapas da pipeline:
#   - logging padronizado com timestamp e nivel
#   - cronometragem de etapas
#   - criacao de diretorios e persistencia de artefatos com log
# =============================================================================

# [Fund. 2.1 Mineracao de Dados e Descoberta de Conhecimento (sec:fund_kdd)]
# Semente aleatoria: fixa o gerador de numeros pseudoaleatorios para que os
# sorteios (divisao, CV, SMOTE, busca de hiperparametros, arvores, Monte Carlo)
# se repitam e o experimento seja reprodutivel.
PIPELINE_SEED <- 42

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
log_section <- function(title) {
  linha <- strrep("=", 90)
  cat("\n", linha, "\n", sep = "")
  cat("  ", title, "\n", sep = "")
  cat(linha, "\n", sep = "")
}

log_subsection <- function(title) {
  cat("\n--- ", title, " ", strrep("-", max(0, 84 - nchar(title))), "\n", sep = "")
}

log_step <- function(message, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] [%-5s] %s\n", timestamp, level, message))
}

log_info  <- function(...) log_step(sprintf(...), "INFO")
log_ok    <- function(...) log_step(sprintf(...), "OK")
log_warn  <- function(...) log_step(sprintf(...), "WARN")
log_error <- function(...) log_step(sprintf(...), "ERROR")

# Imprime um data.frame / matriz com indentacao
log_table <- function(obj, digits = 4) {
  txt <- capture.output(print(obj, digits = digits))
  cat(paste0("    ", txt, collapse = "\n"), "\n")
}

# Imprime pares nome : valor alinhados
log_kv <- function(nomes, valores, digits = 4) {
  if (is.numeric(valores)) valores <- format(round(valores, digits))
  largura <- max(nchar(nomes))
  for (i in seq_along(nomes)) {
    cat(sprintf("    %-*s : %s\n", largura, nomes[i], valores[i]))
  }
}

# ----------------------------------------------------------------------------
# Cronometragem
# ----------------------------------------------------------------------------
timer_start <- function() Sys.time()

timer_end <- function(inicio, rotulo = "Etapa") {
  segundos <- as.numeric(difftime(Sys.time(), inicio, units = "secs"))
  log_ok("%s finalizada em %.1f segundos.", rotulo, segundos)
  invisible(segundos)
}

# ----------------------------------------------------------------------------
# Infraestrutura e persistencia
# ----------------------------------------------------------------------------
ensure_dirs <- function(dirs) {
  for (d in dirs) {
    if (!dir.exists(d)) {
      dir.create(d, showWarnings = FALSE, recursive = TRUE)
      log_info("Diretorio criado: %s", d)
    }
  }
}

salvar_rds <- function(obj, caminho) {
  saveRDS(obj, caminho)
  log_ok("Salvo: %s (%.1f KB)", caminho, file.info(caminho)$size / 1024)
}

# --- Graficos no tamanho da pagina do TCC ------------------------------------
# A area util da pagina A4 com margens ABNT (3 cm a esquerda e 2 cm a direita)
# tem 16 cm de largura. Cada grafico e salvo, em centimetros, na largura em que
# sera impresso na monografia (a mesma do \includegraphics), de modo que o texto
# do grafico saia com tamanho proximo ao do texto (8 a 9 pt), sem reducao.
LARGURA_TCC_CM <- 16

salvar_plot <- function(p, caminho, width = LARGURA_TCC_CM, height = 10, dpi = 300,
                        sem_titulo = TRUE) {
  # O titulo e o subtitulo sao retirados porque a legenda da figura no LaTeX
  # (\caption) ja cumpre esse papel; use sem_titulo = FALSE para mante-los.
  if (sem_titulo && inherits(p, "ggplot")) p <- p + ggplot2::labs(title = NULL, subtitle = NULL)
  ggplot2::ggsave(caminho, p, width = width, height = height, units = "cm", dpi = dpi, bg = "white")
  log_ok("Grafico salvo: %s (%.1f x %.1f cm)", caminho, width, height)
}

# Tema padrao dos graficos do TCC.
tema_tcc <- function(base_size = 9, legenda = "bottom") {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(legend.position = legenda,
                   panel.grid.minor = ggplot2::element_blank(),
                   strip.text = ggplot2::element_text(face = "bold"))
}

# Nomes dos modelos com acentos nos graficos (nos objetos ficam sem acento).
rotulo_modelo <- function(x) sub("Regressao Logistica", "Regress\u00e3o Log\u00edstica", x, fixed = TRUE)

salvar_csv <- function(df, caminho) {
  utils::write.csv(df, caminho, row.names = FALSE)
  log_ok("CSV salvo: %s (%d linhas)", caminho, nrow(df))
}

# ----------------------------------------------------------------------------
# SMOTE (smotefamily) como funcao de subamostragem do caret
# Aplicado DENTRO de cada fold da validacao cruzada (apenas na particao de
# treino do fold), evitando que amostras sinteticas correlacionadas caiam nas
# particoes de validacao e inflem as metricas da CV.
# ----------------------------------------------------------------------------
# O SMOTE interpola linearmente TODOS os atributos. Para uma variavel binaria
# (Gender: 0/1 antes da padronizacao), a interpolacao entre pacientes de sexos
# diferentes geraria valores intermediarios sem significado. Apos o SMOTE, cada
# valor sintetico e substituido pela categoria valida mais proxima, ou seja, a
# amostra sintetica herda o sexo de um dos dois pacientes que a originaram.
# [Fund. 2.3.4 Desbalanceamento de Classes e SMOTE (sec:fund_smote)]
#   - SMOTE: nova observacao sintetica em um ponto do segmento entre uma
#     observacao da classe minoritaria e um de seus k vizinhos mais proximos (K = 5).
#   - Correcao de Gender: mesma ideia da variante SMOTE-NC para variaveis binarias.
# [Fund. 2.3.2 Registros Duplicados e Vazamento de Dados (sec:fund_vazamento)]
#   - SMOTE dentro de cada fold: a terceira origem de vazamento descrita no texto.
VARS_BINARIAS <- c("Gender")

categoria_mais_proxima <- function(valores, categorias) {
  vapply(valores, function(v) categorias[which.min(abs(categorias - v))], numeric(1))
}

aplicar_smote <- function(x, y, K = 5) {
  x <- as.data.frame(x)   # o caret entrega uma matriz (interface de formula)
  positiva <- levels(y)[2]
  alvo <- ifelse(y == positiva, 1, 0)
  res  <- suppressWarnings(smotefamily::SMOTE(X = x, target = alvo, K = K, dup_size = 0))
  # smotefamily devolve rbind(minoritaria original, sinteticas, majoritaria original)
  dados      <- res$data
  sinteticas <- res$syn_data
  n_corrigidas <- 0L
  for (v in intersect(VARS_BINARIAS, names(x))) {
    categorias      <- sort(unique(x[[v]]))
    n_corrigidas    <- n_corrigidas + sum(!(sinteticas[[v]] %in% categorias))
    dados[[v]]      <- categoria_mais_proxima(dados[[v]], categorias)
    sinteticas[[v]] <- categoria_mais_proxima(sinteticas[[v]], categorias)
  }
  classe <- factor(ifelse(dados$class == 1, positiva, levels(y)[1]), levels = levels(y))
  dados$class <- NULL
  sinteticas$class <- NULL
  list(x = dados, y = classe, n_sinteticas = nrow(sinteticas), sinteticas = sinteticas,
       n_corrigidas = n_corrigidas)
}

smote_caret <- list(
  name  = "SMOTE (smotefamily)",
  func  = function(x, y) { r <- aplicar_smote(x, y); list(x = r$x, y = r$y) },
  first = TRUE   # sampling antes do pre-processamento interno do caret
)

# ----------------------------------------------------------------------------
# Modelo customizado LightGBM para o caret
# (caret nao possui metodo nativo; este wrapper permite usar a mesma
#  validacao cruzada, tuning e comparacao via resamples dos demais modelos)
# ----------------------------------------------------------------------------
# [Fund. 2.4.5 LightGBM (sec:fund_lgbm)] modelo de boosting com histogramas e
#   crescimento leaf-wise; num_leaves e min_data_in_leaf controlam o sobreajuste.
# [Fund. 2.5.2 Hiperparametros e Sua Otimizacao (sec:fund_hiperparametros)]
#   a funcao grid(search = "random") define o espaco da busca aleatoria.
lightgbm_caret <- list(
  label   = "LightGBM",
  library = "lightgbm",
  type    = "Classification",
  parameters = data.frame(
    parameter = c("num_leaves", "learning_rate", "nrounds", "feature_fraction", "min_data_in_leaf"),
    class     = rep("numeric", 5),
    label     = c("Num. folhas", "Taxa de aprendizado", "Iteracoes", "Fracao de atributos", "Min. obs. por folha")
  ),
  grid = function(x, y, len = NULL, search = "grid") {
    if (search == "random") {
      n <- if (is.null(len)) 30 else len
      data.frame(num_leaves       = sample(c(4, 7, 15, 31, 63), n, replace = TRUE),
                 learning_rate    = round(10^runif(n, -2, -0.5), 4),
                 nrounds          = sample(seq(50, 500, by = 50), n, replace = TRUE),
                 feature_fraction = round(runif(n, 0.5, 1.0), 2),
                 min_data_in_leaf = sample(c(5, 10, 20, 40), n, replace = TRUE))
    } else {
      expand.grid(num_leaves       = c(7, 15, 31),
                  learning_rate    = c(0.03, 0.1),
                  nrounds          = c(100, 300),
                  feature_fraction = c(0.7, 1.0),
                  min_data_in_leaf = 20)
    }
  },
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    label  <- as.integer(y == lev[2])   # lev[2] = classe positiva
    dtrain <- lightgbm::lgb.Dataset(as.matrix(x), label = label)
    params <- list(objective = "binary", metric = "auc",
                   num_leaves = param$num_leaves, learning_rate = param$learning_rate,
                   feature_fraction = param$feature_fraction,
                   min_data_in_leaf = param$min_data_in_leaf,
                   verbose = -1, num_threads = 1, seed = PIPELINE_SEED)
    modelo <- lightgbm::lgb.train(params = params, data = dtrain, nrounds = param$nrounds, verbose = -1)
    list(booster = modelo, lev = lev)   # lightgbm >= 4.0: booster serializavel via saveRDS
  },
  predict = function(modelFit, newdata, submodels = NULL) {
    p <- predict(modelFit$booster, as.matrix(newdata))
    # [Fund. 2.5.3 Metricas de Desempenho / 2.5.5 Limiar de Decisao] limiar padrao 0,5
    factor(ifelse(p >= 0.5, modelFit$lev[2], modelFit$lev[1]), levels = modelFit$lev)
  },
  prob = function(modelFit, newdata, submodels = NULL) {
    p   <- predict(modelFit$booster, as.matrix(newdata))
    out <- data.frame(1 - p, p)
    names(out) <- modelFit$lev
    out
  },
  varImp = function(object, ...) {
    imp <- lightgbm::lgb.importance(object$booster)
    data.frame(Overall = imp$Gain, row.names = imp$Feature)
  },
  levels = function(x) x$lev,
  sort   = function(x) x[order(x$nrounds, x$num_leaves), ]
)
