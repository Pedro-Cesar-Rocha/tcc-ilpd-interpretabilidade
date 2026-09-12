# =============================================================================
# UtilsPipeline.R
# Funcoes utilitarias compartilhadas por todas as etapas da pipeline:
#   - logging padronizado com timestamp e nivel
#   - cronometragem de etapas
#   - criacao de diretorios e persistencia de artefatos com log
# =============================================================================

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

salvar_plot <- function(p, caminho, width = 8, height = 5, dpi = 150) {
  ggplot2::ggsave(caminho, p, width = width, height = height, dpi = dpi)
  log_ok("Grafico salvo: %s", caminho)
}

salvar_csv <- function(df, caminho) {
  utils::write.csv(df, caminho, row.names = FALSE)
  log_ok("CSV salvo: %s (%d linhas)", caminho, nrow(df))
}

# ----------------------------------------------------------------------------
# Modelo customizado LightGBM para o caret
# (caret nao possui metodo nativo; este wrapper permite usar a mesma
#  validacao cruzada, tuning e comparacao via resamples dos demais modelos)
# ----------------------------------------------------------------------------
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
    expand.grid(num_leaves       = c(7, 15, 31),
                learning_rate    = c(0.03, 0.1),
                nrounds          = c(100, 300),
                feature_fraction = c(0.7, 1.0),
                min_data_in_leaf = 20)
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
