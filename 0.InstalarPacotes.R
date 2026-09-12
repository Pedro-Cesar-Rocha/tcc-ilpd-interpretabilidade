# =============================================================================
# 0.InstalarPacotes.R
# Verifica e instala as dependencias da pipeline, fixando versoes onde ha
# incompatibilidades conhecidas (caret + xgboost; fastshap arquivado no CRAN).
# =============================================================================
source("UtilsPipeline.R")
options(repos = c(CRAN = "https://cloud.r-project.org"))

log_section("ETAPA 0 - INSTALACAO E VERIFICACAO DE PACOTES")
t0 <- timer_start()

pacotes <- c(
  "tidyverse",     # manipulacao de dados e graficos
  "caret",         # framework de treino / validacao cruzada
  "randomForest",  # Random Forest
  "lightgbm",      # LightGBM
  "e1071",         # dependencias do caret
  "smotefamily",   # SMOTE
  "pROC",          # curvas ROC e AUC
  "shapviz",       # visualizacao SHAP
  "ggplot2",       # graficos
  "corrplot"       # matriz de correlacao
)

log_info("Verificando %d pacotes base...", length(pacotes))
instalado <- vapply(pacotes, requireNamespace, logical(1), quietly = TRUE)
for (p in pacotes) {
  log_step(sprintf("%-14s %s", p, if (instalado[p]) "ja instalado" else "FALTANDO"),
           if (instalado[p]) "OK" else "WARN")
}

faltando <- pacotes[!instalado]
if (length(faltando) > 0) {
  log_info("Instalando: %s", paste(faltando, collapse = ", "))
  install.packages(faltando, quiet = TRUE)
  log_ok("Instalacao concluida.")
} else {
  log_ok("Todos os pacotes base ja estao instalados.")
}

# --- xgboost: versao fixada por compatibilidade com caret ------------------
log_subsection("xgboost (versao fixada 1.7.8.1)")
if (!requireNamespace("xgboost", quietly = TRUE) ||
    packageVersion("xgboost") != "1.7.8.1") {
  log_warn("xgboost ausente ou em versao incompativel. Instalando 1.7.8.1...")
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", quiet = TRUE)
  remotes::install_version("xgboost", version = "1.7.8.1",
                           repos = "https://cloud.r-project.org", quiet = TRUE)
}
log_ok("xgboost %s", as.character(packageVersion("xgboost")))

# --- fastshap: arquivado no CRAN, instala do archive ------------------------
log_subsection("fastshap (archive CRAN 0.1.1)")
if (!requireNamespace("fastshap", quietly = TRUE)) {
  log_warn("fastshap ausente. Instalando 0.1.1 a partir do archive...")
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", quiet = TRUE)
  remotes::install_version("fastshap", version = "0.1.1",
                           repos = "https://cloud.r-project.org",
                           quiet = TRUE, upgrade = "never")
}
log_ok("fastshap %s", as.character(packageVersion("fastshap")))

log_subsection("Ambiente")
log_kv(c("R", "Plataforma", "Seed global"),
       c(R.version.string, R.version$platform, as.character(PIPELINE_SEED)))

timer_end(t0, "Etapa 0")
