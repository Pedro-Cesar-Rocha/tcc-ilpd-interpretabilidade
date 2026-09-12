# =============================================================================
# ExecutarTudo.R
# Orquestra a pipeline completa, cronometra cada etapa, trata erros e grava
# uma copia integral do console em logs/pipeline_<timestamp>.log.
# =============================================================================
project_dir <- getwd()
source(file.path(project_dir, "UtilsPipeline.R"))

dir.create("logs", showWarnings = FALSE)
log_path <- file.path("logs", format(Sys.time(), "pipeline_%Y%m%d_%H%M%S.log"))
sink(log_path, split = TRUE)   # espelha tudo no console e no arquivo

log_section("PIPELINE - CLASSIFICACAO DE DOENCA HEPATICA (ILPD)")
log_info("Diretorio de trabalho: %s", project_dir)
log_info("Log completo em: %s", log_path)
log_info("Inicio: %s", format(Sys.time(), "%d/%m/%Y %H:%M:%S"))

scripts <- c(
  "0.InstalarPacotes.R",
  "1.CarregamentoDeDados.R",
  "2.PreProcessamento.R",
  "3.Balanceamento.R",
  "4.Modelos.R",
  "5.Avaliacao.R",
  "6.SHAP.R"
)

tempos <- numeric(0)
t_total <- timer_start()

resultado <- tryCatch({
  for (script in scripts) {
    caminho <- file.path(project_dir, script)
    if (!file.exists(caminho)) stop(sprintf("Arquivo nao encontrado: %s", caminho))

    ti <- Sys.time()
    source(caminho, local = FALSE)
    tempos[script] <- as.numeric(difftime(Sys.time(), ti, units = "secs"))
  }
  "ok"
}, error = function(e) {
  log_error("Pipeline interrompida: %s", conditionMessage(e))
  "erro"
})

log_section("RESUMO DA EXECUCAO")
if (length(tempos) > 0) {
  log_info("Tempo por etapa:")
  log_kv(names(tempos), sprintf("%6.1f s", tempos))
}
log_info("Tempo total: %.1f s", as.numeric(difftime(Sys.time(), t_total, units = "secs")))

if (resultado == "ok") {
  log_info("Artefatos gerados:")
  for (d in c("data", "models", "results", "plots")) {
    arquivos <- list.files(d)
    log_info("  %s/ (%d arquivos): %s", d, length(arquivos), paste(arquivos, collapse = ", "))
  }
  log_ok("PIPELINE CONCLUIDA COM SUCESSO")
} else {
  log_error("PIPELINE FINALIZADA COM ERRO - verifique o log acima.")
}

sink()
if (resultado != "ok") stop("Execucao interrompida por erro.")
