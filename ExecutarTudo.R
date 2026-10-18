# =============================================================================
# ExecutarTudo.R
# Orquestra a pipeline completa, cronometra cada etapa, trata erros e grava
# uma copia integral do console em logs/pipeline_<timestamp>.log.
# =============================================================================
# -----------------------------------------------------------------------------
# MAPA ENTRE O CODIGO E A FUNDAMENTACAO TEORICA (Capitulo 2 da monografia)
# A pipeline implementa o fluxo da Figura 2.1 (secao 2.1, \label{fig:fund_kdd}).
# Cada script corresponde a uma ou mais etapas dessa figura:
#   0.InstalarPacotes.R     -> 2.8 Ferramentas Computacionais
#   1.CarregamentoDeDados.R -> Etapa 1, 2.2 Compreensao e Exploracao dos Dados
#   2.PreProcessamento.R    -> Etapa 2, 2.3 Preparacao dos Dados (2.3.1 a 2.3.3)
#   3.Balanceamento.R       -> Etapa 2, 2.3.4 Desbalanceamento de Classes e SMOTE
#   4.Modelos.R             -> Etapa 3, 2.4 Modelagem; Etapa 4, 2.5.1, 2.5.2 e 2.5.6
#   5.Avaliacao.R           -> Etapa 4, 2.5.3 a 2.5.5; e 2.7 Vies Algoritmico
#   6.SHAP.R                -> Etapa 5, 2.6 Interpretabilidade; e 2.7 Vies Algoritmico
#   UtilsPipeline.R         -> semente (2.1), SMOTE por fold (2.3.4) e LightGBM (2.4.5)
# Dentro de cada script, comentarios no formato [Fundamentação Teórica - X.Y ...] indicam a secao
# (ou equacao) da Fundamentacao em que o conceito usado naquele trecho e explicado.
# Numeracao conforme a versao atual da monografia; entre parenteses, o \label.
# -----------------------------------------------------------------------------
# MAPA ENTRE O CODIGO E A METODOLOGIA (Capitulo 3 da monografia)
#   1.CarregamentoDeDados.R -> 3.1 Base de Dados e 3.2 Analise Exploratoria de Dados
#   2.PreProcessamento.R    -> 3.3.1 a 3.3.3 (duplicatas, log, divisao, imputacao,
#                              padronizacao)
#   3.Balanceamento.R       -> 3.3.4 Reamostragem com SMOTE (diagnostico no treino)
#   4.Modelos.R             -> 3.4 Modelos Preditivos e 3.5 Treinamento e Validacao
#                              Cruzada
#   5.Avaliacao.R           -> 3.6 Avaliacao de Desempenho (3.6.1 a 3.6.4)
#   6.SHAP.R                -> 3.7 Interpretabilidade com SHAP (3.7.1 a 3.7.4)
#   UtilsPipeline.R         -> SMOTE por fold (3.3.4) e LightGBM (3.4.4)
# Dentro de cada script, comentarios no formato [Metodologia - X.Y ...] indicam a secao da
# Metodologia que descreve o que aquele trecho faz (entre parenteses, o \label;
# Tab./Figura indicam a tabela ou figura produzida a partir dele).
# -----------------------------------------------------------------------------
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
