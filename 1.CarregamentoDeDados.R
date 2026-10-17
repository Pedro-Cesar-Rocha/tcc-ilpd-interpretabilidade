# =============================================================================
# 1.CarregamentoDeDados.R
# Carrega o dataset ILPD e realiza a analise exploratoria inicial (EDA).
# =============================================================================
# [Fund. 2.2 Compreensao e Exploracao dos Dados (sec:fund_estatistica)]
# Etapa 1 da Figura 2.1: Analise Exploratoria de Dados (AED). Os valores
# calculados aqui sao apenas descritivos e nao transformam os dados dos modelos
# (ver 2.3.2, vazamento de dados).
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(corrplot)
}))

log_section("ETAPA 1 - CARREGAMENTO DE DADOS E ANALISE EXPLORATORIA")
t0 <- timer_start()
ensure_dirs(c("data", "plots", "results"))

# --- Citacoes bibliograficas (linguagem R e pacotes utilizados) -------------
# Gera um arquivo .bib com as entradas BibTeX oficiais da linguagem R e dos
# pacotes tidyverse/corrplot, para serem referenciadas no capitulo de
# Metodologia (\cite{}) sem a necessidade de digitar as citacoes manualmente.
log_subsection("Gerando citacoes bibliograficas dos pacotes utilizados")
citacoes_bib <- c(
  "% Citacao da linguagem R",
  toBibtex(citation()),
  "",
  "% Citacao do pacote tidyverse",
  toBibtex(citation("tidyverse")),
  "",
  "% Citacao do pacote corrplot",
  toBibtex(citation("corrplot"))
)
writeLines(citacoes_bib, "results/citacoes.bib")
log_ok("Citacoes bibliograficas salvas em results/citacoes.bib")
log_info("Copie as chaves BibTeX geradas (@...{chave, ...) para o arquivo .bib do TCC e use \\cite{chave} no capitulo de Metodologia.")
log_info("Reprodutibilidade: a semente fixa PIPELINE_SEED = %d (definida em UtilsPipeline.R) garante resultados reproduziveis em toda a pipeline.", PIPELINE_SEED)

# --- Leitura -----------------------------------------------------------------
# [Fund. 2.2.1 Dominio do Problema (sec:fund_figado)] as variaveis sao os
#   marcadores laboratoriais (bilirrubinas, enzimas, proteinas) descritos no texto.
# [Fund. 2.2.2 Conceitos Basicos de Dados (sec:fund_conceitos)] cada linha e uma
#   observacao (paciente), cada coluna uma variavel; Dataset e a variavel-alvo.
arquivo <- "data/Indian Liver Patient Dataset (ILPD).csv"
colunas <- c("Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
             "Alkaline_Phosphotase", "Alamine_Aminotransferase",
             "Aspartate_Aminotransferase", "Total_Proteins",
             "Albumin", "AG_Ratio", "Dataset")

log_info("Lendo arquivo: %s", arquivo)
df <- read.csv(arquivo, header = FALSE, col.names = colunas)
log_ok("Dataset carregado: %d observacoes x %d variaveis", nrow(df), ncol(df))

# --- Estrutura ---------------------------------------------------------------
log_subsection("Estrutura das variaveis")
tipos <- data.frame(Variavel = names(df),
                    Tipo     = vapply(df, function(x) class(x)[1], character(1)),
                    row.names = NULL)
log_table(tipos)

# --- Valores ausentes --------------------------------------------------------
log_subsection("Valores ausentes")
# [Fund. 2.3.1 Valores Ausentes e Imputacao (sec:fund_ausentes)] aqui apenas
#   identificados; a imputacao pela mediana ocorre na etapa 2.
na_por_coluna <- colSums(is.na(df))
if (sum(na_por_coluna) == 0) {
  log_ok("Nenhum valor ausente encontrado.")
} else {
  for (v in names(na_por_coluna)[na_por_coluna > 0]) {
    log_warn("%-28s %d ausentes (%.2f%%)", v, na_por_coluna[v], 100 * na_por_coluna[v] / nrow(df))
  }
}

# --- Duplicatas --------------------------------------------------------------
# [Fund. 2.3.2 Registros Duplicados e Vazamento de Dados (sec:fund_vazamento)]
#   aqui apenas contados; a remocao ocorre na etapa 2.
n_dup <- sum(duplicated(df))
if (n_dup > 0) log_warn("%d linhas duplicadas detectadas (mantidas nesta etapa).", n_dup) else log_ok("Sem linhas duplicadas.")

# --- Variavel alvo -----------------------------------------------------------
# [Fund. 2.2.2 Conceitos Basicos (sec:fund_conceitos)] classe positiva/negativa,
#   base desbalanceada, classe majoritaria/minoritaria e prevalencia.
log_subsection("Distribuicao da variavel alvo (Dataset: 1 = doente, 2 = saudavel)")
tab_alvo <- table(df$Dataset)
log_kv(paste0("Classe ", names(tab_alvo)),
       sprintf("%d (%.1f%%)", tab_alvo, 100 * prop.table(tab_alvo)))
razao <- max(tab_alvo) / min(tab_alvo)
log_info("Razao de desbalanceamento: %.2f : 1", razao)

# --- Genero ------------------------------------------------------------------
# [Fund. 2.7 Vies Algoritmico e Equidade em Saude (sec:fund_vies)] grupo
#   sub-representado (mulheres), motivo da analise estratificada por sexo.
log_subsection("Distribuicao por genero")
tab_gen <- table(df$Gender, df$Dataset, dnn = c("Gender", "Dataset"))
log_table(tab_gen)

# --- Estatisticas descritivas ------------------------------------------------
# [Fund. 2.2.3 Estatisticas Descritivas e Assimetria (sec:fund_assimetria)]
#   media, desvio-padrao, mediana e coeficiente de assimetria b1 (Eq. 2.1,
#   eq:assimetria), calculado na linha 'Assimetria' abaixo.
log_subsection("Estatisticas descritivas das variaveis numericas")
num_cols <- names(df)[sapply(df, is.numeric) & names(df) != "Dataset"]
desc <- df %>%
  select(all_of(num_cols)) %>%
  pivot_longer(everything(), names_to = "Variavel", values_to = "Valor") %>%
  group_by(Variavel) %>%
  summarise(Media  = mean(Valor, na.rm = TRUE),
            DP     = sd(Valor, na.rm = TRUE),
            Min    = min(Valor, na.rm = TRUE),
            Mediana = median(Valor, na.rm = TRUE),
            Max    = max(Valor, na.rm = TRUE),
            Assimetria = mean(((Valor - mean(Valor, na.rm = TRUE)) / sd(Valor, na.rm = TRUE))^3, na.rm = TRUE),
            .groups = "drop") %>%
  as.data.frame()
log_table(desc, digits = 3)
salvar_csv(desc, "results/estatisticas_descritivas.csv")

# [Fund. 2.2.3] limiar |b1| > 2 adotado como convencao (Curran et al.).
assimetricas <- desc$Variavel[abs(desc$Assimetria) > 2]
if (length(assimetricas) > 0) {
  log_warn("Variaveis com forte assimetria (|skew| > 2): %s", paste(assimetricas, collapse = ", "))
  log_info("Sera aplicada transformacao log1p nessas variaveis na etapa 2.")
}

# --- Comparacao das medias por classe ---------------------------------------
# [Fund. 2.2.5 Testes de Hipotese e Comparacoes Multiplas (sec:fund_welch)]
#   teste t de Welch (variancias desiguais), valor-p e nivel de significancia 0,05.
log_subsection("Media das variaveis por classe (teste t de Welch)")
comp <- lapply(num_cols, function(v) {
  g1 <- df[[v]][df$Dataset == 1]; g2 <- df[[v]][df$Dataset == 2]
  tt <- t.test(g1, g2)
  data.frame(Variavel = v, Media_Doente = mean(g1, na.rm = TRUE),
             Media_Saudavel = mean(g2, na.rm = TRUE), p_valor = tt$p.value)
}) %>% bind_rows()
comp$Significativo <- ifelse(comp$p_valor < 0.05, "sim", "nao")
log_table(comp, digits = 3)
salvar_csv(comp, "results/comparacao_medias_por_classe.csv")

significativas <- comp$Variavel[comp$Significativo == "sim"]
nao_significativas <- comp$Variavel[comp$Significativo == "nao"]
log_info("Variaveis com diferenca significativa (p < 0.05): %s",
         if (length(significativas) > 0) paste(significativas, collapse = ", ") else "nenhuma")
log_info("Variaveis sem diferenca significativa (p >= 0.05): %s",
         if (length(nao_significativas) > 0) paste(nao_significativas, collapse = ", ") else "nenhuma")

# --- Correlacao --------------------------------------------------------------
# [Fund. 2.2.4 Correlacao e Multicolinearidade (sec:fund_correlacao)]
#   coeficiente de Pearson (Eq. 2.2, eq:pearson) e limiar |r| > 0,7 de correlacao forte.
log_subsection("Matriz de correlacao (Pearson)")
mat_cor <- cor(df[, num_cols], use = "complete.obs")
pares <- which(abs(mat_cor) > 0.7 & upper.tri(mat_cor), arr.ind = TRUE)
if (nrow(pares) > 0) {
  pares_df <- data.frame(
    Variavel_1 = rownames(mat_cor)[pares[, 1]],
    Variavel_2 = colnames(mat_cor)[pares[, 2]],
    r = round(mat_cor[pares], 3)
  )
  pares_df <- pares_df[order(-abs(pares_df$r)), ]
  for (i in seq_len(nrow(pares_df))) {
    log_warn("Alta correlacao: %s x %s = %.3f",
             pares_df$Variavel_1[i], pares_df$Variavel_2[i], pares_df$r[i])
  }
  salvar_csv(pares_df, "results/pares_alta_correlacao.csv")
  log_info("Pares com |r| > 0.7 salvos em results/pares_alta_correlacao.csv (use para preencher [PARES] no cap4-metodologia.tex).")
} else {
  log_ok("Nenhum par com |r| > 0.7.")
}

png("plots/eda_correlacao.png", width = 1000, height = 900, res = 130)
corrplot(mat_cor, method = "color", type = "upper", addCoef.col = "black",
         number.cex = 0.7, tl.cex = 0.8, tl.col = "black",
         title = "Matriz de correlacao - ILPD", mar = c(0, 0, 2, 0))
invisible(dev.off())
log_ok("Grafico salvo: plots/eda_correlacao.png")

# --- Graficos exploratorios -------------------------------------------------
p_alvo <- ggplot(df, aes(x = factor(Dataset, labels = c("Doente", "Saudavel")), fill = factor(Dataset))) +
  geom_bar(show.legend = FALSE) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4) +
  labs(title = "Distribuicao da variavel alvo", x = NULL, y = "Frequencia") +
  theme_minimal()
salvar_plot(p_alvo, "plots/eda_distribuicao_alvo.png", width = 6, height = 4)

# [Fund. 2.2.3] boxplots: valores extremos (outliers) e assimetria por classe.
p_box <- df %>%
  mutate(Classe = factor(Dataset, labels = c("Doente", "Saudavel"))) %>%
  select(all_of(num_cols), Classe) %>%
  pivot_longer(-Classe, names_to = "Variavel", values_to = "Valor") %>%
  ggplot(aes(x = Classe, y = Valor, fill = Classe)) +
  geom_boxplot(outlier.size = 0.6, show.legend = FALSE) +
  facet_wrap(~Variavel, scales = "free_y", ncol = 5) +
  labs(title = "Distribuicao das variaveis por classe (escala original)", x = NULL) +
  theme_minimal(base_size = 9)
salvar_plot(p_box, "plots/eda_boxplots_por_classe.png", width = 12, height = 6)

salvar_rds(df, "data/df_raw.rds")
timer_end(t0, "Etapa 1")
