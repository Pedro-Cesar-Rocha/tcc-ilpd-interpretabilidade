# =============================================================================
# 1.CarregamentoDeDados.R
# Carrega o dataset ILPD e realiza a analise exploratoria inicial (EDA).
# =============================================================================
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(corrplot)
}))

log_section("ETAPA 1 - CARREGAMENTO DE DADOS E ANALISE EXPLORATORIA")
t0 <- timer_start()
ensure_dirs(c("data", "plots", "results"))

# --- Leitura -----------------------------------------------------------------
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
na_por_coluna <- colSums(is.na(df))
if (sum(na_por_coluna) == 0) {
  log_ok("Nenhum valor ausente encontrado.")
} else {
  for (v in names(na_por_coluna)[na_por_coluna > 0]) {
    log_warn("%-28s %d ausentes (%.2f%%)", v, na_por_coluna[v], 100 * na_por_coluna[v] / nrow(df))
  }
}

# --- Duplicatas --------------------------------------------------------------
n_dup <- sum(duplicated(df))
if (n_dup > 0) log_warn("%d linhas duplicadas detectadas (mantidas nesta etapa).", n_dup) else log_ok("Sem linhas duplicadas.")

# --- Variavel alvo -----------------------------------------------------------
log_subsection("Distribuicao da variavel alvo (Dataset: 1 = doente, 2 = saudavel)")
tab_alvo <- table(df$Dataset)
log_kv(paste0("Classe ", names(tab_alvo)),
       sprintf("%d (%.1f%%)", tab_alvo, 100 * prop.table(tab_alvo)))
razao <- max(tab_alvo) / min(tab_alvo)
log_info("Razao de desbalanceamento: %.2f : 1", razao)

# --- Genero ------------------------------------------------------------------
log_subsection("Distribuicao por genero")
tab_gen <- table(df$Gender, df$Dataset, dnn = c("Gender", "Dataset"))
log_table(tab_gen)

# --- Estatisticas descritivas ------------------------------------------------
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

assimetricas <- desc$Variavel[abs(desc$Assimetria) > 2]
if (length(assimetricas) > 0) {
  log_warn("Variaveis com forte assimetria (|skew| > 2): %s", paste(assimetricas, collapse = ", "))
  log_info("Sera aplicada transformacao log1p nessas variaveis na etapa 2.")
}

# --- Comparacao das medias por classe ---------------------------------------
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

# --- Correlacao --------------------------------------------------------------
log_subsection("Matriz de correlacao (Pearson)")
mat_cor <- cor(df[, num_cols], use = "complete.obs")
pares <- which(abs(mat_cor) > 0.7 & upper.tri(mat_cor), arr.ind = TRUE)
if (nrow(pares) > 0) {
  for (i in seq_len(nrow(pares))) {
    log_warn("Alta correlacao: %s x %s = %.3f",
             rownames(mat_cor)[pares[i, 1]], colnames(mat_cor)[pares[i, 2]],
             mat_cor[pares[i, 1], pares[i, 2]])
  }
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
