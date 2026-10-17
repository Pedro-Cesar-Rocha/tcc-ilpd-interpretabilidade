# =============================================================================
# 3.Balanceamento.R
# Demonstra e valida o SMOTE sobre o conjunto de treino completo (diagnostico
# e grafico). Na etapa 4 o MESMO procedimento e aplicado dentro de cada fold da
# validacao cruzada (via caret::trainControl(sampling = smote_caret)), de modo
# que as particoes de validacao nunca contenham amostras sinteticas. O conjunto
# de teste permanece com a distribuicao real das classes.
# =============================================================================
# [Fund. 2.3.4 Desbalanceamento de Classes e SMOTE (sec:fund_smote)] Etapa 2 da
# Figura 2.1 (ultimo item: SMOTE so no treino). A implementacao esta em
# UtilsPipeline.R (aplicar_smote).
source("UtilsPipeline.R")
suppressWarnings(suppressPackageStartupMessages({
  library(tidyverse)
  library(smotefamily)
}))

log_section("ETAPA 3 - BALANCEAMENTO DE CLASSES (SMOTE - DIAGNOSTICO NO TREINO)")
t0 <- timer_start()

treino <- readRDS("data/df_proc.rds")
log_info("Conjunto de treino carregado: %d x %d", nrow(treino), ncol(treino))

# --- Distribuicao antes -------------------------------------------------------
# [Fund. 2.3.4] desbalanceamento: classe majoritaria x minoritaria.
log_subsection("Distribuicao antes do SMOTE")
tab_antes <- table(treino$Diagnostico)
log_kv(paste0("Classe ", names(tab_antes)),
       sprintf("%d (%.1f%%)", tab_antes, 100 * prop.table(tab_antes)))
log_info("Razao majoritaria/minoritaria: %.2f : 1", max(tab_antes) / min(tab_antes))

# --- SMOTE --------------------------------------------------------------------
# [Fund. 2.3.4] oversampling da classe minoritaria com k = 5 vizinhos mais proximos.
log_subsection("Executando SMOTE (K = 5, dup_size = 0 -> balanceamento automatico)")
set.seed(PIPELINE_SEED)
res <- aplicar_smote(treino %>% select(-Diagnostico), treino$Diagnostico, K = 5)
df_bal <- cbind(res$x, Diagnostico = res$y)
n_sinteticas <- res$n_sinteticas
sinteticas <- res$sinteticas
log_ok("SMOTE concluido: %d amostras sinteticas geradas para a classe minoritaria.", n_sinteticas)

# --- Distribuicao depois ------------------------------------------------------
log_subsection("Distribuicao apos o SMOTE")
tab_depois <- table(df_bal$Diagnostico)
log_kv(paste0("Classe ", names(tab_depois)),
       sprintf("%d (%.1f%%)", tab_depois, 100 * prop.table(tab_depois)))
log_info("Razao majoritaria/minoritaria: %.2f : 1", max(tab_depois) / min(tab_depois))
log_info("Tamanho do treino: %d -> %d observacoes", nrow(treino), nrow(df_bal))

# --- Verificacao de integridade ---------------------------------------------
log_subsection("Verificacao das amostras sinteticas")
minoritaria <- names(tab_antes)[which.min(tab_antes)]
comp <- data.frame(
  Variavel        = names(select(df_bal, -Diagnostico)),
  Media_Minoritaria_Original = round(colMeans(select(treino[treino$Diagnostico == minoritaria, ], -Diagnostico)), 3),
  Media_Sintetica = round(colMeans(sinteticas), 3),
  row.names = NULL
)
log_table(comp, digits = 3)
log_info("Amostras sinteticas devem ter medias proximas as da classe minoritaria original ('%s').", minoritaria)
# [Fund. 2.3.4] variavel binaria Gender: valores intermediarios corrigidos,
#   mesma ideia da variante SMOTE-NC (exemplo do ILPD no texto).
categorias_gender <- sort(unique(treino$Gender))   # valores padronizados: Feminino < 0 < Masculino
log_info("Gender nas sinteticas: %d amostra(s) com valor intermediario (pacientes de origem de sexos diferentes) devolvida(s) a categoria mais proxima",
         res$n_corrigidas)
log_info("Gender nas sinteticas: %d Feminino, %d Masculino | apenas categorias validas (sem valores intermediarios): %s",
         sum(sinteticas$Gender == categorias_gender[1]), sum(sinteticas$Gender == categorias_gender[2]),
         if (all(sinteticas$Gender %in% categorias_gender)) "sim" else "NAO")

# --- Grafico comparativo ------------------------------------------------------
df_plot <- bind_rows(
  data.frame(Momento = "Antes",  Classe = names(tab_antes),  n = as.vector(tab_antes)),
  data.frame(Momento = "Depois", Classe = names(tab_depois), n = as.vector(tab_depois))
) %>% mutate(Momento = factor(Momento, levels = c("Antes", "Depois")))

p <- ggplot(df_plot, aes(x = Classe, y = n, fill = Classe)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = n), vjust = -0.4) +
  facet_wrap(~Momento) +
  labs(title = "Distribuicao das classes no treino: antes e depois do SMOTE",
       x = "Classe", y = "Observacoes") +
  theme_minimal()
salvar_plot(p, "plots/smote_antes_depois.png", width = 7, height = 4)

salvar_rds(df_bal, "data/df_balanceado.rds")
log_info("Nota: a etapa 4 NAO usa este arquivo diretamente; o SMOTE e reaplicado dentro de cada fold da CV.")
timer_end(t0, "Etapa 3")
