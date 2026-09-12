```
INSTITUTO FEDERAL DE EDUCAÇÃO,
CIÊNCIA E TECNOLOGIA GOIANO
Campus Urutaí
NÚCLEO DE INFORMÁTICA
Sistemas de Informação
```
### PROPOSTA DO TRABALHO DE CURSO

## Além da Acurácia: Análise de Interpretabilidade e

## Desempenho de Modelos de Black-Box na Predição de

## Doenças Hepáticas (ILPD)

```
Pedro Cesar Rocha
```
```
Junio Cesar de Lima
Orientador
```
```
PARECER, ASSINATURA DO AVALIADOR
```
```
Urutaí, 6 de março de 2026
```

## Identificação

- Título do Projeto: Além da Acurácia: Análise de Interpretabilidade e Desempenho de Mo-
    delos de Black-Box na Predição de Doenças Hepáticas (ILPD)
- Modalidade: Monografia
- Estudante: Pedro Cesar Rocha
- Orientador(a): Orientador
- Instituição: Instituto Federal Goiano – Campus Urutaí
- Local e Data: Urutaí, 6 de março de 2026

## Introdução

As doenças hepáticas representam um dos maiores desafios de saúde pública em nível glo-
bal, sendo frequentemente caracterizadas por sua progressão silenciosa. Muitas vezes, os sinto-
mas clínicos tornam-se evidentes apenas em estágios avançados da patologia, o que compromete
significativamente o prognóstico e a eficácia das intervenções terapêuticas. Além da progressão
biológica, a epidemiologia dessas doenças apresenta disparidades significativas entre os gêne-
ros, o que exige um olhar atento para possíveis vieses diagnósticos em modelos automatizados
[1]. Nesse cenário, o diagnóstico precoce por meio da análise de exames laboratoriais de rotina
desempenha um papel fundamental na mitigação de danos orgânicos irreversíveis.
Com o advento da Ciência de Dados e da Inteligência Artificial (IA), algoritmos de Apren-
dizado de Máquina (Machine Learning - ML) têm sido amplamente aplicados para auxiliar na
detecção de diversas condições médicas. Modelos preditivos de alta complexidade, baseados
em árvores de decisão e ensembles (como Random Forest e XGBoost), demonstram notável
capacidade técnica ao identificar padrões não-lineares sutis em conjuntos de dados clínicos, a
exemplo do Indian Liver Patient Dataset (ILPD). Recentemente, pesquisas destacaram que a
utilização de múltiplas técnicas de Ensemble Learning aliadas a um pré-processamento rigoroso
pode superar significativamente os métodos de classificação tradicionais [2, 3]. No entanto, a
transição e a aceitação dessas tecnologias no ambiente clínico real esbarram em um obstáculo
metodológico e ético: a natureza "caixa-preta"(black-box) desses algoritmos.
Na área da saúde, a adoção de sistemas de suporte à decisão clínica exige mais do que um
alto índice de acertos (acurácia); exige transparência, confiança e interpretabilidade. Profissio-
nais médicos necessitam compreender por que um modelo computacional classificou um paci-
ente como portador de uma doença hepática antes de definir ou alterar um plano de tratamento.
É para solucionar essa lacuna que emerge a Inteligência Artificial Explicável (XAI - Explaina-
ble AI), um subcampo dedicado a tornar as predições de modelos complexos semanticamente
compreensíveis para seres humanos.


Diante da necessidade de aliar precisão preditiva à confiabilidade médica, esta proposta de
trabalho aborda a predição de doenças hepáticas transcendendo as avaliações tradicionais de
desempenho. O estudo visa integrar o poder preditivo de algoritmos de estado da arte, cuja
eficácia em dados de pacientes indianos foi recentemente reafirmada pela literatura [3], com
técnicas avançadas de interpretabilidade, especificamente o método SHAP (SHapley Additive
exPlanations). Dessa forma, busca-se não apenas identificar o modelo que melhor lida com as
características inerentes aos dados laboratoriais, mas também mapear e explicar o peso de cada
biomarcador nas decisões automatizadas, aproximando a eficiência da Inteligência Artificial da
segurança exigida pela prática médica.

## Objetivos

### Objetivo Geral

Avaliar e comparar o desempenho preditivo e a interpretabilidade de diferentes algoritmos
de Aprendizado de Máquina na classificação de pacientes com doenças hepáticas, utilizando o
Indian Liver Patient Dataset (ILPD) e aplicando técnicas de Inteligência Artificial Explicável
(XAI) para garantir a transparência no suporte à decisão médica.

### Objetivos Específicos

```
Para alcançar o objetivo geral, foram definidos os seguintes objetivos específicos:
```
1. Realizar a análise exploratória e o pré-processamento da base de dados ILPD, incluindo
    o tratamento de valores ausentes, a normalização de variáveis contínuas e a aplicação de
    técnicas de balanceamento de classes (como o SMOTE).
2. Implementar e treinar modelos de classificação de diferentes níveis de complexidade, es-
    tabelecendo um modelo linear interpretável (ex: Regressão Logística) como baseline para
    comparação com modelos de caixa-preta baseados em ensembles (ex: Random Forest e
    XGBoost ou LightGBM).
3. Otimizar os hiperparâmetros dos algoritmos selecionados visando maximizar o desempe-
    nho preditivo, com especial atenção à métrica de Recall (Sensibilidade), dada a necessi-
    dade de minimizar falsos negativos no contexto clínico.
4. Avaliar o desempenho dos modelos computacionais por meio de métricas estatísticas ri-
    gorosas, tais como Acurácia, Precisão, F1-Score e a área sob a curva ROC (ROC-AUC).
5. Aplicar o método SHAP (SHapley Additive exPlanations) aos modelos de melhor desem-
    penho para extrair a importância dos atributos (feature importance) e explicar o impacto


```
individual de cada biomarcador nas predições, investigando como variáveis demográficas
como o gênero influenciam a decisão do algoritmo [1].
```
6. Analisar o trade-off (compromisso) entre o poder preditivo dos modelos e sua capacidade
    de explicação, discutindo qual abordagem oferece a melhor viabilidade e segurança para
    adoção em um ambiente médico real.

## Justificativa

A utilização de sistemas de inteligência artificial no suporte ao diagnóstico médico tem
avançado de forma exponencial, impulsionada pela disponibilidade de grandes bases de dados e
pelo aumento do poder computacional. No entanto, a implementação prática dessas ferramentas
em hospitais e clínicas enfrenta uma barreira crítica conhecida como o "dilema da caixa-preta".
Modelos altamente precisos, como os baseados em gradient boosting, são frequentemente opa-
cos, o que gera uma lacuna de confiança e insegurança ética para o profissional de saúde, que
detém a responsabilidade final pela conduta terapêutica e pelo bem-estar do paciente.
Nesse contexto, a realização deste trabalho justifica-se pela necessidade premente de aliar
a alta performance estatística à transparência explicativa. Ao utilizar o Indian Liver Patient
Dataset (ILPD), esta pesquisa não se limita a buscar o algoritmo com a maior taxa de acerto
nominal, mas propõe uma metodologia que revela a lógica intrínseca por trás da decisão auto-
matizada. A literatura recente destaca que a combinação de técnicas de balanceamento de dados
com algoritmos de ensemble é o caminho para reduzir drasticamente erros diagnósticos em do-
enças hepáticas [3]. Complementarmente, a aplicação do método SHAP permite transformar
esses dados brutos em conhecimento clínico interpretável, identificando quais biomarcadores
hepáticos — como a bilirrubina, a albumina ou as transaminases — estão pesando mais para o
diagnóstico de cada paciente individualmente. Esse processo é vital para que o modelo deixe
de ser um oráculo inquestionável e passe a ser uma ferramenta de apoio fundamentada.
Além da contribuição técnica no campo da Ciência de Dados, o projeto possui relevân-
cia direta para a segurança do paciente e para a eficiência do sistema de saúde. Ao focar na
otimização do Recall e na explicação das variáveis determinantes, o modelo reduz o risco de
negligenciar casos críticos, mitigando a ocorrência de falsos negativos. A pesquisa também se
posiciona na vanguarda acadêmica ao explorar a Inteligência Artificial Explicável (XAI) como
uma ferramenta para auditar algoritmos médicos e prevenir a perpetuação de vieses sistêmicos
relacionados ao sexo do paciente [1]. Essa abordagem busca humanizar a computação avan-
çada e garantir que a tecnologia atue como uma extensão confiável e equânime da inteligência
humana.


## Metodologia

A metodologia proposta para este trabalho segue o processo de Descoberta de Conheci-
mento em Bases de Dados (KDD), estruturada em fases que abrangem desde a preparação dos
dados até a interpretação dos resultados por meio de técnicas de explicabilidade. O ambiente
de desenvolvimento será fundamentado na linguagem de programação Python, utilizando bibli-
otecas consolidadas como Scikit-Learn para modelagem clássica, XGBoost ou CatBoost para
modelos de ensemble, e a biblioteca SHAP para a análise de interpretabilidade.

### Fase 1: Preparação e Pré-processamento dos Dados

A primeira etapa consistirá na análise exploratória do Indian Liver Patient Dataset (ILPD),
visando identificar padrões, outliers e inconsistências. O tratamento de dados faltantes será
realizado, com foco especial na variável de proteínas, onde serão aplicadas técnicas de impu-
tação (média ou mediana) conforme a distribuição dos dados, seguindo recomendações recen-
tes de abordagens de pré-processamento aprimorado para este dataset específico [2]. Para as
variáveis categóricas, como o gênero do paciente, será aplicada a codificação binária (Label
Encoding). Dada a natureza das variáveis laboratoriais, que possuem escalas distintas, será
aplicado o escalonamento de atributos (StandardScaler ou MinMaxScaler) para garantir que
algoritmos sensíveis à escala, como o SVM, operem de forma eficiente. Por fim, para mitigar o
desbalanceamento entre as classes de pacientes doentes e saudáveis, será utilizada a técnica de
sobreamostragem sintética SMOTE (Synthetic Minority Over-sampling Technique), abordagem
que tem se mostrado fundamental para elevar a sensibilidade e a precisão do diagnóstico no
ILPD [3].

### Fase 2: Desenvolvimento e Treinamento dos Modelos

Os dados pré-processados serão divididos em conjuntos de treino e teste, utilizando uma
proporção de 80/20, com a aplicação de validação cruzada (k-fold cross-validation) para ga-
rantir a robustez dos resultados e evitar o overfitting. Serão implementados três modelos de
diferentes categorias para fins comparativos, priorizando algoritmos de Ensemble que demons-
traram desempenho superior em estudos recentes de detecção de doenças hepáticas [2]:

1. Regressão Logística: Atuará como o modelo de baseline devido à sua simplicidade e
    natureza intrinsecamente interpretável.
2. Random Forest: Um modelo de florestas aleatórias para capturar relações não-lineares
    através de múltiplos estimadores baseados em Bagging.
3. XGBoost ou LightGBM: Algoritmos de Gradient Boosting de alto desempenho para
    representar o estado da arte em modelos de "caixa-preta"e capturar padrões complexos
    nos biomarcadores [3].


A otimização de hiperparâmetros será realizada por meio de busca bayesiana ou Random Se-
arch, visando o melhor equilíbrio entre as métricas de desempenho.

### Fase 3: Avaliação de Desempenho e Explicabilidade (XAI)

Os modelos serão avaliados por meio de uma matriz de confusão e métricas estatísticas:
Acurácia, Precisão, F1-Score e, primordialmente, o Recall, para assegurar a detecção do maior
número possível de casos positivos. A análise da Curva ROC e da Área Sob a Curva (AUC)
será utilizada para medir a capacidade de separação entre as classes.
Após a definição do modelo de melhor desempenho, será aplicada a técnica SHAP (SHapley
Additive exPlanations), baseada na teoria dos jogos cooperativos. Esta etapa permitirá decom-
por a predição final para identificar a contribuição individual de cada atributo. Serão gerados
gráficos de summary plot, para visualizar a importância global das variáveis, e force plots, para
explicar predições individuais, permitindo correlacionar as decisões da máquina com o conhe-
cimento médico estabelecido.

## Cronograma

As atividades planejadas para a execução deste trabalho estão dividas em quatro fases prin-
cipais, descritas a seguir:

```
0.1 Fase 1: Fundamentação e Planejamento
```
```
a) Pesquisa bibliográfica e estado da arte (XAI e Doenças Hepáticas).
b) Estudo detalhado do dataset ILPD e seleção de ferramentas Python.
c) Elaboração e entrega da proposta formal de TCC.
```
```
0.2 Fase 2: Desenvolvimento Técnico
```
```
a) Pré-processamento, limpeza e balanceamento de dados (SMOTE).
b) Implementação e treinamento dos modelos (RF, XGBoost, Regressão Logística).
c) Otimização de hiperparâmetros e ajuste fino dos modelos.
```
```
0.3 Fase 3: Análise e Interpretabilidade
```
```
a) Avaliação de desempenho e extração de métricas (Recall, AUC, F1).
b) Aplicação da técnica SHAP e análise de explicabilidade.
c) Estudo de viés e ética diagnóstica (estratificação por gênero).
```
```
0.4 Fase 4: Finalização
```

```
a) Redação final da monografia e análise dos resultados.
```
b) Revisão gramatical, formatação ABNT e Defesa do TCC.

```
Tabela 1: Detalhamento das Atividades por Mês
Atividade MAR ABR MAI JUN JUL AGO SET NOV
a) • • •
b) • •
c) • • •
a) • •
b) • • •
c) • •
a) • •
b) • • •
c) • •
a) •
b) •
```

# Referências Bibliográficas

[1] STRAW, I.; WU, H. Investigating for bias in healthcare algorithms: a sex-stratified analysis
of supervised machine learning models in liver disease prediction. BMJ Health & Care In-
formatics, v. 29, n. 1, e100457, 2022. Disponível em: https://doi.org/10.1136/
bmjhci-2021-100457.

[2] MD, A. Q. et al. Enhanced Preprocessing Approach Using Ensemble Machine Learning
Algorithms for Detecting Liver Disease. Biomedicines, v. 11, n. 2, p. 581, 2023. Disponível
em: https://doi.org/10.3390/biomedicines11020581.

[3] RANI, R. et al. Enhancing liver disease diagnosis with hybrid SMOTE-ENN balanced ma-
chine learning models—an empirical analysis of Indian patient liver disease datasets. Fron-
tiers in Medicine, v. 12, art. 1502749, 2025. Disponível em: https://doi.org/10.
3389/fmed.2025.1502749.


