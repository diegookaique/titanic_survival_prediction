# Previsão de Sobrevivência no Titanic: Análise e Otimização com XGBoost e LightGBM"

## Visão Geral do Projeto
Este projeto foca na aplicação de técnicas de Machine Learning para prever a sobrevivência de passageiros no famoso desastre do Titanic. Utilizando o dataset disponível no Kaggle, desenvolvemos e comparamos dois modelos de boosting, XGBoost e LightGBM, com o objetivo de identificar o modelo com melhor desempenho para esta tarefa de classificação binária. O pipeline inclui pré-processamento de dados, engenharia de features, balanceamento de classes, escalonamento e otimização de hiperparâmetros.

## Objetivo
O principal objetivo deste projeto é desenvolver e comparar modelos de Machine Learning (XGBoost e LightGBM) para prever a sobrevivência de passageiros no desastre do Titanic. A meta é aplicar técnicas de pré-processamento de dados, engenharia de features, balanceamento de classes e otimização de hiperparâmetros para alcançar a maior acurácia possível, superando os benchmarks e demonstrando a capacidade preditiva de modelos de boosting.

## Dataset
O conjunto de dados utilizado é o clássico "Titanic - Machine Learning from Disaster" do Kaggle, que contém informações sobre passageiros a bordo do RMS Titanic, incluindo se eles sobreviveram ou não. As variáveis principais incluem:
- `PassengerId`: Identificador único do passageiro.
- `Survived`: Variável alvo (0 = Não Sobreviveu, 1 = Sobreviveu).
- `Pclass`: Classe do passageiro (1ª, 2ª ou 3ª).
- `Name`: Nome do passageiro.
- `Sex`: Sexo do passageiro.
- `Age`: Idade do passageiro.
- `SibSp`: Número de irmãos/cônjuges a bordo.
- `Parch`: Número de pais/filhos a bordo.
- `Ticket`: Número do bilhete.
- `Fare`: Tarifa paga pelo passageiro.
- `Cabin`: Número da cabine.
- `Embarked`: Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton).

## Pré-processamento e Engenharia de Features
As seguintes etapas foram realizadas para preparar os dados para modelagem:

1.  **Tradução de Colunas:** Colunas foram traduzidas para o português para facilitar a análise e o entendimento.
2.  **Tratamento de Valores Ausentes:**
    *   `Idade`: Preenchida com a média.
    *   `Tarifa`: Preenchida com a média (apenas no conjunto de teste, onde um valor estava ausente).
    *   `Embarked`: Preenchida com o modo (valor mais frequente), embora não tenha sido explicitamente mostrado no notebook, o one-hot encoding tratou os nulos restantes.
3.  **Engenharia de Features:**
    *   `Cabine_known`: Nova feature binária (`0` ou `1`) indicando se a informação da cabine estava presente. A coluna `Cabine` original foi então descartada.
    *   `Tamanho_Familia`: Criada pela soma de `Irmaos_Conjuge`, `Pais_Filhos` e `1` (para incluir o próprio passageiro). As colunas originais `Irmaos_Conjuge` e `Pais_Filhos` foram descartadas.
4.  **Codificação de Variáveis Categóricas:**
    *   `Sexo`: Label Encoding (`male`: 0, `female`: 1).
    *   `Embarque`: One-Hot Encoding (criando `Embarque_C`, `Embarque_Q`, `Embarque_S`). Posteriormente, `Embarque_Q` foi descartada devido à sua correlação insignificante com a variável alvo.
5.  **Remoção de Colunas:**
    *   `Nome` e `Bilhete`: Descartadas por não serem diretamente preditivas.
    *   `ID_Passageiro`: Descartada do conjunto de treino (mantida separadamente para submissão no Kaggle).
    *   `Embarque_Q`: Descartada devido à baixa correlação.
6.  **Balanceamento de Classes:** O método SMOTE (Synthetic Minority Over-sampling Technique) foi aplicado ao conjunto de treino para lidar com o desbalanceamento entre as classes `Sobreviveu` (1) e `Não Sobreviveu` (0).
7.  **Padronização de Dados:** `StandardScaler` foi utilizado para padronizar as features numéricas, garantindo que todas as variáveis contribuam igualmente para o modelo.

## Análise de Correlação
Uma matriz de correlação foi gerada para entender as relações entre as variáveis. Pontos chave:
- `Sexo` (0.54) e `Classe_Passageiro` (-0.34) foram os preditores mais fortes de `Sobreviveu`.
- `Cabine_known` (0.32) e `Tarifa` (0.26) também mostraram correlações moderadas.
- Variáveis como `Idade`, `Irmaos_Conjuge` e `Pais_Filhos` mostraram correlações mais fracas individualmente, mas `Tamanho_Familia` foi criada para capturar o efeito combinado.
- Foi observada alta multicolinearidade entre `Classe_Passageiro` e `Cabine_known` (-0.82), o que foi considerado na seleção de features.

## Modelagem
Foram utilizados dois modelos de Machine Learning, XGBoost e LightGBM, ambos conhecidos por seu alto desempenho em problemas de classificação.

### Otimização de Hiperparâmetros (GridSearchCV)
Para cada modelo, `GridSearchCV` foi empregado para encontrar a melhor combinação de hiperparâmetros, utilizando a acurácia como métrica de avaliação e validação cruzada com 5 folds.

#### 1. XGBoost (XGBClassifier)
**Melhores Parâmetros Encontrados:** `{'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 1.0}`
**Melhor Acurácia de Validação Cruzada:** `0.8315`

#### 2. LightGBM (LGBMClassifier)
**Melhores Parâmetros Encontrados:** `{'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 100, 'num_leaves': 20, 'subsample': 0.7}`
**Melhor Acurácia de Validação Cruzada:** `0.8124`

## Avaliação e Resultados

### XGBoost
**Relatório de Classificação (no conjunto de treino balanceado):**
```
              precision    recall  f1-score   support

           0       0.87      0.94      0.90       549
           1       0.93      0.86      0.89       549

    accuracy                           0.90      1098
   macro avg       0.90      0.90      0.90      1098
weighted avg       0.90      0.90      0.90      1098
```
**Acurácia na Submissão do Kaggle:** `0.73684`

### LightGBM
**Relatório de Classificação (no conjunto de treino balanceado):**
```
              precision    recall  f1-score   support

           0       0.85      0.82      0.84       549
           1       0.83      0.86      0.84       549

    accuracy                           0.84      1098
   macro avg       0.84      0.84      0.84      1098
weighted avg       0.84      0.84      0.84      1098
```
**Acurácia na Submissão do Kaggle:** `0.71531`

## Comparação e Conclusão

Ambos os modelos foram treinados e avaliados, e os resultados indicam o seguinte:

| Métrica                      | XGBoost | LightGBM |
| :--------------------------- | :------ | :------- |
| Acurácia CV (GridSearchCV)   | 0.8315  | 0.8124   |
| Acurácia Kaggle              | 0.73684 | 0.71531  |

Com base tanto na acurácia de validação cruzada quanto no desempenho na submissão pública do Kaggle, o **XGBoost demonstrou um desempenho superior** ao LightGBM neste desafio. O XGBoost obteve uma acurácia aproximadamente 2.15 pontos percentuais maior na plataforma Kaggle, consolidando-o como o modelo mais eficaz para este problema.

## Submissões Geradas
- `xgboost_submission.csv`
- `lgbm_submission.csv`

## Resultado da Submissão no Kaggle

<img width="1475" height="204" alt="Captura de tela 2026-04-09 164001" src="https://github.com/user-attachments/assets/95e476e2-09ae-4cd4-bc8c-5f33bf37a5d9" />


## Próximos Passos (Potenciais Melhorias)
- **Análise de Erros:** Investigar os falsos positivos e falsos negativos para identificar padrões e refinar o modelo.
- **Engenharia de Features Avançada:** Explorar a criação de mais features a partir das existentes (e.g., títulos dos nomes, grupos de família, etc.).
- **Modelos de Ensemble:** Combinar as previsões de múltiplos modelos (stacking, blending) para buscar um desempenho ainda melhor.
- **Otimização de Hiperparâmetros Mais Abrangente:** Usar `RandomizedSearchCV` ou otimização bayesiana para explorar um espaço maior de hiperparâmetros de forma mais eficiente.
- **Outros Modelos:** Testar outros algoritmos de classificação como Random Forest, SVC com diferentes kernels ou redes neurais simples.
