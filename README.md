# Project machine-learning - Workflows-pipeline

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/R-3.6.0-red.svg)](https://www.r-project.org/)
[![](https://img.shields.io/badge/ggplot2-white.svg)](https://ggplot2.tidyverse.org/)
[![](https://img.shields.io/badge/dplyr-blue.svg)](https://dplyr.tidyverse.org/)
[![](https://img.shields.io/badge/readr-green.svg)](https://readr.tidyverse.org/)
[![](https://img.shields.io/badge/ggvis-black.svg)](https://ggvis.tidyverse.org/)
[![](https://img.shields.io/badge/Shiny-red.svg)](https://shiny.tidyverse.org/)
[![](https://img.shields.io/badge/plotly-green.svg)](https://plotly.com/)
[![](https://img.shields.io/badge/XGBoost-red.svg)](https://xgboost.readthedocs.io/en/stable/#)
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Caret-orange.svg)](https://caret.tidyverse.org/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Scikit_Learn-green.svg)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)
[![](https://img.shields.io/badge/PowerBI-red.svg)](https://powerbi.microsoft.com/pt-br/)

              
![Logo](https://as1.ftcdn.net/v2/jpg/02/13/59/36/1000_F_213593664_Q7E5xrx8cKXmalLOfRFe9CY2jHvNp4f5.jpg)


# Descrição do projeto
Projeto criação de modelos de pipeline para modelo de machine learning.

- Projeto Workflows - M.l
- Neste projeto fiz um modelo de classificação emails são real e spam base de dados tem total de 5172 linhas 3002 e colunas. No dataset existe uma coluna chamado de Prediction, essa coluna a ser classificado ela está como email para spam 0 para verdadeiro real 1. Objetivo e classificar email que são spam e real. Nesse projeto está tudo documentado os passos que fiz nesse projeto.

O que pipeline: Pipeline otimização para o modelo Machine learning para não ter overfitting no modelo.

# Stack utilizada
Programação Python

Machine learning: Scikit-learn

Leitura CSV: Pandas

Análise de dados: Seaborn, Matplotlib

## Variáveis de Ambiente

Para rodar esse projeto, você vai precisar adicionar as seguintes variáveis de ambiente no seu .env

Instalando a virtualenv

`pip install virtualenv`

Nova virtualenv

`virtualenv nome_virtualenv`

Ativando a virtualenv

`source nome_virtualenv/bin/activate` (Linux ou macOS)

`nome_virtualenv/Scripts/Activate` (Windows)

Retorno da env

`projeto_py source venv/bin/activate` 

Desativando a virtualenv

`(venv) deactivate` 

Instalando pacotes

`(venv) projeto_py pip install flask`

Instalando as bibliotecas

`pip freeze`

## Instalação 

Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```

Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```

## Demo modelo M.L

```bash
  # Carregando as bibliotecas 
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  # Carregando o dataset
  data = pd.read_csv("data.csv")
  
  # Visualizando os 5 primeiros itens
  data.head()

  # visualizando linhas e colunas com shape
  data.shape

  # Informações das variaveis
  data.info()

  # Treino e teste da base de dados para x e y
  x = df_train.iloc[:, 0: 10]
  y = df_train.iloc[:, 10]

  # Visualizando o shape da variavel x
  x.shape

  # Visualizando o shape da variavel y
  y.shape

  # Treinando modelo de machine learning
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

  # Visualizando linhas e colunas do dado de treino x_train
  x_train.shape

  # Visualizando linhas e colunas do dado de treino y_train
  y_train.shape

  # Modelo pipeline
  from sklearn.feature_selection import VarianceThreshold
  from sklearn.pipeline import Pipeline

  data_pipeline = Pipeline([
  ("scaler", StandardScaler()), # Scaler : Para pré-processamento de dados, ou seja, transforme os dados em média zero e variância de unidade usando o StandardScaler ().
  ("selector", VarianceThreshold()), # Seletor de recurso : Use VarianceThreshold () para descartar recursos cuja variação seja menor que um determinado limite definido.
  ("classifier", KNeighborsClassifier()) # Classificador : KNeighborsClassifier (), que implementa o classificador de k-vizinho mais próximo e seleciona a classe dos k pontos      principais, que estão mais próximos do exemplo de teste.
  ])
  data_pipeline_fit = data_pipeline.fit(x_train, y_train)
  data_pipeline_score = data_pipeline.score(x_train, y_train)

  print('Treinamento base treino - Pipeline: ' + str(data_pipeline.score(x_train,y_train)))
  print('Treinamento base teste - Pipeline: ' + str(data_pipeline.score(x_test,y_test)))

  # Previsão do pipeline do modelo
  data_pipeline_pred_1 = data_pipeline.predict(x_test)
  data_pipeline_pred_1

  # Accuracy do pipeline
  from sklearn.metrics import accuracy_score
  accuracy_pipeline_1 = accuracy_score(y_test, data_pipeline_pred_1)
  print("Accuracy - pipeline: %.2f" % (accuracy_pipeline_1 * 100))

  # Confusion matrix do modelo
  from sklearn.metrics import confusion_matrix
  matrix_1 = confusion_matrix(y_test, data_pipeline_pred_1)
  plot_confusion_matrix(matrix_1, show_normed=True, colorbar=False, class_names=['SPAM', 'NAO-SPAM'])

  # Curva ROC do modelo
  from sklearn.metrics import roc_curve, roc_auc_score
  roc = data_pipeline.predict_proba(x_test)[:,1]
  tfp, tvp, limite = roc_curve(y_test, roc)
  print('roc_auc', roc_auc_score(y_test, roc))

  plt.subplots(1, figsize=(5,5))
  plt.title('Curva ROC')
  plt.plot(tfp,tvp)
  plt.xlabel('Especifidade')
  plt.ylabel('Sensibilidade')
  plt.plot([0, 1], ls="--", c = 'red')
  plt.plot([0, 0], [1, 0], ls="--", c = 'green'), plt.plot([1, 1], ls="--", c = 'green')
  plt.show()

  # Classification report do modelo
  from sklearn.metrics import classification_report
  classification = classification_report(y_test, data_pipeline_pred_1)
  print("Modelo - Pipeline 1")
  print()
  print(classification)

  # Métricas do modelo 
  from sklearn.metrics import precision_score
  from sklearn.metrics import recall_score
  from sklearn.metrics import recall_score
  from sklearn.metrics import f1_score

  precision = precision_score(y_test, data_pipeline_pred_1)
  Recall = recall_score(y_test, data_pipeline_pred_1)
  Accuracy = accuracy_score(y_test, data_pipeline_pred_1)
  F1_Score = f1_score(y_test, data_pipeline_pred_1)

  precisao = pd.DataFrame({
    
    "Metricas" : ["precision",
                 "Recall", 
                  "Accuracy", 
                  "F1_Score"],
    
    "Resultado": [precision,
                Recall, 
                Accuracy, 
                F1_Score]})

```
# Melhorias
Que melhorias você fez no seu código?

Ex: refatorações, melhorias de performance, acessibilidade, etc

#Suporte
Para suporte, mande um email para rafaelhenriquegallo@gmail.com

## Documentação

[Documentação](https://link-da-documentação)

## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Relacionados

Segue alguns projetos relacionados

[Awesome README](https://github.com/matiassingers/awesome-readme)
