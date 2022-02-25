# Projeto 1

O projeto 1 foi feito utilizando as bibliotecas do sklearn.

## Como rodar

Os gráficos são disponibilizados na pasta output

Para instalar as dependencias:

`pip install -r requirements.txt`

Para rodar:

`python3 main.py`

Foi feita também uma solução utilizando jupiter notebook para melhorar a visualização/correção e ṕde ser acessado por
esse link:
https://github.com/PedroAugustoRamalhoDuarte/fsi/blob/main/projeto1/main.ipynb

## Explicação dos passos

1 - Para o passo um foi utilizado o método do panda describe e foram plotados dos gráficos, um para médias das variáveis
e um para o desvio padrão

2,3,4 - Para esses passos foram utilizados o mesmo trecho de código para plotagem da informação, alterando somente o
modelo empregado, para isso foi utilizado o modelo kfold para fazer a validação cruzada e coletar os dados de matriz de
confusão, roc, roc auc, as melhores features. Nesse passo são plotados os gráficos de matriz de confusão e curva roc e
auc para cada um dos modelos

5 - No passo 5 foi escolhido o melhor modelo e o melhor fold com base no roc auc, em geral o melhor modelo é o do
RandomForest com a raiz das variáveis, porém em algumas execuções é possível perceber que o melhor modelo pode ser o
Random forest com 9 variáveis e as vezes o melhor fold não pertence ao melhor modelo na média. Nesse passo é exibido no
terminal o melhor modelo na média o melhor fold entre todos e é plotado um gráfico com as variáveis mais importantes,
que sempre são a age e tobacco
