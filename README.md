# Contexto

Projeto baseado em um trecho do Episódio 4 da 4ª temporada do seriado Sillicon Valley. 

[Link para o vídeo.](https://youtu.be/vIci3C4JkL0?t=52)

# Entendendo um pouco sobre redes convolucionais

## Origem
A primeira aplicação com sucesso de uma CNN foi desenvolvida por Yann LeCun em 1998, com sete camadas entre convoluções e fully connected. Desde então as CNNs ficaram cada vez mais profundas e complexas, como AlexNet em 2012, que, apesar de ter apenas oito camadas (cinco convoluções e três fully connected), apresenta sessenta milhões de parâmetros, e a GoogleNet com vinte e duas camadas e quatro milhões de parâmetros.

## Entradas
Quando falamos em reconhecimento/classificação de imagens, as entradas são usualmente matrizes tridimensionais com altura e largura (de acordo com as dimensões da imagem) e profundidade, determinada pela quantidade de canais de cores. Em geral as imagens utilizam três canais, RGB, com os valores de cada pixel.

![image](https://user-images.githubusercontent.com/87540453/179066421-b9f5447c-0791-4a4d-bcde-6d1143a2d500.png)

## Convoluções
As convoluções funcionam como filtros que enxergam pequenos quadrados e vão “escorregando” por toda a imagem captando os traços mais marcantes. Explicando melhor, com uma imagem 32x32x3 e um filtro que cobre uma área de 5x5 da imagem com movimento de 2 saltos (chamado de stride), o filtro passará pela imagem inteira, por cada um dos canais, formando no final um feature map ou activation map de 28x28x1.

![image](https://user-images.githubusercontent.com/87540453/179066585-6442f353-cd89-479b-98cd-0507661f1c21.png)

A profundidade da saída de uma convolução é igual a quantidade de filtros aplicados. Quanto mais profundas são as camadas das convoluções, mais detalhados são os traços identificados com o activation map.

O filtro, que também é conhecido por kernel, é formado por pesos inicializados aleatoriamente, atualizando-os a cada nova entrada durante o processo de backpropagation. A pequena região da entrada onde o filtro é aplicado é chamada de receptive field.

Exemplo:

![image](https://user-images.githubusercontent.com/87540453/179066908-fa4c7ce1-65ff-40ee-81d6-e038d9f8f752.png)

## Função de ativação
As funções de ativação servem para trazer a não-linearidades ao sistema, para que a rede consiga aprender qualquer tipo de funcionalidade. Há muitas funções, como sigmoid, tanh e softmax, mas a mais indicada para redes convolucionais é a Relu por ser mais eficiente computacionalmente sem grandes diferenças de acurácia quando comparada a outras funções. Essa função zera todos os valores negativos da saída da camada anterior.

## Pooling
Uma camada de pooling serve para simplificar a informação da camada anterior. Assim como na convolução, é escolhida uma unidade de área, por exemplo 2x2, para transitar por toda a saída da camada anterior. A unidade é responsável por resumir a informação daquela área em um único valor. Se a saída da camada anterior for 24x24, a saída do pooling será 12x12. Além disso, é preciso escolher como será feita a sumarização. O método mais utilizado é o maxpooling, no qual apenas o maior número da unidade é passado para a saída. Essa sumarização de dados serve para diminuir a quantidade de pesos a serem aprendidos e também para evitar overfitting.

![image](https://user-images.githubusercontent.com/87540453/179067537-60eb0bf0-20b3-4fae-b218-14d62f0b555b.png)

## Fully connected
Ao final da rede é colocada uma camada Fully connected, onde sua entrada é a saída da camada anterior e sua saída são N neurônios, com N sendo a quantidade de classes do seu modelo para finalizar a classificação.

Texto retirado do [Medium](https://medium.com/infosimples/understanding-convnets-cnn-712f2afe4dd3).

# Usando CNN

## Exemplos de imagens usadas no treino e no teste

![graph](https://user-images.githubusercontent.com/87540453/179227964-96cfa2aa-0a47-4a2b-b2fa-426b9fa79e93.png)

