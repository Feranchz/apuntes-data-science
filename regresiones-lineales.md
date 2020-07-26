# Hablemos de Regresiones Lineales

Created: Jul 25, 2020 6:27 PM
Reviewed: No

Antes de comenzar, te explico de qué va esto: Esto es una pequeña guía que toma un poco de todo el conocimiento que se puede encontrar por Internet (mención especial al profesor Adrian Catalán de Platzi y a DotCSV), además de libros y busca resumirlo con mucho cuidado a la par que explicamos la razón de cada línea de código y porqué esta ahí.

Dicho eso, antes de empezar, deberías tener una noción mínima de Python, entender como funciona la libreria PyTorch y también saber un poco de matemáticas.

Puedes revisar todo el código con más detenimiento en Kaggle, en el siguiente link: [https://www.kaggle.com/fernetico/regression-simple-example](https://www.kaggle.com/fernetico/regression-simple-example)

Entonces, si estas preparado, vamos a comenzar importando todo lo que necesitaremos.

```python
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import ast
import json

import matplotlib.pyplot as plt
```

# Un momento... ¿Qué es una regresión lineal?

En la vida podemos observar un montón de fenómenos que implican variables dependientes e independientes.

Por ejemplo, imagina que decidiste comprar un nuevo carro, así que vas a tu concesionario más cercano y empiezas a ver las diferentes marcas y modelos de carro. Si ves muchos carros, empezarás a notar que el precio de cada uno de ellos depende de varios factores. En este caso, el precio es tu variable dependiente. Es fácil darse cuenta que el precio sube o baja dependiendo del año del vehículo, la cantidad de kilómetros recorridos para carros usados, el número de puertas, incluso la marca del vehículo hace variar el precio de un carro con características similares a uno de otra marca, todo esto son nuestras variables dependientes

Pero para este ejemplo no hablemos de carro, hablemos de videojuegos, especificamente hablemos de League of Legends.

League of Legends es un juego muy complejo, pero aquí no queremos entrar en esos detalles, sino que nos quedaremos con datos muy básicos para prácticar cómo hacer una regresión lineal. Con eso en mente vamos a aclarar lo siguiente, en este juego se enfrentan dos equipos, esos equipos reciben oro por diferentes acciones, una de ellas es matar a un personaje del equipo rival. Así que hoy lo que intentaremos explicar es la cantidad de muertes según el oro total del equipo. Vamos a intentar predecir cuantos asesinatos logró un equipo si obtuvo cierta cantidad de oro en la partida.

Para ello vamos a usar un Dataset que reune datos de partidas competitivas de League of Legends en Kaggle, para no complicarnos demasiado, vamos a limitarnos a las partidas que ocurren en la liga norteamericana.

Así que lo primero que hacemos es recuperar nuestros datos.

```python
league_data = pd.read_csv("../input/leagueoflegends/LeagueofLegends.csv")

league_data = league_data[['goldblue', 'bKills', 'goldred', 'rKills', 'League']]
league_data = league_data[league_data['League'] == 'NALCS']

kills = [] ## Variable dependiente
gold = [] ## Variable independiente

for index, row in league_data.iterrows():
    kills.append(len(ast.literal_eval(row['bKills'])))
    gold.append(json.loads(row['goldblue'])[-1])
    kills.append(len(ast.literal_eval(row['rKills'])))
    gold.append(json.loads(row['goldred'])[-1])
```

Lo que esta ocurriendo acá es muy simple, filtramos de todo nuestro dataset solamente los datos que queremos y los ordenamos todos en dos listas de Python y procedemos a graficarlas para ver qué esta sucediendo.

```python
plt.ylabel('Kills')
plt.xlabel('Gold')
plt.scatter(gold, kills, s=10)
```

![Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled.png](Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled.png)

Tenemos unos datos muy variados, y es que la cantidad de oro que reúne un equipo depende de muchos factores, pero si podemos notar que al menos en los primeros elementos obtenemos más oro mientras más asesinatos logramos.

## ¿Y de qué me sirve esto?

A simple vista esto puede parecer algo demasiado obvio y no nos dice mucho, pero acá queremos predecir comportamientos (que no necesariamente se cumpliran), por ejemplo ¿Si un equipo terminó una partida con 150.000 de oro, cuantos asesinatos podría tener?

Hagamos un ejercicio a ojo e intentemos trazar una línea encima de nuestro plano que tenga el mismo comportamiento que nuestros puntos:

![Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%201.png](Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%201.png)

Esta línea que acabamos de trazar, es un modelo.

## ¿Y qué es un modelo?

Bueno, ahora tenemos una pregunta que puede parecer complicada, pero no lo es. Vamos a definir al modelo como una ecuación que nos permite explicar algo que ocurre en la realidad usando a nuestros queridos números.

![Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%202.png](Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%202.png)

## ¿Entonces la línea que dibujamos es una ecuación?

¡Si!, recordemos que esta línea que acabamos de dibujar la podemos escribir también como una ecuación y tendría la siguiente forma:

$y = b + mx$

En esta ecuación $x$ y $y$ son variables, pero $b$ y $m$ son escalares, es decir, son un número fijo. Por una parte $b$ es el punto exacto en el que la recta corta al eje y, y $m$ es la pendiente de la recta, un número que nos indica cuanto se mueve hacia arriba o hacia abajo nuestra recta.

Con esta ecuación, podemos escribir cualquier recta que exista en el mundo infinito de las rectas.

Ahora, en nuestra ecuación, $y$ es la cantidad de asesinatos, es nuestra variable dependiente, y $x$ es la cantidad de oro con el que finalizo la partida un equipo, es nuestra variable independiente. Quizás estas pensando que esto no tiene mucho sentido, el oro debería depender de la cantidad de asesinatos, entre otros factores. Pero este cambio es simplemente para hacer el ejercicio de verlo desde otro ángulo.

Si sabemos que tanto $m$ como $b$ son valores que ya tenemos, entonces con simplemente darle un valor a $x$ podríamos tener un valor para $y$, y de esta forma tendríamos una predicción de cuantos asesinatos hizo un equipo según la cantidad de oro que reunió. Esto es lo que queremos predecir/adivinar.

## ¡Pues vamos a hacerlo!

Calma, esta línea que acabamos de dibujar encima de nuestro plano a puro ojo no es la línea correcta. Nisiquiera sabemos en qué punto corta al eje y, y mucho menos la pendiente de la recta. 

Entonces, tenemos un plano con muchos puntos, y necesitamos una recta que pase por ese plano, pero si vemos detenidamente nuestros puntos notamos que no es posible dibujar una recta que pase exactamente por encima de cada punto. Algunos puntos estan más arriba, otros más abajos. ¿Es hora de caer en el pánico?

No, veamos el siguiente plano con muchos menos puntos para entender mejor lo que sucede:

![Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%203.png](Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%203.png)

En este caso, hemos dibujado una recta (roja) mucho menos precisa que la anterior, y tenemos líneas verdes que van desde cada uno de los puntos a la recta, moviendose solo verticalmente. Estas líneas verdes tienen una distancia.

Lo que necesitamos, es una recta tal que estas distancias verdes de los puntos a la recta, sea la menor posible, esa será la regresión lineal. 

## Que comience la función.

Podríamos dibujar rectas y rectas cada vez más cercanas a nuestros puntos, y calcular las distancias hasta encontrar una que sea lo suficientemente buena para nosotros, pero esto sería agotador. Además, queremos crear algo automático que sea capaz de hacerse para cualquiera de los datos que recibamos, entonces, es hora de codear.

Primeramente, y sin detenernos mucho a explicarlo, vamos a transformar nuestros datos para poder tenerlos de la manera que queremos en PyTorch.

```python
kills_array = np.array(kills).reshape(-1, 1)
gold_array = np.array(gold).reshape(-1, 1)
kills_tensor = torch.from_numpy(kills_array).float().requires_grad_(True)
gold_tensor = torch.from_numpy(gold_array).float()
```

Quizás estas notando algo raro, ¿Por qué nuestro tensor de kills llama a esa funcion "requires_grad_" al final?

Esta herramienta nos permite calcular y almacenar los gradientes de esta función, ¿Pero qué es un gradiente? El gradiente podemos verlo simplemente como una guía que nos va a indicar hacia qué camino debemos ir. 

¿Quieres entender mejor qué es un gradiente y cómo lo usamos? [https://www.youtube.com/watch?v=A6FiCDoz8_4](https://www.youtube.com/watch?v=A6FiCDoz8_4)

## ¡Comencemos a programar nuestra regresión!

Ya llegaste hasta aquí, y eso tiene su recompensa, el código para hacer una regresión lineal es bastante sencillo pero es importante entender todo lo que hay detrás de él porque es la base para cosas mucho más avanzadas.

Comencemos declarando nuestro modelo lineal, este método recibira dos parametros que indicaran la dimensión de nuestra entrada y la dimensión de salida respectivamente.

```python
model = nn.Linear(1, 1)
```

Este método de torch nos crea un modelo de la siguiente forma:

$Y = xA^T + b$

¿Notan algo familiar? Si, es la misma forma que tenía nuestra ecuación anterior de una recta, pero ahora tenemos matrices. Pero no te asustes, con esto solamente estamos escribiendo todas nuestras posibles rectas según los puntos que tenemos. En este caso $A$ representa un vector con todas nuestras variables independientes (el oro reunido).

Ya tenemos la forma de nuestro modelo, pero tenemos que ajustarlo, porque ahora mismo no representa el comportamiento de los puntos que tenemos. Para ello haremos lo siguiente:

```python
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
```

¿Recuerdas que antes hablabamos de que teniamos que encontrar la distancia entre nuestra recta y nuestro punto, para así poder buscar la menor posible?

Pues para exactamente eso servira nuestra variable loss_function, que ahora es una función que debe recibir dos parametros y estos son nuestro punto $(x, y)$ original pero también el punto exacto si nos movemos verticalmente de la recta que estamos intentando dibujar. Así podremos saber qué tan lejos esta nuestra recta.

También declaramos un optimizer. Aquí es donde ocurre la magia, esto es lo que se va a encargar de ir acomodando nuestro modelo poco a poco al que más se ajuste. Existen muchos algoritmos de optimización, pero en este caso usamos el SGD (Stochastic gradient descent)

## Descenso del gradiente.

Antes de hablar del SGD, debemos entender cómo funciona el método del descenso del gradiente. 

¿Recuerdas que anteriormente calculamos unos gradientes? No vamos a complicarnos con derivadas, por ahora solo debemos saber que este gradiente lo que hace es indicarnos hacia dónde debemos ir. Nosotros estamos buscando tener el mínimo error posible, por lo tanto estamos buscando los puntos más bajos de la función, así que siempre intentaremos bajar.

El lr que le pasamos a nuestro optimizer, representa cuanto queremos movernos cada vez que decidamos bajar en la dirección que nuestro gradiente nos esta indicando. En este ejercicio podemos poner un lr a simple vista, pero su definición suele ser muy importante según el problema que tenemos. 

Imagina por un momento que nuestro lr es muy alto, y cada vez que nos movemos damos pasos muy grandes, puede suceder que el paso que demos sea tan grande que perdamos el punto al que queremos acercanos. Por otra parte, si damos pasos muy pequeños, podríamos estar toda la vida caminando antes de llegar al punto que queremos, y en la computación, el rendimiento siempre es importante.

Ahora que tenemos una idea de lo que hace el descenso del gradiente, entendamos que SGD es una variación de este algoritmo que le agrega un elemento estocastico. Es decir, cada vez que vamos a dar un paso, agregamos un pequeño elemento aleatorio que evitara que nos estanquemos en algún punto.

## ¿Será que al fin podemos hacer la regresión lineal?

Más despacio. Ya tenemos lo que necesitamos, pero antes vamos a declarar dos cosas, declaramos un arreglo para guardar nuestro error en cada iteración, así podremos ver como vamos avanzando y además vamos a decidir cuantas iteraciones hacer. Aunque también podrías simplemente ejecutar el algoritmo hasta que el error sea tan pequeño que no te importe.

```python
losses = []
iterations = 2000
```

¡Y finalmente, vamos a ejecutar nuestro algoritmo!

```python
for i in range(iterations):
	pred = model(kills_tensor)
	loss = loss_function(pred, gold_tensor)
	losses.append(loss.data)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
```

Lo primero que hacemos es realizar una predicción inicial usando nuestro modelo y las x que tenemos, de aquí obtendremos un vector que tendrá nuestras predicciones de asesinatos por equipo.

Con este vector, vamos a usar nuestra loss_function que declaramos anteriormente y comparamos los asesinatos que predijimos contra los asesinatos reales que conocemos, esto nos dará el error, este error lo guardamos en losses para posteriormente verlo en una gráfica.

Más adelante es cuando comienza nuestra predicción, lo primero que hacemos es limpiar los gradientes, PyTorch acumula los gradientes cada vez que da un paso para aproximarse a nuestra recta, es por ello que lo primero que hacemos es limpiarlos.

Luego, ejecutamos al fin nuestro backward, en esta función, haciendo uso de derivadas vamos a obtener los gradientes que nos indican hacia dónde debemos movernos.

Finalmente, nuestro optimizer da el paso con la dirección que obtuvo de antes, y al hacer esto actualiza nuestro modelo con los nuevos datos.

Conforme se ejecuta cada iteración, el modelo se adapta más y más a la recta que buscabamos, y podemos usar la función para predecir cualquier valor. RECUERDA: esto es solo una predicción. 

Vamos a usar los errores que guardamos anteriormente para ver qué esta sucediendo.

```python
plt.plot(range(iterations), losses)
```

![Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%204.png](Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%204.png)

Aca notamos el comportamiento del algoritmo de gradientes descendentes, empezamos con un error grande y conforme avanzamos este error cada vez se disminuye hasta ser practicamente nulo.

## Finalmente... nuestra regresión.

```python
plt.plot(model(kills_tensor).tolist(), kills, 'r')
plt.scatter(gold, kills, s=10)
plt.show()
```

![Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%205.png](Hablemos%20de%20Regresiones%20Lineales%201d6aa36e148246f0a97f9551c188a700/Untitled%205.png)

Y así, finalmente, hemos obtenido la recta que estabamos buscando. A simple vista podemos notar que no es 100% estricta, pero nos servira para hallar predicciones, o lo que es más importante, le servira a nuestros futuros algoritmos para que ellos mismos puedan conseguirlas.

## The end.