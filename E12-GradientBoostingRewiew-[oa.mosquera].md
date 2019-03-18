Gradient Boosting vs XGBoots

El Gradient Boosting es también un algoritmo de “impulso”, por lo que también intenta crear un “aprendiz” fuerte a partir de un conjunto de “aprendices” débiles. Este algoritmo es similar al Adaptive Boosting (AdaBoost) pero difiere de él en ciertos aspectos. En este método, se intenta visualizar el problema de impulso como un problema de optimización, es decir, se asume una función de pérdida y se busca optimizarlo.

El XGBoost (aumento de gradiente extremo), es similar al algoritmo de Gradient Boosting, pero tiene algunas
características que lo diferencian, como lo son la penalización inteligente de los árboles, la reducción proporcional de los nodos de las hojas y el uso de un parámetro de aleatorización adicional (Newton Boosting).

En XGBoost, los árboles pueden tener un número variable de nodos terminales y el peso izquierdo de los árboles que se calculan con menos evidencia se reduce más. Newton Boosting utiliza el método de aproximaciones de Newton-Raphson que proporciona una ruta directa a los mínimos que el descenso de gradiente. El parámetro de asignación aleatoria adicional se puede usar para reducir la correlación entre los árboles, cuanto menor sea la correlación entre los clasificadores, mejor resulta el conjunto de clasificadores.

Los dos modelos siguen el principio de aumento de gradiente, sin embargo, la principal diferencia entre ellos, radica en los detalles de modelado, el XGBoots utiliza un modelo de formalización más regular para controlar el ajuste excesivo (over-fitting).

En este sentido se puede asegurar que el XGBoost, es un algoritmo de aumento de gradiente regularizado, que tiene una gran rapidez de ejecución debido a que, si bien no es posible paralelizar el “ensemble” en sí mismo, porque cada árbol depende del anterior, puede paralelizar la construcción de varios nodos dentro de cada profundidad de cada árbol.

Adicionalmente el XGBoost realiza uso de matrices dispersas con algoritmos conscientes de dispersión, estructuras de datos mejoradas para una mejor utilización del caché del procesador, y tiene un mejor soporte para el procesamiento multinúcleo, lo que reduce en sí, el tiempo total de entrenamiento. Estas características suponen una reducción significativa en la utilización de la memoria, al entrenar grandes conjuntos de datos con Gradient Boosting, es más fácil usar múltiples núcleos para reducir dicho tiempo.

Otra diferencia importante se centra en que XGBoots ha realizado la implementación de DART (Regularización de abandono para los árboles de regresión).

Se puede concluir entonces, que la diferencia significativa entre el Gradient Boosting vs XGBoots, es que en XGBoots el algoritmo se centra en la potencia de cálculo, al paralelizar la formación de los árboles, y el Gradient Boosting solo se centra en la varianza, pero no en el intercambio entre sesgos, mientras que el aumento de “Extreme Gradient” también puede centrarse en el factor de regularización.
