# Proyecto 1: &nbsp; `Regresión Lineal

## Información

El objetivo de este proyecto es predecir el área quemada por los incendios forestales en la región noreste de Portugal.
Para lograrlo, se le proporcionará un conjunto de datos que contiene 517 instancias y 12 features. La feature 13 es el área que necesita predecir.

1. &nbsp; _**X**_ - coordenada x espacial del parque de Montesinho: &nbsp; `1` a `9`

2. &nbsp; _**Y**_ - coordenada y espacial del parque de Montesinho: &nbsp; `2` a `9`

3. &nbsp; _**month**_ - mes del año: &nbsp; `"jan"` a `"dec"`

4. &nbsp; _**day**_ - día de semana: &nbsp; `"mon"` a `"sun"`

5. &nbsp; _**FFMC**_ - índice FFMC del sistema FWI: &nbsp; `18.7` a `96.20`

6. &nbsp; _**DMC**_ - índice DMC del sistema FWI: &nbsp; `1.1` a `291.3`

7. &nbsp; _**DC**_ - índice DC del sistema FWI: &nbsp; `7.9` a `860.6`

8. &nbsp; _**ISI**_ - índice ISI del sistema FWI: &nbsp; `0.0` a `56.10`

9. &nbsp; _**temp**_ - temperatura (°C): &nbsp; `2.2` a `33.30`

10. &nbsp; _**RH**_ - Humead relativa (%): &nbsp; `15.0` a `100`

11. &nbsp; _**wind**_ - velocidad de viento (km/h): &nbsp; `0.40` a `9.40`

12.  &nbsp; _**rain**_ - lluvia exterior (mm/m$^2$) : &nbsp; `0.0` a `6.4`

13.  &nbsp; _**area**_ - área de bosque quemado (ha o 10$^4$ m$^2$): &nbsp; `0.00` a `1090.84` (mas sesgada a `0.0`)

> [!NOTE]
> - No tiene valores faltantes
> - Varios de los atributos pueden estar correlacionados, por lo que tiene sentido aplicar algún tipo de selección de features.