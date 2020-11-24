# Research Journal

## 16/10/2020
- I included a binary variable to guarantee that the battery bidirectional inverter is charging XOR discharging.
- I checked the CREG price algorithm and found a discrepancy. The formula asks for a power curve, but currently it is presented as a percentage. I fixed this by using the maximum load power value as 100% of the typical load curve. This chenges the price of the energy price computation. I still have to figure out how to correctly compute this.

## 23/10/2020
I changed the code structure by preparing functions that solve the optimization problem and present the results in a figure. This will allow us to solve the problem for multiple scenarios and show a comparison of the results.
TODO:
- [ ] Check if solutions where the battery charge and discharge power are equal but nonzero are correct. Ask Adrian why two signals are used in this case.
- [ ] Ask Adrian why the constraint Pd \leq Pl is included.

## 28/05/2020
### Reunión con Adrián:
- [ ] Costo unitario: 550$/kW.  
- [ ] Comparar con el problema base con el costo unitario sin optimización.  
- [x] Quitar load shifting. El incentivo es el precio
- Venta de servicios adicionales: remuneración por subir o bajar la demanda.
- Control de frecuencia: 60Hz. AGC: plantas que ofrecen mover la generación hacia arriba o abajo para mantener la frecuencia alrededor de 60 Hz. Permitir al usuario final participar en éste mercado. Agregadores que reunan paquetes de usuarios finales para participar en el mercado.
  1. Alternativa 1: Explicar que los paquetes deben ser grandes para prestar el servicio. Nuestro modelo podría tomar la señal del agregador para participar
  2. Alternativa 2: Servicio auxiliar (no frecuencia). Variaciones para resolver problemas al operador de red (Codensa). Aliviar problemas de congestión, mejorar perfiles de distribución. Paquetes más pequeños.
- Keywords: Prosumer (consumidor que produce), frequency service.
- Papers de referencia:
  - P. Olivella-Rosell, E. Bullich-Massagué, M. Aragüés-Peñalba, A. Sumper, S.Ødegaard Ottesen, J.-A. Vidal-Clos, and R. Villafáfila-Robles, “Optimization problem for meeting distribution system operator requests in local flexibility markets with distributed energy resources,” Applied Energy, vol. 210, pp. 881 – 895, 2018. [Online].
  - Otros del mismo autor.

## 08/05/2020
I checked again in detail the file `PreciosHorarios.xlsx` provided by Adrián to understand the computations performed to obtain the energy prices. The basic idea is to compute the total price based on two components: market price and network usage. The variables can be defined as follows:
1. Load Power Forecast [kW]: $p_l(k)$
2. PV Power Forecast [kW]: $p_s(k)$
3. Net Power Flow [kW]: $p_n(k) = |p_l(k) - p_s(k)|$
4. Market Energy Price [$]: $c_m(k)$
5. Unit Energy Price [$]: $c_u$
6. Total Energy Price [$]: $c_t(k)$

The formula for obtaining the total price is:
$$c_t(k) = \frac{c_u}{2}\left[\left(\frac{c_m(k)}{\bar{c}_m}\right) + \left(\frac{p_n(k)}{\bar{p}_n}\right)\right]$$

After checking the results, it seems that the basic idea is to obtain a dynamic pricing scheme that in average yields the unit energy price.

TODO:
- [x] Compute the average market price for the current year.
- [x] Compute the average net power flow for the current year.
- [x] Compute the formula

## 28/02/2020
I implemented changes in the script `microgrid.py` for reading energy prices from a file with historic market prices obtained from XM. The records span the years 1995 to 2020.

## 27/02/2020
I have reviewed the code implemented in `microgrid2.py` for solving the optimization problem. I improved the presentation of the previous version of the script, and checked the results. So far it seems to be working OK.

I should now implement the approaches for determining the energy prices, obtaining this information from some dataset or computing it dynamically. I can start using historic market prices obtained from [XM](http://portalbissrs.xm.com.co/trpr/Paginas/Historicos/Historicos.aspx).

## 26/12/2020
There must be a causal relation between the rain precipitation levels and the energy price in the market, given that most of the energy generated in Colombia comes from hydroelectric plants. Maybe I could use a LSTM network for predicting the market price based on historic IDEAM measurements of weather variables. I should try to find out if there is data from locations with hydro power plants.
I checked the information available on [XM](http://www.xm.com). This company is in charge of the energy market exchange in Colombia. There is information available regarding market prices, grid operation measurements, etc.

## 12/02/2020
Meeting with Adrian
We discussed the different possibilities for determining the energy price in the context of the microgrid. He suggested three possibilities to the problem:
1. Use an approach where the price is dynamically computed based on changes in market price and use of the distribution network (depends on PV and load).
2. Use CREG 015/2018 to compute the prices for three different time intervals.
3. Just using market prices.

## 06/08/2019
Meeting with Adrian, Andres - Energy Management Project
- Datos: Maximo punto de potencia.
- Voltaje, Corriente panel / batería, potencia.
- Radiación en el panel, eficiencia del convertidor (algunos datos disponibles del IDEAM, otros provenientes de paneles instalados en el edificio de ingeniería).
- Resolución creg 030 de 2018: precio de la energía.
- Cargas residenciales (aleatorias, escuela).
- Precio de la energía: abierto (legislación muy reciente), usar varios casos con diferentes esquemas de remuneración.
- State of Charge: tratar de determinarlo para las baterías disponibles, buscar literatura, o asumir algunos valores mínimos y máximos.
- Definir entorno de modelamiento: GAMS o python.
- Script de GAMS disponible para un ejemplo de microred.
- Script de python - gurobi disponible.
- Esperar documentación, papers relevantes.
- Hablar con Diego Bernal, Alberto Avila para obtener datos.

# Energy Dispatch in Microgrid

## Components
- Photovoltaic system:
  - Solar panels: to include this panels in the mathematical problem we require to know the efficiency and the effective area.
  - DC-DC converter: We require to know the efficiency of this converter
- Battery storage: We require to know the maximum capacity of the battery bank
- Loads: We have to consider several cases. The first one is to have a load profile signal that characterizes the power consumption throughout the day. The second is when we have control over the loads that are turned on or off at each time instant.
- Distribution grid: This supplies the power that required when the solar panels and batteries are not enough

## Data needed in the mathematical model
- Price curves: I found the websites [powersmartpricing](https://www.powersmartpricing.org) and [hourlypricing](https://hourlypricing.comed.com). These offer day-ahead prices and real-time prices. Also there is an API that returns the prices for http requests.
- Solar radiation: The IDEAM website allows to download historical datasets with different variables from diferent stations. I could use this data to train a model for perform day-ahead predictions on the solar radiation levels. I implemented an estimator for forecasting day-ahead solar radiation values based on a history of past measurements for multiple variables. This was done using deep neural networks implemented in Tensorflow 2.0.
- Solar panel efficiency: This is required to estimate the output power of the photovoltaic system.

## Mathematical model
The first model we consider involves the use of day-ahead prices gathered from one of the websites mentioned before, a predefined load curve and solar radiation data gathered from the IDEAM website. The problem is to decide when to charge and discharge the batteries in order to minimize the cost of the energy drawn from the distribution grid. We have to use the solar radiation values to compute the power generated by the photovoltaic array. Let us define the following variables:
$t$: Time of day in hours.  
$p_d(t)$: Power drawn from the distribution grid at time $t$ in [kW].  
$r_v(t)$: Solar radiation at time $t$ in [kW/m^2].  
$p_v(t)$: Power generated by the photovoltaic array at time $t$ in [kW].  
$p_l(t)$: Power consumed by the loads at time $t$ in [kW].  
$p_{ch}(t)$: Charge power delivered to the batteries at time $t$ in [kW].  
$p_{dch}(t)$: Discharge power extracted from the batteries at time $t$ in [kW].  
$e_s(t)$: Stored energy in the batteries at time $t$ in [kWh].  
$c_d(t)$: Cost of energy drawn from the distribution grid at time $t$ in [USD].  
$\eta_{ch}$: Battery AC/DC charge efficiency  
$\eta_{dch}$: Battery AC/DC discharge efficiency  
$\eta_v$: Photovoltaic AC/DC converter efficiency  
