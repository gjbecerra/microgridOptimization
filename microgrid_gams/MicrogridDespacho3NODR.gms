$title Despacho Microred con DR .$
$ontext

Despahco de Microred con microturbina, solar y baterias con DR

$offtext

sets
     t franja horaria /t1*t24/            ;

table datos(*,*) 'Datos de generadores'
*  dummy,  costo, Pmin,  Pmax
*  MicroT
*  Bat
$ondelim
$include tabladatosURED.csv
$offdelim
display datos;

scalar PNOM Potencia nominal de baterias /20/;
display PNOM;

table DyPV(t,*) 'demanda de cada franja horaria'
* dummy,dem,PV,Cbolsa
$ondelim
* $include DemandayPV24DR.csv
$include DemandayPV24DRGEO.csv
* $include DemandayPV24DRGEO2.csv
$offdelim
display DyPV;

variables
    costo costo del despacho
    Pred(t) Potencia entregada por la red
    PmicroT(t) Potencia generada por la uturbina
    PbatCh(t) Potencia de carga de la bateria
    PbatDCh(t) Potencia de descarga de la bateria
    Ebat(t) energia almacenada en la bateria
    Cut1(t) Demanda del corte 1
    Cut2(t) Demanda del corte 2
    Cut3(t) Demanda del corte 3
    CutCom(t) Complemento de la energia de los cortes;
    Pred.up(t)=50;
    Pred.lo(t)=0;
    pMicroT.up(t)=datos('MicroT','Pmax');
    PmicroT.lo(t)=0;
    PbatCh.up(t)=PNOM*datos('Bat','Pmax');
    PbatCh.lo(t)=0*datos('Bat','Pmin');
    PbatDCh.up(t)=PNOM*datos('Bat','Pmax');
    PbatDCh.lo(t)=0*datos('Bat','Pmin');
    Ebat.up(t)=0.8*PNOM*5;
    Ebat.lo(t)=0.2*PNOM*5;
    Cut1.up(t)=0*0.1*DyPV(t,'dem');
    Cut1.lo(t)=0;
    Cut2.up(t)=0*(0.1)*DyPV(t,'dem');
    Cut2.lo(t)=0;
    Cut3.up(t)=0*(0.10)*DyPV(t,'dem');
    Cut3.lo(t)=0;
*    CutCom.up(t)=0.5*DyPV(t,'dem');
    CutCom.lo(t)=0;

display Pred.up;
display PmicroT.up;
display PbatCh.up,PbatCh.lo;

equations
   FO funcion objetivo
   bal(t) balance de potencia
   Et1(t) carga de bateria
   Edesplazada Balance de Energia debido a DR
   Einifin Energia inicial y final de bateria;
   FO.. costo =e= sum(t,DyPV(t,'Cbolsa')*Pred(t)+500*PmicroT(t)+1*(38*Cut1(t)+56*Cut2(t)+114*Cut3(t)));
   bal(t).. Pred(t)-PbatCh(t)+PbatDCh(t)+PmicroT(t)+(25/20)*DyPV(t,'PV')-DyPV(t,'dem')+1*(Cut1(t)+Cut2(t)+Cut3(t))-CutCom(t) =e= 0;
   Et1(t)$(ord(t) gt 1) .. Ebat(t)=e=Ebat(t-1)+0.95*PbatCh(t-1)-PbatDCh(t-1)/0.95;
   Einifin .. Ebat('t1')=e=Ebat('t24');
   Edesplazada.. sum(t,Cut1(t)+Cut2(t)+Cut3(t))=e=sum(t,CutCom(t));
*   Et1(t)$(ord(t) gt 1) .. Ebat(t+1)=e=Ebat(t)+Pbat(t);
*   Et1(t)..

model DespachoMicroRed /all/;
solve DespachoMicroRed minimizing costo using lp;
display Ebat.l
execute_unload "results.gdx"  Ebat
execute 'gdxxrw.exe results.gdx var=Ebat.L'
*execute 'gdxxrw.exe results.gdx var=Pred.L'
*execute 'gdxxrw.exe results.gdx var=Pred.L range=A3:R24'



