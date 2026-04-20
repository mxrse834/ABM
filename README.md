# ABM

>>>>>GOAL :
Build an ABM that predicts inflation over time using real India data from MOSPI 
1)Get the data — MOSPI 
2)Build the ABM — 1M+ agents, simulate how prices evolve
3)Output — inflation prediction over time

>>>>>ABM Primitive(base) Model working :

1) Design:
Agents (firms, households, banks, CB(RBI)), their state variables, and behavioral rules.
    How to setup the state variables and behavioral rules(parameters are set at initialization and stay constant while state variables vary per iteration) :
        1.  What decisions does this agent make?
        2.  What info is needed to make that decision?
        3.  What minimal info can be used for this purpose?

REFERENCE: <!-- https://rbidocs.rbi.org.in/rdocs/Bulletin/PDFs/02ARTICLE24112025AE19C2E642BA40CCB8F2505CB4B406D7.PDF -->
Drawn inferences : 

1) Diesel up 10% → inflation up 0.9%
2) Exchange rate down 10% → inflation up 0.8%
3) Demand up 1% → inflation up 0.2% 



curent stage : pulled CPI DATA from gov stats , currently adjusting model parameters on the basis of this data ( in python)

