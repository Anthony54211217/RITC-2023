====
RITC-2023
====

Algorithmic trading programs written in Python for the Rotman International Trading Competition (RITC) 2023. 

The 2023 competition consisted of 4 different cases: Volatility Trading, Electricity Trading, Algorithmic Trading and Liquidity Risk. The volatility trading and algorithmic trading cases where algorithmic whereas the electricity and liquidity risk case were manual trading cases. Included in this repository are programs for the two algorithmic trading cases. 

I competed for the University of Toronto team and placed 4th overall, with notable placements of 1st place in the Algorithmic Trading Case and 3rd in the Volatility Trading Case. 

Due to the proprietary nature of the competition, the case package cannot be shared. 

The Volatility Trading Case involved the trading of 40 seperate European options contracts on a single stock. The case made simplifying assumptions which satisifed the assumptions of the Black-Scholes Model. During the case, estimates of future volatility are given to participants which combined with the Black-Scholes Model could be used to calculate a theoretical fair value of these options. Comparing current market prices and our theoretical fair value, we would be able to make trades. 

The Algorithmic Trading Case involved the trading of 2 currencies (CAD and USD), 2 individual stocks and an ETF containing the two stocks. The ETF traded in USD while the two individual stocks traded in CAD. Our first strategy was to use ETF arbitrage to make profit. However, due to the competitive nature of this strategy, and the fact that unusual trading caused the prices to converge very slowly, ETF arbitrage was not very profitable. So instead, we used a market making strategy. 
