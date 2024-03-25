from dataclasses import dataclass, field
from math import erf, exp, log, sqrt, pi, floor
from os import system
from re import fullmatch
from time import sleep
from typing import Optional
from warnings import warn
import requests

from ritc import Case, News, Security, RIT

OPTIONS_ORDER_SIZE = 100
DELTA_MULTIPLIER = 0.9
TRIMMING_MULTIPLIER = 0.9
OPTIONS_POSITION_LIMIT = 900
DELTA_CORRECTION_THRESHOLD = 0.25  # means you correct delta when abs(portfolio delta) > 0.25 * Delta Limit
CLOSE_OUT_SPREAD = 0.035
REPLACEMENT_SPREAD = 0.08
AGGRESSIVE_BONUS_SPREAD = 0.06

X_API_KEY: str = 'CI99GO8J '
PARAMETERS_NEWS_BODY: str \
    = r'The current risk free rate is (?P<risk_free_rate>\d+)%\. RTM is an ETF that mimics one of the major indices in the simulated world and its current annualized volatility is (?P<volatility>\d+)%\. This heat consists of (?P<trading_day_count>\d+) trading days\.'
LIMITS_NEWS_BODY: str \
    = r'The delta limit for this heat is (?P<delta_limit>\d*,?\d+) and the penalty percentage is (?P<penalty_percentage>\d+)%'
EXPECTED_NEWS_BODY: str \
    = r'The annualized volatility of RTM is expected to be between (?P<min_volatility>\d+)% ~ (?P<max_volatility>\d+)% at the start of Week \d+'
LATEST_NEWS_BODY: str \
    = r'The latest annualized volatility of RTM is (?P<volatility>\d+)%'
STOCK_TICKER: str = 'RTM'
OPTION_TICKER: str \
    = r'RTM(?P<time_to_maturity>\d+)(?P<option_type>C|P)(?P<strike_price>\d+)'

s = requests.Session()
s.headers.update({'X-API-key': X_API_KEY})


def submit_market_order(ticker, quantity, action):
    resp = s.post('http://localhost:9999/v1/orders',
           params={'ticker': ticker, 'type': 'MARKET', 'quantity': quantity,
                   'action': action})
    # if resp.json != "calm"
    print(resp.json())


def submit_large_option_order(ticker, quantity, action):
    number_of_orders = quantity // OPTIONS_ORDER_SIZE  # example: 900 // 100 = 9, 950 // 100 = 9
    excess_quantity = quantity - (number_of_orders * OPTIONS_ORDER_SIZE)
    
    submitted = 0
    while submitted < number_of_orders:
        submit_market_order(ticker, OPTIONS_ORDER_SIZE, action)
        submitted += 1
    
    if excess_quantity != 0:
        submit_market_order(ticker, excess_quantity, action)


def submit_large_stock_order(ticker, quantity, action):
    number_of_orders = quantity // 10000
    excess_quantity = quantity - (number_of_orders * 10000)
    
    submitted = 0
    while submitted < number_of_orders:
        submit_market_order(ticker, 10000, action)
        submitted += 1
    
    if excess_quantity != 0:
        submit_market_order(ticker, excess_quantity, action)
        

def big_RTM_delta_hedge(current_delta, RTM_position):
    if current_delta > 0:
        quantity = round(current_delta)
        if RTM_position < -40000:  # To partially hedge if net limit is blocking us
            quantity = min(quantity, 50000 - abs(RTM_position))
        submit_large_stock_order('RTM', quantity, 'SELL')
        current_delta -= quantity
        RTM_position -= quantity
    elif current_delta < 0:
        quantity = abs(round(current_delta))
        if RTM_position > 40000:  # To partially hedge if net limit is blocking us
            quantity = min(quantity, 50000 - RTM_position)
        submit_large_stock_order('RTM', quantity, 'BUY')
        current_delta += quantity
        RTM_position += quantity

    return current_delta, RTM_position
    

@dataclass
class Algorithm:
    rit: RIT
    risk_free_rate: float = 0
    volatility: float = 0
    trading_day_count: int = 0
    delta_limit: int = 0
    penalty_percentage: float = 0
    latest_news_id: Optional[int] = None
    spot_price: float = 0
    realized_volatility: float = 0
    delta_hedges: dict = field(default_factory=lambda: {"RTM1C45": 0, "RTM1P54": 0, "RTM2C45": 0, "RTM2P54": 0})
    net_hedges: dict = field(default_factory=lambda: {"RTM1P45": 0, "RTM1C54": 0, "RTM2P45": 0, "RTM2C54": 0})
    arb_positions: dict = field(default_factory=lambda: {"RTM1C45": 0, "RTM1P45": 0, 
                                                         "RTM1C46": 0, "RTM1P46": 0, 
                                                         "RTM1C47": 0, "RTM1P47": 0,
                                                         "RTM1C48": 0, "RTM1P48": 0,
                                                         "RTM1C49": 0, "RTM1P49": 0,
                                                         "RTM1C50": 0, "RTM1P50": 0,
                                                         "RTM1C51": 0, "RTM1P51": 0,
                                                         "RTM1C52": 0, "RTM1P52": 0,
                                                         "RTM1C53": 0, "RTM1P53": 0,
                                                         "RTM1C54": 0, "RTM1P54": 0,
                                                         "RTM2C45": 0, "RTM2P45": 0, 
                                                         "RTM2C46": 0, "RTM2P46": 0, 
                                                         "RTM2C47": 0, "RTM2P47": 0,
                                                         "RTM2C48": 0, "RTM2P48": 0,
                                                         "RTM2C49": 0, "RTM2P49": 0,
                                                         "RTM2C50": 0, "RTM2P50": 0,
                                                         "RTM2C51": 0, "RTM2P51": 0,
                                                         "RTM2C52": 0, "RTM2P52": 0,
                                                         "RTM2C53": 0, "RTM2P53": 0,
                                                         "RTM2C54": 0, "RTM2P54": 0,
                                                         })

    def reset(self) -> None:
        self.risk_free_rate = 0
        self.volatility = 0
        self.trading_day_count = 0
        self.delta_limit = 0
        self.penalty_percentage = 0
        self.latest_news_id = None
        self.spot_price = 0
        self.realized_volatility = 0
        self.delta_hedges = {"RTM1C45": 0, "RTM1P54": 0, "RTM2C45": 0, "RTM2P54": 0}
        self.net_hedges = {"RTM1P45": 0, "RTM1C54": 0, "RTM2P45": 0, "RTM2C54": 0}
        self.arb_positions = {"RTM1C45": 0, "RTM1P45": 0, 
                            "RTM1C46": 0, "RTM1P46": 0, 
                            "RTM1C47": 0, "RTM1P47": 0,
                            "RTM1C48": 0, "RTM1P48": 0,
                            "RTM1C49": 0, "RTM1P49": 0,
                            "RTM1C50": 0, "RTM1P50": 0,
                            "RTM1C51": 0, "RTM1P51": 0,
                            "RTM1C52": 0, "RTM1P52": 0,
                            "RTM1C53": 0, "RTM1P53": 0,
                            "RTM1C54": 0, "RTM1P54": 0,
                            "RTM2C45": 0, "RTM2P45": 0, 
                            "RTM2C46": 0, "RTM2P46": 0, 
                            "RTM2C47": 0, "RTM2P47": 0,
                            "RTM2C48": 0, "RTM2P48": 0,
                            "RTM2C49": 0, "RTM2P49": 0,
                            "RTM2C50": 0, "RTM2P50": 0,
                            "RTM2C51": 0, "RTM2P51": 0,
                            "RTM2C52": 0, "RTM2P52": 0,
                            "RTM2C53": 0, "RTM2P53": 0,
                            "RTM2C54": 0, "RTM2P54": 0,
                            }
        sleep(1)

    def parse_news(self, news: News) -> None:
        self.latest_news_id = news.news_id

        if match := fullmatch(PARAMETERS_NEWS_BODY, news.body):
            self.risk_free_rate = float(match.group('risk_free_rate')) / 100
            self.volatility = float(match.group('volatility')) / 100
            self.realized_volatility = self.volatility
            print('ACTUAL VOLATILITY = {}'.format(self.realized_volatility))
            self.trading_day_count = int(match.group('trading_day_count'))
        elif match := fullmatch(LIMITS_NEWS_BODY, news.body):
            self.delta_limit = int(match.group('delta_limit').replace(',', ''))
            self.penalty_percentage \
                = float(match.group('penalty_percentage')) / 100
        elif match := fullmatch(EXPECTED_NEWS_BODY, news.body):
            min_volatility = float(match.group('min_volatility')) / 100
            max_volatility = float(match.group('max_volatility')) / 100

            # deviation_1 = abs(self.volatility - min_volatility)
            # deviation_2 = abs(self.volatility - max_volatility)
            #
            # if min_volatility <= self.volatility and self.volatility <= max_volatility: # if volatility is within the new volatility range
            #     warn('Volatility was within range')
            #     volatility_sum = min_volatility + max_volatility
            #     self.volatility = volatility_sum / 2
            # elif deviation_1 < deviation_2:
            #     self.volatility = min_volatility + 0.01
            # elif deviation_1 >= deviation_2:
            #     self.volatility = max_volatility - 0.01
            # else:
            #     warn('SHOULD NOT REACH HERE')

            volatility_sum = min_volatility + max_volatility
            self.volatility = volatility_sum / 2
        elif match := fullmatch(LATEST_NEWS_BODY, news.body):
            self.volatility = float(match.group('volatility')) / 100
            self.realized_volatility = self.volatility
            # print('ACTUAL VOLATILITY = {}'.format(self.realized_volatility))
        else:
            warn(f'unable to handle news \'{news}\'')

    def check_news(self) -> None:
        news = self.rit.get_news(since=self.latest_news_id)

        for item in news:
            self.parse_news(item)

    def get_option_values(
            self,
            strike_price: float,
            time_to_maturity: float,
    ) -> tuple[float, float, float, float, float, float, float, float, float]:
        def phi(x: float) -> float:
            return (1 + erf(x / sqrt(2))) / 2

        d1 = (log(self.spot_price / strike_price)
              + (self.risk_free_rate + self.volatility ** 2 / 2)
              * time_to_maturity) / (self.volatility * sqrt(time_to_maturity))
        d2 = d1 - self.volatility * sqrt(time_to_maturity)
        p1 = phi(d1)
        p2 = phi(d2)
        n1 = 1 - p1
        n2 = 1 - p2
        n_prime_d1 = (1/sqrt(2*pi))*exp((-(d1)**2)/2)
        
        d3 = (log(self.spot_price / strike_price)
              + (self.risk_free_rate + self.realized_volatility ** 2 / 2)
              * time_to_maturity) / (self.realized_volatility * sqrt(time_to_maturity))
        p3 = phi(d3)

# call price, put price, call delta, put delta, gamma, vega, call theta, put theta
        return (
            p1 * self.spot_price
            - p2 * strike_price * exp(-self.risk_free_rate * time_to_maturity), #call price
            n2 * strike_price * exp(-self.risk_free_rate * time_to_maturity)
            - n1 * self.spot_price, #put price
            exp(0)*p3, #call delta
            exp(0)*(p3 - 1), #put delta
            (exp(0)/(self.spot_price*self.volatility*sqrt(time_to_maturity)))*n_prime_d1, #gamma
            (1/100)*self.spot_price*exp(0)*sqrt(time_to_maturity)*n_prime_d1, #vega
            (1/336)*(-(((self.spot_price*self.volatility*exp(0))/(2*sqrt(time_to_maturity)))*n_prime_d1)), #call theta
            (1/336)*(-(((self.spot_price*self.volatility*exp(0))/(2*sqrt(time_to_maturity)))*n_prime_d1)), #put theta
            self.volatility, time_to_maturity, self.spot_price
            
        )

    def update_stock(self, case: Case, security: Security) -> tuple:
        assert security.ticker == STOCK_TICKER

        self.spot_price = (security.bid + security.ask) / 2

        # TODO
        tokens = (
            security.ticker,
            security.position,
            f'{self.spot_price:.2f}',
            f'{security.bid:.2f}',
            f'{security.ask:.2f}',
        )
        # print('\t'.join(tokens))
        return tokens

    def update_option(self, case: Case, security: Security) -> tuple:
        match = fullmatch(OPTION_TICKER, security.ticker)

        assert match

        strike_price = float(match.group('strike_price'))
        option_type = match.group('option_type')
        time_to_maturity = (float(match.group('time_to_maturity'))
                            - (case.period - 1)
                            - case.tick / case.ticks_per_period) / 12

        if time_to_maturity < 0:
            return

        try:
            option_values = self.get_option_values(
                strike_price,
                time_to_maturity,
            )
        except (ValueError, ZeroDivisionError):
            return

        option_gamma = option_values[4]
        option_vega = option_values[5]
        option_time = option_values[9]
        option_spot = option_values[10]
        option_volatility = option_values[8]

        if option_type == 'C':
            option_value = option_values[0]
            option_delta = option_values[2]
            option_theta = option_values[6]
        elif option_type == 'P':
            option_value = option_values[1]
            option_delta = option_values[3]
            option_theta = option_values[7]
        else:
            raise ValueError(f'unknown option type \'{option_type}\'')

        positive = None

        if option_value - security.ask > 0: # Should we buy the option?
            if option_delta > 0:  # if its a call option
                positive = True
            else:  # if its a put option
                positive = False

        elif option_value - security.bid < 0: # Should we sell the option?
            if option_delta > 0: # if its a call option, but becasue we're
            # selling it, it decreases our delta and thus, positive = False
                positive = False
            else: # if its a put option
                positive = True

        # TODO
        tokens = (
            security.ticker, #0
            security.position, #1
            option_value, #2
            (security.bid + security.ask) / 2, #3
            security.bid, #4
            security.ask, #5
            option_value - security.bid, #6
            option_value - security.ask, #7
            option_delta, #8
            option_gamma, #9
            option_vega, #10
            option_theta, #11
            option_time, #12
            option_spot, #13
            option_volatility, #14
            positive, #15
            option_type #16
        )
        # print('\t'.join(tokens))
        return tokens

    def period_2_values(self):
        self.delta_hedges['RTM1C45'] = 0
        self.delta_hedges['RTM1P54'] = 0
        self.net_hedges['RTM1P45'] = 0
        self.net_hedges['RTM1C54'] = 0
        self.arb_positions["RTM1C45"] = 0
        self.arb_positions["RTM1P45"] = 0
        self.arb_positions["RTM1C46"] = 0
        self.arb_positions["RTM1P46"] = 0
        self.arb_positions["RTM1C47"] = 0
        self.arb_positions["RTM1P47"] = 0
        self.arb_positions["RTM1C48"] = 0
        self.arb_positions["RTM1P48"] = 0
        self.arb_positions["RTM1C49"] = 0
        self.arb_positions["RTM1P49"] = 0
        self.arb_positions["RTM1C50"] = 0
        self.arb_positions["RTM1P50"] = 0
        self.arb_positions["RTM1C51"] = 0
        self.arb_positions["RTM1P51"] = 0
        self.arb_positions["RTM1C52"] = 0
        self.arb_positions["RTM1P52"] = 0
        self.arb_positions["RTM1C53"] = 0
        self.arb_positions["RTM1P53"] = 0
        self.arb_positions["RTM1C54"] = 0
        self.arb_positions["RTM1P54"] = 0


    def restart_case(self):
        case = self.rit.get_case()

        if case.status == Case.Status.ACTIVE:
            if case.period == 1 and case.tick <= 298:
                securities = self.rit.get_securities()
    
            else:
                lst = self.rit.get_securities()
                securities = [lst[i] for i in [1, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
        # securities = self.rit.get_securities()
            for security in securities:
                if security.type == Security.Type.OPTION:
                    tokens = self.update_option(case, security)
                    
                    if tokens[0] in self.net_hedges.keys():
                        self.net_hedges[tokens[0]] = tokens[1]
                    elif tokens[0] in self.delta_hedges.keys():
                        self.delta_hedges[tokens[0]] = tokens[1]
                    else:
                        self.arb_positions[tokens[0]] = tokens[1]
                    
                    # upate net position an gross position and delta (and furthermore exclude the hedging donnies)
                
                


    def trim_options_position(self, current_delta, RTM_position, net_options, gross_options):
        case = self.rit.get_case()
        if case.period == 1 and case.tick <= 298:
            securities = self.rit.get_securities()

        else:
            lst = self.rit.get_securities()
            securities = [lst[i] for i in [1, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
        # securities = self.rit.get_securities()
        
        if abs(RTM_position) == 50000:  # i.e. we are potentially in a position where delta can go over limit
            # print('reached here, Our Delta: {}, Delta Limit: {}'.format(portfolio_delta, self.delta_limit))
            if current_delta > TRIMMING_MULTIPLIER * self.delta_limit:
                smallest_misprice = 999999
                option_to_trim = ''
                action = ''
                delta_change = 0
                for security in securities:
                    if security.type == Security.Type.OPTION:
                        if self.arb_positions[security["ticker"]] != 0:
                            tokens = self.update_option(case, security)
                            if tokens[1] != 0 and tokens[15] == True and abs(tokens[7]) < smallest_misprice:  # if there is a position, delta is positive, and mispricing is the smallest
                                option_to_trim = tokens[0]  # This is now the option whose position we should trim
                                if tokens[1] > 0:  # means it is a call --> we have a long position on it
                                    action = 'SELL'
                                    delta_change = -tokens[8]
                                    # self.arb_positions[tokens[0]] -= OPTIONS_ORDER_SIZE
                                else:
                                    action = 'BUY'
                                    delta_change = tokens[8]
                                    # self.arb_positions[tokens[0]] += OPTIONS_ORDER_SIZE
                submit_market_order(option_to_trim, OPTIONS_ORDER_SIZE, action)
                current_delta += delta_change * OPTIONS_ORDER_SIZE * 100
                if action == 'SELL':
                    self.arb_positions[option_to_trim] -= OPTIONS_ORDER_SIZE
                    net_options -= OPTIONS_ORDER_SIZE
                else:
                    self.arb_positions[option_to_trim] += OPTIONS_ORDER_SIZE
                    net_options += OPTIONS_ORDER_SIZE
                gross_options -= OPTIONS_ORDER_SIZE
                print('TRIMMED POSITION TO DECREASE DELTA, ticker: {}'.format(option_to_trim))

            elif current_delta < -TRIMMING_MULTIPLIER * self.delta_limit:
                smallest_misprice = 999999
                option_to_trim = ''
                action = ''
                delta_change = 0
                for security in securities:
                    if security.type == Security.Type.OPTION:
                        if self.arb_positions[security['ticker']] != 0:
                            tokens = self.update_option(case, security)
                            if tokens[1] != 0 and tokens[15] == False and abs(tokens[7]) < smallest_misprice:  # if there is a position, delta is negative, and mispricing is the smallest
                                option_to_trim = tokens[0]  # This is now the option whose position we should trim
                                if tokens[1] > 0:  # means it is a put --> we have a long position on it
                                    action = 'SELL'
                                    delta_change = -tokens[8]
                                    # self.arb_positions[tokens[0]] -= OPTIONS_ORDER_SIZE
                                else:
                                    action = 'BUY'
                                    delta_change = tokens[8]
                                    # self.arb_positions[tokens[0]] += OPTIONS_ORDER_SIZE
                submit_market_order(option_to_trim, OPTIONS_ORDER_SIZE, action)
                current_delta += delta_change * OPTIONS_ORDER_SIZE * 100
                if action == 'SELL':
                    self.arb_positions[option_to_trim] -= OPTIONS_ORDER_SIZE
                    net_options -= OPTIONS_ORDER_SIZE
                else:
                    self.arb_positions[option_to_trim] += OPTIONS_ORDER_SIZE
                    net_options += OPTIONS_ORDER_SIZE
                gross_options -= OPTIONS_ORDER_SIZE
                print('TRIMMED POSITION TO INCREASE DELTA, ticker: {}'.format(option_to_trim))
        return current_delta, net_options, gross_options
    
    def close_arbitrages(self, portfolio_delta, net_options, gross_options) -> tuple:
        case = self.rit.get_case()
        if case.period == 1 and case.tick <= 298:
            securities = self.rit.get_securities()
        else:
            lst = self.rit.get_securities()
            securities = [lst[i] for i in [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
        
        # add a mechanism that closes arbitages when we are at the end of a week.
        if case.tick in [32,33,34,35,36,107,108,109,110,111,182,183,184,185,186,257,258,259,260,261]:
            extra = AGGRESSIVE_BONUS_SPREAD
        else:
            extra = 0
        
        closed = False
        for security in securities:
            if security.type == Security.Type.OPTION:
                tokens = self.update_option(case, security)
                quantity = OPTIONS_ORDER_SIZE
                # if tokens[1] > 0:  # if long position
                if tokens[6] < (CLOSE_OUT_SPREAD + extra) and self.arb_positions[tokens[0]] > 0 and net_options >= -(OPTIONS_POSITION_LIMIT):
                    submit_large_option_order(tokens[0], quantity, 'SELL')
                    portfolio_delta -= (quantity * 100 * tokens[8])
                    net_options -= quantity
                    gross_options -= quantity
                    closed = True
                    self.arb_positions[tokens[0]] -= quantity
                    print("Closed", tokens[0])
 
                # elif tokens[1] < 0:
                if tokens[7] > -(CLOSE_OUT_SPREAD + extra) and self.arb_positions[tokens[0]] < 0 and net_options <= (OPTIONS_POSITION_LIMIT):
                    submit_large_option_order(tokens[0], quantity, 'BUY')
                    portfolio_delta += (quantity * 100 * tokens[8])
                    net_options += quantity
                    gross_options -= quantity
                    closed = True
                    self.arb_positions[tokens[0]] += quantity
                    print("Closed", tokens[0])
        if closed:
            pass
            # print('PRE HEDGING DELTA: {}'.format(portfolio_delta))
        return portfolio_delta, net_options, gross_options

    def get_acceptable_spread(self) -> float:
        case = self.rit.get_case()
        volatility_difference = round(abs(self.volatility - self.realized_volatility), 4)
        
        if case.period == 1:
            # if volatility_difference >= 0.05:
            #     spread = 0.5 + ((volatility_difference - 0.045) * 7)
            # elif volatility_difference >= 0.02:
            #     spread = 0.3 + ((volatility_difference - 0.025) * 10)
            # else:
            spread = 0.2 + ((volatility_difference - 0.005)* 5)
        else:
            multiplier = 1
            if case.tick > 180:
                multiplier = 0.75
            spread = (0.175 + ((volatility_difference - 0.005) * 2.5)) * multiplier
        
        if -0.0002 < volatility_difference < 0.0002:
            spread = 0.3
        
        spread = round(spread, 4)
        # print('Spread Threshold = {}'.format(spread))
        return spread
    
    
    def find_max_spread(self, case, securities) -> tuple:
        portfolio_delta = 0
        net_options = 0
        gross_options = 0
        stock_position = 0
        arbitrage_amount_positive = 0
        arbitrage_amount_negative = 0
        arbitrage_ticker_positive = None
        arbitrage_ticker_negative = None
        arbitrage_ticker_type_positive = None #not sure if it should be None to begin with
        arbitrage_ticker_type_negative = None
        arbitrage_delta_positive = 0
        arbitrage_delta_negative = 0
        arbitrage_position_positive = 0
        arbitrage_position_negative = 0

        # Loop through securites list to identify and store arbitrage opportunities #

        for security in securities:

            if security.type == Security.Type.OPTION:
                tokens = self.update_option(case, security)
                portfolio_delta += tokens[1]*100*tokens[8]
                net_options += tokens[1]
                gross_options += abs(tokens[1])

                # This part gets you the info for the options with the most mispricing #
                if tokens[15] == True:
                    if (tokens[16] == "C") & (tokens[7] > arbitrage_amount_positive):
                        arbitrage_amount_positive = tokens[7]
                        arbitrage_position_positive = self.arb_positions[tokens[0]]
                        arbitrage_ticker_positive = tokens[0]
                        arbitrage_ticker_type_positive = "C"
                        arbitrage_delta_positive = tokens[8]
                        
                    elif (tokens[16] == "P") & (abs(tokens[6]) > arbitrage_amount_positive):
                        arbitrage_amount_positive = abs(tokens[6])
                        arbitrage_position_positive = self.arb_positions[tokens[0]]
                        arbitrage_ticker_positive = tokens[0]
                        arbitrage_ticker_type_positive = "P"
                        arbitrage_delta_positive = tokens[8]

                elif tokens[15] == False:
                    if (tokens[16] == "C") & (abs(tokens[6]) > arbitrage_amount_negative):
                        arbitrage_amount_negative = abs(tokens[6])
                        arbitrage_position_negative = self.arb_positions[tokens[0]]
                        arbitrage_ticker_negative = tokens[0]
                        arbitrage_ticker_type_negative = "C"
                        arbitrage_delta_negative = tokens[8]
                        
                    elif (tokens[16] == "P") & (tokens[7] > arbitrage_amount_negative):
                        arbitrage_amount_negative = tokens[7]
                        arbitrage_position_negative = self.arb_positions[tokens[0]]
                        arbitrage_ticker_negative = tokens[0]
                        arbitrage_ticker_type_negative = "P"
                        arbitrage_delta_negative = tokens[8]
                        
            elif security.type == Security.Type.STOCK:  
                tokens = self.update_stock(case, security)
                portfolio_delta += tokens[1]
                stock_position = tokens[1]

            else:
                warn(f'unknown security \'{security}\'')
                
        return (portfolio_delta, 
                net_options,
                gross_options,
                stock_position,
                arbitrage_amount_positive, 
                arbitrage_position_positive,
                arbitrage_ticker_positive, 
                arbitrage_ticker_type_positive, 
                arbitrage_delta_positive,
                arbitrage_amount_negative, 
                arbitrage_position_negative,
                arbitrage_ticker_negative, 
                arbitrage_ticker_type_negative,
                arbitrage_delta_negative)

    def free_net(self, portfolio_delta, net_options, gross_options):
        if gross_options >= 2400:
            return portfolio_delta, net_options, gross_options
        
        case = self.rit.get_case()
        if case.period == 1 and case.tick <= 298:
            securities = self.rit.get_securities()
            P45 = securities[2]
            C54 = securities[19]
        else:
            lst = self.rit.get_securities()
            securities = [lst[i] for i in [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
            P45 = securities[2]
            C54 = securities[19]

        # These are the two possible candidates of OTM options to use
        tokens1 = self.update_option(case, P45)
        tokens2 = self.update_option(case, C54)
        
        if net_options == 1000: # net position is 1000, need to sell an option to free room
            if abs(tokens1[8]) < abs(tokens2[8]): # use the option that has the lowest delta (positive or negative) # this tells us the more OTM option
                submit_market_order(tokens1[0], OPTIONS_ORDER_SIZE, "SELL")
                portfolio_delta -= tokens1[8]*OPTIONS_ORDER_SIZE*100
                self.net_hedges[tokens1[0]] -= OPTIONS_ORDER_SIZE
                if tokens1[1] > 0:
                    gross_options -= OPTIONS_ORDER_SIZE
                elif tokens1[1] <= 0:
                    gross_options += OPTIONS_ORDER_SIZE
                net_options -= OPTIONS_ORDER_SIZE
                print("Freed net position by selling {}".format(tokens1[0]))
            else:
                submit_market_order(tokens2[0], OPTIONS_ORDER_SIZE, "SELL")
                portfolio_delta -= tokens2[8]*OPTIONS_ORDER_SIZE*100
                self.net_hedges[tokens2[0]] -= OPTIONS_ORDER_SIZE
                if tokens2[1] > 0:
                    gross_options -= OPTIONS_ORDER_SIZE
                elif tokens2[1] <= 0:
                    gross_options += OPTIONS_ORDER_SIZE
                net_options -= OPTIONS_ORDER_SIZE 
                print("Freed net position by selling {}".format(tokens2[0]))
                    
        elif net_options == -1000: # net position is -1000, need to buy an option to free room
            if abs(tokens1[8]) < abs(tokens2[8]):  # use the option that has the lowest delta (positive or negative)
                submit_market_order(tokens1[0], OPTIONS_ORDER_SIZE, "BUY")
                portfolio_delta += tokens1[8]*OPTIONS_ORDER_SIZE*100
                self.net_hedges[tokens1[0]] += OPTIONS_ORDER_SIZE
                if tokens1[1] >= 0:
                    gross_options += OPTIONS_ORDER_SIZE
                elif tokens1[1] < 0:
                    gross_options -= OPTIONS_ORDER_SIZE
                net_options -= OPTIONS_ORDER_SIZE
                print("Freed net position by buying {}".format(tokens1[0]))
            else:
                submit_market_order(tokens2[0], OPTIONS_ORDER_SIZE, "BUY")
                portfolio_delta += tokens2[8]*OPTIONS_ORDER_SIZE*100
                self.net_hedges[tokens2[0]] += OPTIONS_ORDER_SIZE
                if tokens2[1] >= 0:
                    gross_options += OPTIONS_ORDER_SIZE
                elif tokens2[1] < 0:
                    gross_options -= OPTIONS_ORDER_SIZE
                net_options -= OPTIONS_ORDER_SIZE
                print("Freed net position by buying {}".format(tokens2[0]))
                
        return portfolio_delta, net_options, gross_options
    
    def close_net_hedges(self, portfolio_delta, net_options, gross_options):
        """
        TODO MAKE CODE MORE EFFICIENT BY REMOVING THE WHOLE LIST OF SECURITIES
        We know that net hedges only has 4 options. TODO TODO TODO
        """
        case = self.rit.get_case()
        if case.period == 1 and case.tick <= 298:
            securities = self.rit.get_securities()
        else:
            lst = self.rit.get_securities()
            securities = [lst[i] for i in [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
            
        for security in securities:
            if security.type == Security.Type.OPTION:
                tokens = self.update_option(case, security)
                if tokens[0] in self.net_hedges.keys():
                    if self.net_hedges[tokens[0]] != 0:
                        if self.net_hedges[tokens[0]] > 0:
                            if net_options - OPTIONS_ORDER_SIZE > -1000:  # "If we have space to sell"
                                submit_market_order(tokens[0], OPTIONS_ORDER_SIZE, "SELL")
                                net_options -= OPTIONS_ORDER_SIZE
                                gross_options -= OPTIONS_ORDER_SIZE  # always does this cuz we are long the option
                                portfolio_delta -= tokens[8]*OPTIONS_ORDER_SIZE*100
                                self.net_hedges[tokens[0]] -= OPTIONS_ORDER_SIZE
                                print("Closed a free net position by selling {}".format(tokens[0]))
                        elif self.net_hedges[tokens[0]] < 0:
                            if net_options + OPTIONS_ORDER_SIZE < 1000:  # "if we have space to buy"
                                submit_market_order(tokens[0], OPTIONS_ORDER_SIZE, "BUY")
                                net_options += OPTIONS_ORDER_SIZE
                                gross_options -= OPTIONS_ORDER_SIZE  # we were short before, so gross decreases
                                portfolio_delta += tokens[8]*OPTIONS_ORDER_SIZE*100
                                self.net_hedges[tokens[0]] += OPTIONS_ORDER_SIZE
                                print("Closed a free net position by buying {}".format(tokens[0]))
        return portfolio_delta, net_options, gross_options
    
    def delta_hedge_options(self, portfolio_delta, net_options, gross_options, stock_position):
        if gross_options >= 2400:
            return portfolio_delta, net_options, gross_options, stock_position
        total = 0
        for val in self.delta_hedges.values():
            total += abs(val)
        if total >= 300:
            return portfolio_delta, net_options, gross_options, stock_position
        
        case = self.rit.get_case()
        if case.period == 1 and case.tick <= 298:
            securities = self.rit.get_securities()
            C45 = securities[1]
            P54 = securities[20]
        else:
            lst = self.rit.get_securities()
            securities = [lst[i] for i in [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
            C45 = securities[1]
            P54 = securities[20]
        
        # To increase delta, can either buy call or sell put
        tokens1 = self.update_option(case, C45) # delta = +1
        tokens2 = self.update_option(case, P54) # delta = -1

        if stock_position > 48000: # want to replace long position with long delta, i.e. buy deep ITM call or sell deep ITM put
            if abs(tokens1[8]) > abs(tokens2[8]) and net_options <= OPTIONS_POSITION_LIMIT:  # use the option with a higher delta
                submit_market_order(tokens1[0], OPTIONS_ORDER_SIZE, "BUY")  # buy the call
                submit_market_order("RTM", round(OPTIONS_ORDER_SIZE*100*abs(tokens1[8])), "SELL")
                stock_position -= round(OPTIONS_ORDER_SIZE*100*abs(tokens1[8]))
                net_options += OPTIONS_ORDER_SIZE
                if tokens1[1] >= 0:
                    gross_options += OPTIONS_ORDER_SIZE
                elif tokens1[1] < 0:
                    gross_options -= OPTIONS_ORDER_SIZE
                self.delta_hedges[tokens1[0]] += OPTIONS_ORDER_SIZE
                print("Added to the delta hedge by selling {}".format(tokens1[0]))
            elif abs(tokens2[8]) > abs(tokens1[8]) and net_options >= -OPTIONS_POSITION_LIMIT:  # So it is a put
                submit_market_order(tokens2[0], OPTIONS_ORDER_SIZE, "SELL")
                submit_market_order("RTM", round(OPTIONS_ORDER_SIZE*100*abs(tokens2[8])), "SELL")
                stock_position -= round(OPTIONS_ORDER_SIZE*100*abs(tokens2[8]))
                net_options -= OPTIONS_ORDER_SIZE
                if tokens2[1] > 0:
                    gross_options -= OPTIONS_ORDER_SIZE
                elif tokens2[1] <= 0:
                    gross_options += OPTIONS_ORDER_SIZE
                self.delta_hedges[tokens2[0]] -= OPTIONS_ORDER_SIZE
                print("Added to the delta hedge by selling {}".format(tokens2[0]))
                
        elif stock_position < -48000:
            if abs(tokens1[8]) > abs(tokens2[8]) and net_options >= -OPTIONS_POSITION_LIMIT:  # use the option with a higher delta
                submit_market_order(tokens1[0], OPTIONS_ORDER_SIZE, "SELL")  # sell call
                submit_market_order("RTM", round(OPTIONS_ORDER_SIZE*100*abs(tokens1[8])), "BUY")
                stock_position += round(OPTIONS_ORDER_SIZE*100*abs(tokens1[8]))
                net_options -= OPTIONS_ORDER_SIZE
                if tokens1[1] > 0:
                    gross_options -= OPTIONS_ORDER_SIZE
                elif tokens1[1] <= 0:
                    gross_options += OPTIONS_ORDER_SIZE
                self.delta_hedges[tokens1[0]] -= OPTIONS_ORDER_SIZE
                print("Added to the delta hedge by buying {}".format(tokens1[0]))
            
            elif abs(tokens2[8]) > abs(tokens1[8]) and net_options <= OPTIONS_POSITION_LIMIT:
                submit_market_order(tokens2[0], OPTIONS_ORDER_SIZE, "BUY")  # buy put
                submit_market_order("RTM", round(OPTIONS_ORDER_SIZE*100*abs(tokens2[8])), "BUY")
                stock_position += round(OPTIONS_ORDER_SIZE*100*abs(tokens2[8]))
                net_options += OPTIONS_ORDER_SIZE
                if tokens2[1] >= 0:
                    gross_options += OPTIONS_ORDER_SIZE
                elif tokens2[1] < 0:
                    gross_options -= OPTIONS_ORDER_SIZE
                self.delta_hedges[tokens2[0]] += OPTIONS_ORDER_SIZE
                print("Added to the delta hedge by buying {}".format(tokens2[0]))
                
        return portfolio_delta, net_options, gross_options, stock_position
    
    def close_delta_hedges_options(self, portfolio_delta, net_options, gross_options, stock_position):
        case = self.rit.get_case()
        if case.period == 1 and case.tick <= 298:
            securities = self.rit.get_securities()
        else:
            lst = self.rit.get_securities()
            securities = [lst[i] for i in [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
            
        for security in securities:
            if security.type == Security.Type.OPTION:
                tokens = self.update_option(case, security)
                if tokens[0] in self.delta_hedges.keys():
                    if self.delta_hedges[tokens[0]] != 0:
                        if self.delta_hedges[tokens[0]] > 0:  # if long position
                            if tokens[16] == 'C':
                                if stock_position + round(OPTIONS_ORDER_SIZE*tokens[8]*100) < 40000 and net_options >= -OPTIONS_POSITION_LIMIT:  # if RTM doesn't exceed 25k to correct this option close out
                                    submit_market_order(tokens[0], OPTIONS_ORDER_SIZE, "SELL")
                                    submit_market_order('RTM', round(OPTIONS_ORDER_SIZE*tokens[8]*100), "BUY")
                                    net_options -= OPTIONS_ORDER_SIZE
                                    gross_options -= OPTIONS_ORDER_SIZE  # always true cuz we are long
                                    self.delta_hedges[tokens[0]] -= OPTIONS_ORDER_SIZE
                                    stock_position += round(OPTIONS_ORDER_SIZE*tokens[8]*100) 
                                    print("Closed a delta hedge position by selling {}".format(tokens[0]))
                            elif tokens[16] == 'P':
                                if stock_position - round(OPTIONS_ORDER_SIZE*abs(tokens[8])*100) > -40000 and net_options >= -OPTIONS_POSITION_LIMIT:
                                    submit_market_order(tokens[0], OPTIONS_ORDER_SIZE, "SELL")
                                    submit_market_order('RTM', round(OPTIONS_ORDER_SIZE*abs(tokens[8])*100), "SELL")
                                    net_options -= OPTIONS_ORDER_SIZE
                                    gross_options -= OPTIONS_ORDER_SIZE
                                    self.delta_hedges[tokens[0]] -= OPTIONS_ORDER_SIZE
                                    stock_position -= round(OPTIONS_ORDER_SIZE*abs(tokens[8])*100)
                                    print("Closed a delta hedge position by selling {}".format(tokens[0]))
                        elif self.delta_hedges[tokens[0]] < 0:  # if short position
                            if tokens[16] == 'C':
                                if stock_position - round(OPTIONS_ORDER_SIZE*tokens[8]*100) > -40000 and net_options <= OPTIONS_POSITION_LIMIT:
                                    submit_market_order(tokens[0], OPTIONS_ORDER_SIZE, "BUY")
                                    submit_market_order('RTM', round(OPTIONS_ORDER_SIZE*tokens[8]*100), "SELL")
                                    net_options += OPTIONS_ORDER_SIZE
                                    gross_options -= OPTIONS_ORDER_SIZE
                                    self.delta_hedges[tokens[0]] += OPTIONS_ORDER_SIZE
                                    stock_position -= round(OPTIONS_ORDER_SIZE*tokens[8]*100) 
                                    print("Closed a delta hedge position by buying {}".format(tokens[0]))
                            elif tokens[16] == 'P':
                                if stock_position + round(OPTIONS_ORDER_SIZE*abs(tokens[8])*100) < 40000 and net_options <= OPTIONS_POSITION_LIMIT:
                                    submit_market_order(tokens[0], OPTIONS_ORDER_SIZE, "BUY")
                                    submit_market_order('RTM', round(OPTIONS_ORDER_SIZE*abs(tokens[8])*100), "BUY")
                                    net_options += OPTIONS_ORDER_SIZE
                                    gross_options -= OPTIONS_ORDER_SIZE
                                    self.delta_hedges[tokens[0]] += OPTIONS_ORDER_SIZE
                                    stock_position += round(OPTIONS_ORDER_SIZE*abs(tokens[8])*100)
                                    print("Closed a delta hedge position by buying {}".format(tokens[0]))
                                    
        return portfolio_delta, net_options, gross_options, stock_position
    
    def recalculate(self, case, securities):
        for security in securities:
            if security.type == Security.Type.OPTION:
                tokens = self.update_option(case, security)
                if tokens[0] in self.net_hedges.keys():
                    self.net_hedges[tokens[0]] = tokens[1]
                elif tokens[0] in self.delta_hedges.keys():
                    self.delta_hedges[tokens[0]] = tokens[1]
                else:
                    self.arb_positions[tokens[0]] = tokens[1]

    
    def run(self) -> None:
        period_2_correction = 0
        period_2_recalculation = 0
        while True:
            case = self.rit.get_case()

            if case.status == Case.Status.ACTIVE:
                sleep(0.3)  # TODO # putting this in the start to see what happens
                
                ## Get securites list depending on the period ##
                
                self.check_news()
                system('cls')  # TODO
                if case.period == 1 and case.tick <= 298:
                    securities = self.rit.get_securities()
                    number_of_iterations = 1
                else:
                    securities = [self.rit.get_securities()[i] for i in [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
                    period_2_correction += 1
                    number_of_iterations = 2
                    if period_2_correction == 1:
                        self.period_2_values()
                    # if case.period == 2 and case.tick > 2 and period_2_recalculation == 0:
                    #     period_2_recalculation += 1
                    #     self.recalculate(case, securities)
                    #     sleep(1)    
                
                # Finds max arbitrage opportunities. and upates portfolio_delta, 
                # net_options, gross_options, stock_position using RIT values
                
                portfolio_delta, net_options, gross_options, stock_position, \
                    arbitrage_amount_positive, arbitrage_position_positive, arbitrage_ticker_positive, arbitrage_ticker_type_positive, arbitrage_delta_positive, \
                    arbitrage_amount_negative, arbitrage_position_negative, arbitrage_ticker_negative, arbitrage_ticker_type_negative, arbitrage_delta_negative  \
                        = self.find_max_spread(case, securities)
                
                
                # Closes Options Positions #
                portfolio_delta, net_options, gross_options = self.close_arbitrages(portfolio_delta, net_options, gross_options)
                sleep(0.1)
                
                # Hedges position # TODO CONSIDER REMOVING
                if abs(portfolio_delta) >= (DELTA_CORRECTION_THRESHOLD * self.delta_limit):
                    portfolio_delta, stock_position = big_RTM_delta_hedge(portfolio_delta, stock_position)
                        
                self.free_net(portfolio_delta, net_options, gross_options) # free up net position
                
                # Hedges position #
                if abs(portfolio_delta) >= (DELTA_CORRECTION_THRESHOLD * self.delta_limit):
                    portfolio_delta, stock_position = big_RTM_delta_hedge(portfolio_delta, stock_position)
                
                # REPLACEMENT MECHANISM
                
                if abs(net_options) == 1000:  # if we are maxed out at our options position, look for better misprice
                    best_call_ticker = ''
                    biggest_call_misprice = 0
                    best_call_delta = 0

                    best_put_ticker = ''
                    biggest_put_misprice = 0
                    best_put_delta = 0
                    for security in securities:  # This loops for the best call and the best put available right now
                        if security.type == Security.Type.OPTION:
                            tokens = self.update_option(case, security)
                            if tokens[16] == 'C':
                                if abs(tokens[7]) > abs(biggest_call_misprice):
                                    best_call_ticker = tokens[0]
                                    biggest_call_misprice = tokens[7]
                                    best_call_delta = tokens[8]
                            if tokens[16] == 'P':
                                if abs(tokens[7]) > abs(biggest_put_misprice):
                                    best_put_ticker = tokens[0]
                                    biggest_put_misprice = tokens[7]
                                    best_put_delta = tokens[8]

                    worst_call_misprice = 100
                    worst_call_ticker = ''
                    worst_call_delta = 0
                    worst_call_position = 0

                    worst_put_misprice = 100
                    worst_put_ticker = ''
                    worst_put_delta = 0
                    worst_put_position = 0
                    for security in securities:  # This loops for the worst call and that put we have a position on
                        if security.type == Security.Type.OPTION:
                            # print('reached HERE')
                            tokens = self.update_option(case, security)
                            if self.arb_positions[tokens[0]] != 0 and tokens[16] == 'C':  # if have position and it is a call
                                # if tokens[7] < worst_call_misprice:
                                if abs(tokens[7]) < abs(worst_call_misprice):
                                    worst_call_ticker = tokens[0]
                                    worst_call_misprice = tokens[7]
                                    worst_call_delta = tokens[8]
                                    worst_call_position = self.arb_positions[tokens[0]]
                            if self.arb_positions[tokens[0]] != 0 and tokens[16] == 'P':  # if have position and it is a put
                                # if tokens[7] < worst_put_misprice:
                                if abs(tokens[7]) < abs(worst_put_misprice):
                                    worst_put_ticker = tokens[0]
                                    worst_put_misprice = tokens[7]
                                    worst_put_delta = tokens[8]
                                    worst_put_position = self.arb_positions[tokens[0]]
                    
                    quantity = OPTIONS_ORDER_SIZE
                    if worst_call_ticker != best_call_ticker:  # MAKE SURE YOU ARE NOT REPLACING SAME ONE
                        if worst_call_misprice > 0 and biggest_call_misprice > 0:
                            if biggest_call_misprice - worst_call_misprice > REPLACEMENT_SPREAD:
                                if worst_call_position > 0:
                                    # quantity = abs(worst_call_position)
                                    # submit_large_option_order(worst_call_ticker, quantity, 'SELL')
                                    submit_market_order(worst_call_ticker, quantity, 'SELL')  # LOOK INTO DOING TWO ORDERS AT ONCE
                                    portfolio_delta -= worst_call_delta * quantity * 100
                                    self.arb_positions[worst_call_ticker] -= quantity
                                    sleep(0.1)  # TODO CHANGE THIS TO RESP.OK
                                    # submit_large_option_order(best_call_ticker, quantity, 'BUY')
                                    submit_market_order(best_call_ticker, quantity, 'BUY')
                                    portfolio_delta += best_call_delta * quantity * 100
                                    self.arb_positions[best_call_ticker] += quantity
                                    # transacted = True
                                    print('Sold {} to BUY {}, Spread Difference = {}  Quantity = {}'.format(
                                        worst_call_ticker, best_call_ticker,
                                        biggest_call_misprice - worst_call_misprice, quantity))

                        elif worst_call_misprice < 0 and biggest_call_misprice < 0:
                            if abs(biggest_call_misprice) - abs(worst_call_misprice) > REPLACEMENT_SPREAD:
                                if worst_call_position < 0:
                                    # quantity = abs(worst_call_position)
                                    # submit_large_option_order(worst_call_ticker, quantity, 'BUY')
                                    submit_market_order(worst_call_ticker, quantity, 'BUY')
                                    portfolio_delta += worst_call_delta * quantity * 100
                                    self.arb_positions[worst_call_ticker] += quantity
                                    sleep(0.1)  # TODO CHANGE THIS TO RESP.OK
                                    # submit_large_option_order(best_call_ticker, quantity, 'SELL')
                                    submit_market_order(best_call_ticker, quantity, 'SELL')
                                    portfolio_delta -= best_call_delta * quantity * 100
                                    self.arb_positions[best_call_ticker] -= quantity
                                    # transacted = True
                                    print('Bought {} to SELL {}, Spread Difference = {}  Quantity = {}'.format(
                                        worst_call_ticker, best_call_ticker,
                                        abs(biggest_call_misprice) - abs(worst_call_misprice), quantity))

                    if worst_put_ticker != best_put_ticker:  # MAKE SURE YOU ARE NOT REPLACING SAME ONE
                        if worst_put_misprice > 0 and biggest_put_misprice > 0:
                            if biggest_put_misprice - worst_put_misprice > REPLACEMENT_SPREAD:
                                if worst_put_position > 0:  # just a confirmation
                                    # quantity = abs(worst_put_position)
                                    # submit_large_option_order(worst_put_ticker, quantity, 'SELL')
                                    submit_market_order(worst_put_ticker, quantity, 'SELL')
                                    portfolio_delta += abs(worst_put_delta) * quantity * 100
                                    self.arb_positions[worst_put_ticker] -= quantity
                                    sleep(0.1)  # TODO CHANGE THIS TO RESP.OK
                                    # submit_large_option_order(best_put_ticker, quantity, 'BUY')
                                    submit_market_order(best_put_ticker, quantity, 'BUY')
                                    portfolio_delta -= abs(best_put_delta) * quantity * 100
                                    self.arb_positions[best_put_ticker] += quantity
                                    # transacted = True
                                    print('Sold {} to BUY {}, Spread Difference = {}  Quantity = {}'.format(
                                        worst_put_ticker, best_put_ticker, biggest_put_misprice - worst_put_misprice,
                                        quantity))

                        elif worst_put_misprice < 0 and biggest_put_misprice < 0:
                            if abs(biggest_put_misprice) - abs(worst_put_misprice) > REPLACEMENT_SPREAD:
                                if worst_put_position < 0:  # just a confirmation
                                    # quantity = abs(worst_put_position)
                                    # submit_large_option_order(worst_put_ticker, quantity, 'BUY')
                                    submit_market_order(worst_put_ticker, quantity, 'BUY')
                                    portfolio_delta -= abs(worst_put_delta) * quantity * 100
                                    self.arb_positions[worst_put_ticker] += quantity
                                    sleep(0.1)  # TODO CHANGE THIS TO RESP.OK
                                    # submit_large_option_order(best_put_ticker, quantity, 'SELL')
                                    submit_market_order(best_put_ticker, quantity, 'SELL')
                                    portfolio_delta += abs(best_put_delta) * quantity * 100
                                    self.arb_positions[best_put_ticker] -= quantity
                                    # transacted = True
                                    print('Bought {} to SELL {}, Spread Difference = {}  Quantity = {}'.format(
                                        worst_put_ticker, best_put_ticker,
                                        abs(biggest_put_misprice) - abs(worst_put_misprice), quantity))
                    # if transacted:
                    #     sleep(0.35)
                
                # Get minimum acceptable misprice spread #
                minimum_misprice_spread = self.get_acceptable_spread()

                trading_ticks = [(37, 75), (112, 150), (187, 225), (262, 300)]
                ## Collect arbitrage
                for i in range(number_of_iterations):
                    if case.tick <= trading_ticks[floor(case.tick/75)][1] and case.tick >= trading_ticks[floor(case.tick/75)][0]:  # To avoid trading at bad spread because ANON move price before first news estimate
    
                        if (arbitrage_amount_positive > minimum_misprice_spread) and (portfolio_delta < (DELTA_MULTIPLIER * self.delta_limit)) and abs(stock_position) != 50000 and gross_options <= 2400:
                            delta_change = round(arbitrage_delta_positive*OPTIONS_ORDER_SIZE*100)
    
                            if (arbitrage_ticker_type_positive == "C") & (net_options <= OPTIONS_POSITION_LIMIT):
                                submit_market_order(arbitrage_ticker_positive, OPTIONS_ORDER_SIZE, 'BUY')
                                self.arb_positions[arbitrage_ticker_positive] += OPTIONS_ORDER_SIZE
                                net_options += OPTIONS_ORDER_SIZE
                                portfolio_delta += delta_change
                                if arbitrage_position_positive > 0:
                                    gross_options += OPTIONS_ORDER_SIZE
                                elif arbitrage_position_positive < 0:  # if current position of option is negative
                                    gross_options -= OPTIONS_ORDER_SIZE
        
                            elif (arbitrage_ticker_type_positive == "P") & (net_options >= -OPTIONS_POSITION_LIMIT):
                                submit_market_order(arbitrage_ticker_positive, OPTIONS_ORDER_SIZE, 'SELL')
                                self.arb_positions[arbitrage_ticker_positive] -= OPTIONS_ORDER_SIZE
                                net_options -= OPTIONS_ORDER_SIZE
                                portfolio_delta -= delta_change
                                if arbitrage_position_positive > 0:
                                    gross_options -= OPTIONS_ORDER_SIZE
                                elif arbitrage_position_positive < 0:
                                    gross_options += OPTIONS_ORDER_SIZE
    
                        if (arbitrage_amount_negative > minimum_misprice_spread) and (portfolio_delta > (-DELTA_MULTIPLIER * self.delta_limit)) and abs(stock_position) != 50000 and gross_options <= 2400:
                            delta_change = round(abs(arbitrage_delta_negative)*OPTIONS_ORDER_SIZE*100)
    
                            if (arbitrage_ticker_type_negative == "C") & (net_options >= -OPTIONS_POSITION_LIMIT):
                                submit_market_order(arbitrage_ticker_negative, OPTIONS_ORDER_SIZE, 'SELL')
                                self.arb_positions[arbitrage_ticker_negative] -= OPTIONS_ORDER_SIZE
                                net_options += -OPTIONS_ORDER_SIZE
                                portfolio_delta -= delta_change
                                if arbitrage_position_negative > 0:
                                    gross_options -= OPTIONS_ORDER_SIZE
                                elif arbitrage_position_negative < 0:
                                    gross_options += OPTIONS_ORDER_SIZE
    
                            elif (arbitrage_ticker_type_negative == "P") & (net_options <= OPTIONS_POSITION_LIMIT):
                                submit_market_order(arbitrage_ticker_negative, OPTIONS_ORDER_SIZE, 'BUY')
                                self.arb_positions[arbitrage_ticker_negative] += OPTIONS_ORDER_SIZE
                                net_options += OPTIONS_ORDER_SIZE
                                portfolio_delta -= delta_change
                                if arbitrage_position_negative > 0:
                                    gross_options += OPTIONS_ORDER_SIZE
                                elif arbitrage_position_negative < 0:
                                    gross_options -= OPTIONS_ORDER_SIZE
                
                
                
                # Hedges position #
                if abs(portfolio_delta) >= (DELTA_CORRECTION_THRESHOLD * self.delta_limit):
                    portfolio_delta, stock_position = big_RTM_delta_hedge(portfolio_delta, stock_position)
                
                portfolio_delta, net_options, gross_options = self.close_net_hedges(portfolio_delta, net_options, gross_options) #no longer need freed up net positions
                
                # Hedges position #
                if abs(portfolio_delta) >= (DELTA_CORRECTION_THRESHOLD * self.delta_limit):
                    portfolio_delta, stock_position = big_RTM_delta_hedge(portfolio_delta, stock_position)
                    
                portfolio_delta, net_options, gross_options, stock_position = self.delta_hedge_options(portfolio_delta, net_options, gross_options, stock_position)

                portfolio_delta, net_options, gross_options, stock_position = self.close_delta_hedges_options(portfolio_delta, net_options, gross_options, stock_position)
                
                # BELOW JUST DEALS WITH TRIMMING OPTION POSITION WHEN abs(RTM POSITION) = 50k
                portfolio_delta, net_options, gross_options = self.trim_options_position(portfolio_delta, stock_position, net_options, gross_options)  # This is 50 lines of code.

                # print(round(portfolio_delta, 5))
                print("Options position: {}. Stock Position: {}. Gross Position {}. Portfolio Delta {}". format(net_options, stock_position, gross_options, round(portfolio_delta, 5)))
                print(self.net_hedges)
                print(self.delta_hedges)
                print(self.arb_positions)

                # Sleep has been moved to the top of the algorithm for shits and giggles.
            else:
                pass


def main() -> None:
    rit = RIT(X_API_KEY)
    algorithm = Algorithm(rit)

    algorithm.run()

if __name__ == '__main__':
    main()