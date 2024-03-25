from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import inf
from multiprocessing import Process
from requests import HTTPError
from time import sleep
from typing import Any
from warnings import warn

import numpy as np
from ritc import Case, Order, RIT, Security
from sklearn.linear_model import LinearRegression

X_API_KEY: str = 'JT0ZLPKW'


@dataclass
class Algorithm:
    rit: RIT

    def call_safely(
            self,
            function: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        try:
            return function(*args, **kwargs)
        except HTTPError as error:
            warn(f'error response received \'{error.response.json()}\'')

        return None

    def call_silently(
            self,
            function: Callable[..., Any],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        try:
            return function(*args, **kwargs)
        except HTTPError:
            pass

        return None

    def close(self, ticker: str) -> None:
        while True:
            case = self.rit.get_case()

            if case.status == Case.Status.ACTIVE:
                security = self.rit.get_securities(ticker=ticker)[0]
                bid_quantity = min(security.max_trade_size, -security.position)
                ask_quantity = min(security.max_trade_size, security.position)

                if bid_quantity > 0:
                    self.call_safely(
                        self.rit.post_orders,
                        ticker=ticker,
                        type=Order.Type.MARKET,
                        quantity=bid_quantity,
                        action=Order.Action.BUY,
                    )

                if ask_quantity > 0:
                    self.call_safely(
                        self.rit.post_orders,
                        ticker=ticker,
                        type=Order.Type.MARKET,
                        quantity=ask_quantity,
                        action=Order.Action.SELL,
                    )

                self.call_safely(self.rit.post_commands_cancel, ticker=ticker)
                # sleep(1 / security.api_orders_per_second)
                sleep(0.35)

    def make_market(self, ticker: str) -> None:
        def get_spread() -> tuple[float, float, float]:
            bid_price = book.bids[0].price
            ask_price = book.asks[0].price
            competing_bid_prices = set()
            competing_ask_prices = set()
            quantity = security.max_trade_size // 2

            for order in book.bids:
                if order.trader_id == trader.trader_id:
                    continue

                competing_bid_prices.add(order.price)
                quantity -= order.quantity - order.quantity_filled

                if quantity <= 0:
                    bid_price = order.price
                    break

            quantity = security.max_trade_size // 2

            for order in book.asks:
                if order.trader_id == trader.trader_id:
                    continue

                competing_ask_prices.add(order.price)
                quantity -= order.quantity - order.quantity_filled

                if quantity <= 0:
                    ask_price = order.price
                    break

            while bid_price in competing_bid_prices:
                bid_price += 0.01

            while ask_price in competing_ask_prices:
                ask_price -= 0.01

            spread = ask_price - bid_price

            return spread, round(bid_price, 2), round(ask_price, 2)

        while True:
            case = self.rit.get_case()
            trader = self.rit.get_trader()

            if case.status == Case.Status.ACTIVE:
                security = self.rit.get_securities(ticker=ticker)[0]
                bid_quantity = security.max_trade_size // 2 - security.position
                ask_quantity = security.max_trade_size // 2 + security.position
                book = self.rit.get_securities_book(ticker=ticker)

                if not book.bids or not book.asks:
                    continue

                spread, bid_price, ask_price = get_spread()
                quantity = bid_quantity

                for order in book.bids:
                    if order.trader_id != trader.trader_id \
                            or order.price != bid_price:
                        continue

                    quantity -= order.quantity - order.quantity_filled

                if quantity > 0 and spread > security.trading_fee:
                    self.call_safely(
                        self.rit.post_orders,
                        ticker=ticker,
                        type=Order.Type.LIMIT,
                        quantity=min(
                            security.max_trade_size,
                            max(security.min_trade_size, quantity),
                        ),
                        action=Order.Action.BUY,
                        price=bid_price,
                    )

                quantity = ask_quantity

                for order in book.asks:
                    if order.trader_id != trader.trader_id \
                            or order.price != ask_price:
                        continue

                    quantity -= order.quantity - order.quantity_filled

                if quantity > 0 and spread > security.trading_fee:
                    self.call_safely(
                        self.rit.post_orders,
                        ticker=ticker,
                        type=Order.Type.LIMIT,
                        quantity=min(
                            security.max_trade_size,
                            max(security.min_trade_size, quantity),
                        ),
                        action=Order.Action.SELL,
                        price=ask_price,
                    )

                if bid_quantity <= 0:
                    self.call_safely(
                        self.rit.post_commands_cancel,
                        query=f'Ticker=\'{ticker}\' AND Volume>0',
                    )
                else:
                    self.call_safely(
                        self.rit.post_commands_cancel,
                        query=f'Ticker=\'{ticker}\' AND Volume>0 '
                              f'AND Price<>{bid_price}',
                    )

                if ask_quantity <= 0:
                    self.call_safely(
                        self.rit.post_commands_cancel,
                        query=f'Ticker=\'{ticker}\' AND Volume<0',
                    )
                else:
                    self.call_safely(
                        self.rit.post_commands_cancel,
                        query=f'Ticker=\'{ticker}\' AND Volume<0 '
                              f'AND Price<>{ask_price}',
                    )

    def take_arbitrage(self, ticker: str) -> None:
        def get_lease_id() -> int:
            lease_id = None

            while lease_id is None:
                leases = self.rit.get_leases()

                for lease in leases:
                    if lease.ticker == ticker:
                        lease_id = lease.id
                        break

                if lease_id is None:
                    self.call_safely(self.rit.post_leases, ticker=ticker)
                    sleep(1)

            return lease_id

        def get_bid_and_ask_prices(security: Security) -> tuple[float, float]:
            bid_price = 1.0
            ask_price = 1.0

            while security.is_tradeable:
                bid_price *= security.bid - security.trading_fee
                ask_price *= security.ask + security.trading_fee
                security = securities[security.currency]

            return bid_price, ask_price

        while True:
            case = self.rit.get_case()

            if case.status == Case.Status.ACTIVE:
                asset = self.rit.get_assets(ticker=ticker)[0]
                lease_id = get_lease_id()
                securities = {}

                for security in self.rit.get_securities():
                    securities[security.ticker] = security

                total_from_price = 0.0
                total_to_price = 0.0
                min_from_position = inf
                max_from_position = -inf
                min_to_position = inf
                max_to_position = -inf
                kwargs = dict[str, Any]()

                for i, ticker_quantity in enumerate(asset.convert_from):
                    security = securities[ticker_quantity.ticker]
                    price = get_bid_and_ask_prices(security)[1]
                    total_from_price += ticker_quantity.quantity * price

                    if security.type == Security.Type.STOCK:
                        min_from_position = min(min_from_position, security.position)
                        max_from_position = max(max_from_position, security.position)

                    kwargs[f'from{i + 1}'] = ticker_quantity.ticker
                    kwargs[f'quantity{i + 1}'] = ticker_quantity.quantity

                for ticker_quantity in asset.convert_to:
                    security = securities[ticker_quantity.ticker]
                    price = get_bid_and_ask_prices(security)[0]
                    total_to_price += ticker_quantity.quantity * price

                    if security.type == Security.Type.STOCK:
                        min_to_position = min(min_to_position, security.position)
                        max_to_position = max(max_to_position, security.position)

                if total_from_price < total_to_price and (
                    (max_from_position > 20000 or min_to_position < -20000) \
                    and max_to_position + 10000 < min_from_position - 10000):
                    lease = self.call_safely(
                        self.rit.post_leases,
                        id=lease_id,
                        **kwargs,
                    )

                    if hasattr(lease, 'convert_finish_tick'):
                        sleep(max(0, lease.convert_finish_tick - case.tick))

    def take_tenders(self) -> None:
        def collapse(quantity: float, orders: Sequence[Order]) -> float:
            for order in orders:
                if order.trader_id == trader.trader_id:
                    continue

                quantity -= order.quantity - order.quantity_filled

                if quantity <= 0:
                    return order.price

            return orders[-1].price

        while True:
            case = self.rit.get_case()
            trader = self.rit.get_trader()

            if case.status == Case.Status.ACTIVE:
                tenders = self.rit.get_tenders()
                securities = {}
                books = {}
                histories = {}

                for security in self.rit.get_securities():
                    securities[security.ticker] = security
                    books[security.ticker] = self.rit.get_securities_book(
                        ticker=security.ticker,
                        limit=1000,
                    )
                    histories[security.ticker] = self.rit.get_securities_history(
                        ticker=security.ticker,
                        limit=10,
                    )

                for tender in tenders:
                    security = securities[tender.ticker]
                    book = books[tender.ticker]

                    if len(book.bids) < 10 or len(book.asks) < 10 \
                            or case.tick < 10:
                        continue

                    status = False
                    bid_price = collapse(security.max_trade_size // 2, book.bids)
                    ask_price = collapse(security.max_trade_size // 2, book.asks)
                    ticks = \
                        [history.tick for history in histories[tender.ticker]]
                    prices = \
                        [history.close for history in histories[tender.ticker]]
                    model = LinearRegression().fit(
                        np.array(ticks).reshape((-1, 1)),
                        prices,
                    )

                    if tender.action == Order.Action.BUY:
                        status = (model.coef_.item() > 0.01 and \
                            round(tender.price + 2 * security.trading_fee, 2) \
                            <= ask_price) or \
                            (model.coef_.item() > -0.005 and round(tender.price + 0.1, 2) <= ask_price)
                    elif tender.action == Order.Action.SELL:
                        status = (model.coef_.item() < -0.01 and \
                            round(tender.price - 2 * security.trading_fee, 2) \
                            >= bid_price) or \
                            (model.coef_.item() < 0.005 and round(tender.price - 0.1, 2) >= bid_price)
                    else:
                        warn(f'unknown tender {tender}')

                    if status:
                        self.rit.post_tenders(tender.tender_id)

    def run(self) -> None:
        processes = []

        for security in self.rit.get_securities():
            if security.is_tradeable:
                if security.type == Security.Type.CURRENCY:
                    processes.append(
                        Process(
                            target=self.close,
                            args=(security.ticker,),
                        ),
                    )
                elif security.type == Security.Type.STOCK:
                    processes.append(
                        Process(
                            target=self.make_market,
                            args=(security.ticker,),
                        ),
                    )
                else:
                    warn('unknown security \'{security}\'')

        # for asset in self.rit.get_assets():
        #     if asset.is_available:
        #         if asset.type == Asset.Type.REFINERY:
        #             processes.append(
        #                 Process(
        #                     target=self.take_arbitrage,
        #                     args=(asset.ticker,),
        #                 ),
        #             )
        #         else:
        #             warn('unknown asset \'{asset}\'')

        processes.append(Process(target=self.take_tenders))

        for process in processes:
            process.start()

        for process in processes:
            process.join()


def main() -> None:
    rit = RIT(X_API_KEY)
    algorithm = Algorithm(rit)

    algorithm.run()


if __name__ == '__main__':
    main()
