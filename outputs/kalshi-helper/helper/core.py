"""Deterministic, event-driven paper broker shared by replay and live runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import math


def dec(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal amount") from exc
    if not result.is_finite():
        raise ValueError("Amounts must be finite")
    return result


def stamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def book_midpoint(book):
    """Only positive-quantity levels represent a current quote."""
    yes=[dec(p) for p,q in book['yes'] if dec(q)>0]
    no=[dec(p) for p,q in book['no'] if dec(q)>0]
    return (max(yes)+1-max(no))/2 if yes and no else None


def plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


@dataclass
class Settings:
    bankroll: str = "1000"
    bet_fraction: str = "0.01"
    daily_loss_fraction: str = "0.03"
    session_loss_fraction: str = "0.05"
    event_fraction: str = "0.03"
    total_fraction: str = "0.20"
    min_edge: str = "0.04"
    depth_fraction: str = "0.10"
    delay_seconds: int = 5
    max_book_age_seconds: int = 180
    max_contracts: int = 1000

    def __post_init__(self):
        if not 0 < dec(self.bankroll) <= 100000000:
            raise ValueError("Bankroll must be positive and at most $100 million")
        for name in ("bet_fraction", "daily_loss_fraction", "session_loss_fraction", "event_fraction", "total_fraction", "depth_fraction"):
            if not 0 < dec(getattr(self, name)) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= dec(self.min_edge) < 1:
            raise ValueError("min_edge must be in [0, 1)")
        if self.delay_seconds < 1 or self.max_book_age_seconds < 1:
            raise ValueError("Delay and book age must be positive")
        if not 1 <= self.max_contracts <= 10000:
            raise ValueError("max_contracts must be between 1 and 10000")


def fee_and_cost(price, quantity, rate):
    """Conservative per-level approximation, NOT the exchange's fee accumulator.

    Rate is supplied per market. Round total debit upward to cents, preserving
    Decimal subpenny prices. A level is one simulated child execution.
    """
    gross = price * quantity
    debit = (gross + rate * quantity * price * (1 - price)).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )
    return debit - gross, debit


def validate_record(r):
    kind = r["type"]
    now = stamp(r["available_at"])
    if not isinstance(r.get("ticker"), str) or not r["ticker"]:
        raise ValueError("Every record needs a ticker")
    if kind == "market":
        if not r.get("event_ticker"):
            raise ValueError("Market needs an event_ticker for shared risk limits")
        stamp(r["close_time"])
        if r.get("fee_rate") is not None and not 0 <= dec(r["fee_rate"]) <= 1:
            raise ValueError("Invalid fee rate")
        if r.get("fee_rate") is not None and not r.get("fee_version"):
            raise ValueError("Explicit fee_version required")
    elif kind == "book":
        if 'execution_check' in r:
            check=r['execution_check']
            if not isinstance(check.get('signal'),bool) or not check.get('model_id'):
                raise ValueError('Invalid execution check')
            if not stamp(r['observed_at'])<=stamp(check['made_at'])<=now:
                raise ValueError('Execution check must use this book after arrival')
            if not 0<=dec(check['probability'])<=1:raise ValueError('Invalid execution probability')
        if stamp(r["observed_at"]) > now:
            raise ValueError("A book cannot be available before it was observed")
        for side in ("yes", "no"):
            prices = set()
            for p, q in r[side]:
                p, q = dec(p), dec(q)
                if not 0 <= p <= 1 or q < 0 or p in prices:
                    raise ValueError("Invalid or duplicate book level")
                prices.add(p)
        yes=[dec(p) for p,q in r['yes'] if dec(q)>0]
        no=[dec(p) for p,q in r['no'] if dec(q)>0]
        if yes and no:
            if max(yes) + max(no) > 1:
                raise ValueError("Crossed book")
    elif kind == "prediction":
        if not 0 <= dec(r["probability"]) <= 1:
            raise ValueError("Probability must be in [0, 1]")
        if not stamp(r["features_available_at"]) <= stamp(r["made_at"]) <= now:
            raise ValueError("Prediction uses future information or predates its features")
        if stamp(r["expires_at"]) <= stamp(r["made_at"]):
            raise ValueError("Prediction expiry must follow creation")
        if not r.get("model_id"):
            raise ValueError("Prediction needs a model_id")
        if r.get('signal_side') is not None and r['signal_side'] not in ('yes','no'):
            raise ValueError('Invalid signal side')
        if 'signal_eligible' in r and not isinstance(r['signal_eligible'],bool):raise ValueError('Signal eligibility must be boolean')
    elif kind == "outcome":
        if not 0 <= dec(r["yes_payout"]) <= 1:
            raise ValueError("Settlement payout must be in [0, 1]")
        if stamp(r["settled_at"]) > now:
            raise ValueError("Settlement cannot be known before it occurs")
    else:
        raise ValueError(f"Unknown record type: {kind}")
    return r


class Engine:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.initial = dec(self.settings.bankroll)
        self.cash = self.initial
        self.fees = Decimal(0)
        self.realized = Decimal(0)
        self.markets, self.books, self.positions = {}, {}, {}
        self.pending, self.outcomes, self.predictions = {}, {}, {}
        self.filled_predictions = {}
        self.daily_realized = {}
        self.risk_halted = False
        self.attempted = set()
        self.ledger, self.curve = [], []
        self.now = None

    def log(self, action, ticker, **fields):
        self.ledger.append(plain(dict(at=self.now, action=action, ticker=ticker, **fields)))

    def feed(self, r, allow_orders=True):
        validate_record(r)
        now = stamp(r["available_at"])
        if self.now and now < self.now:
            raise ValueError("Events must arrive in availability-time order")
        self.now = now
        ticker, kind = r["ticker"], r["type"]
        for t, order in list(self.pending.items()):
            if now >= stamp(order["expires_at"]) or not allow_orders:
                self.log("cancel", t, reason="prediction expired or entry window ended")
                del self.pending[t]
        if kind == "market":
            old = self.markets.get(ticker)
            if old and old["event_ticker"] != r["event_ticker"]:
                raise ValueError("Market event identity cannot change")
            self.markets[ticker] = r
        elif kind == "outcome":
            if ticker in self.outcomes:
                if dec(self.outcomes[ticker]["yes_payout"]) != dec(r["yes_payout"]):
                    raise ValueError("Conflicting settlement; manual reconciliation required")
                return
            self.outcomes[ticker] = r
            self.pending.pop(ticker, None)
            position = self.positions.pop(ticker, None)
            if position:
                payout = dec(r["yes_payout"])
                if position["side"] == "no":
                    payout = 1 - payout
                proceeds = position["quantity"] * payout
                profit = proceeds - position["cost"]
                self.cash += proceeds
                self.realized += profit
                day=self.now.date().isoformat()
                self.daily_realized[day]=self.daily_realized.get(day,Decimal(0))+profit
                self.log("settle", ticker, proceeds=proceeds, profit=profit)
        elif kind == "book":
            previous = self.books.get(ticker)
            if previous and stamp(previous["observed_at"]) >= stamp(r["observed_at"]):
                return  # Replayed/stale snapshots never replenish liquidity.
            self.books[ticker] = r
            if ticker in self.pending and allow_orders:
                order = self.pending[ticker]
                age = (stamp(r["observed_at"]) - stamp(order["available_at"])).total_seconds()
                if age >= self.settings.delay_seconds:
                    del self.pending[ticker]
                    self.execute(order, r)
        elif kind == "prediction":
            if ticker in self.outcomes:
                self.log("skip", ticker, reason="outcome already known")
                return
            if now >= stamp(r["expires_at"]):
                self.log("skip", ticker, reason="prediction expired")
                return
            book = self.books.get(ticker)
            baseline = None
            if book and self.fresh(book):baseline=book_midpoint(book)
            # First valid prediction per market avoids overweighting frequent updates.
            self.predictions.setdefault(ticker, dict(r, baseline=baseline))
            self.log("prediction", ticker, probability=r["probability"], model_id=r["model_id"], reason=r.get('explanation',''))
            if r.get('signal_eligible') is False and ticker in self.pending and stamp(r['made_at'])>=stamp(self.pending[ticker]['made_at']):
                self.pending.pop(ticker)
                self.log('cancel',ticker,reason='Newer forecast no longer qualifies at decision time')
            if not allow_orders or ticker in self.attempted or r.get('signal_eligible') is False:
                return
            if ticker in self.pending and stamp(self.pending[ticker]["made_at"]) >= stamp(r["made_at"]):
                return
            self.pending[ticker] = dict(r,baseline=baseline)
            self.log("signal", ticker, reason="awaiting a fresh post-delay book")
        self.mark()

    def fresh(self, book):
        return 0 <= (self.now - stamp(book["observed_at"])).total_seconds() <= self.settings.max_book_age_seconds

    def execute(self, order, book):
        ticker = order["ticker"]
        if order.get('require_execution_check'):
            check=book.get('execution_check')
            if not check or check['model_id']!=order['model_id'] or not check['signal']:
                self.log('skip',ticker,reason='execution_model_unavailable');return
            order=dict(order,decision_probability=order['probability'],probability=check['probability'],
                execution_checked_at=check['made_at'])
        day=self.now.date().isoformat()
        if self.realized<=-self.initial*dec(self.settings.session_loss_fraction):self.risk_halted=True
        if self.risk_halted or self.daily_realized.get(day,Decimal(0))<=-self.initial*dec(self.settings.daily_loss_fraction):
            self.log('skip',ticker,reason='Daily or session realized-loss limit reached');return
        market = self.markets.get(ticker)
        if not market or ticker in self.outcomes or market.get("status") not in ("open", "active"):
            self.log("skip", ticker, reason="market is not open")
            return
        if self.now >= stamp(market["close_time"]) or not self.fresh(book):
            self.log("skip", ticker, reason="closed market or stale book")
            return
        if market.get("fee_rate") is None:
            self.log("skip", ticker, reason="fee assumption missing")
            return
        rate, p = dec(market["fee_rate"]), dec(order["probability"])
        choices = []
        for side, opposing, probability in (("yes", "no", p), ("no", "yes", 1-p)):
            if order.get('signal_side') and side!=order['signal_side']:continue
            asks = sorted((1-dec(price), dec(q)) for price, q in book[opposing] if dec(q) > 0)
            if asks:
                price = asks[0][0]
                choices.append((probability-price-rate*price*(1-price), side, probability, asks))
        if not choices:
            self.log("skip", ticker, reason="empty book")
            return
        _, side, probability, asks = max(choices, key=lambda x: x[0])
        group = market["event_ticker"]
        exposure = sum((x["cost"] for x in self.positions.values()), Decimal(0))
        event_exposure = sum((x["cost"] for x in self.positions.values() if x["event"] == group), Decimal(0))
        budget = min(self.cash, self.initial*dec(self.settings.bet_fraction), self.initial*dec(self.settings.total_fraction)-exposure,
                     self.initial*dec(self.settings.event_fraction)-event_exposure)
        cost, fees, count, levels = Decimal(0), Decimal(0), 0, []
        for price, visible in asks:
            limit = min(int(visible*dec(self.settings.depth_fraction)), self.settings.max_contracts-count)
            selected = 0
            for quantity in range(1, max(0, limit)+1):
                fee, debit = fee_and_cost(price, quantity, rate)
                if debit > budget-cost or probability-debit/quantity < dec(self.settings.min_edge):
                    continue
                selected = quantity
            if selected:
                fee, debit = fee_and_cost(price, selected, rate)
                levels.append(dict(price=price, quantity=selected, fee=fee))
                count += selected
                cost += debit
                fees += fee
        if not count:
            usable=[(price,int(visible*dec(self.settings.depth_fraction))) for price,visible in asks]
            if not any(q>0 for _,q in usable):reason='insufficient_usable_depth'
            elif not any(probability-fee_and_cost(price,1,rate)[1]>=dec(self.settings.min_edge) for price,q in usable if q>0):reason='edge_lost_after_costs'
            else:reason='risk_budget_or_whole_contract_limit'
            self.log("skip", ticker, reason=reason)
            return
        self.attempted.add(ticker)  # One entry per market per run; no recycled depth.
        self.filled_predictions[ticker]=dict(order,execution_baseline=book_midpoint(book))
        self.cash -= cost
        self.fees += fees
        self.positions[ticker] = dict(side=side, quantity=count, cost=cost, event=group,
                                      fee_rate=rate, fee_version=market["fee_version"])
        self.log("fill", ticker, side=side, quantity=count, cost=cost, fees=fees,
                 model_id=order["model_id"], fee_version=market["fee_version"], levels=levels,
                 probability=order['probability'],signal_at=order['available_at'],
                 signal_midpoint=order.get('baseline'),execution_midpoint=book_midpoint(book),
                 signal_age_seconds=(self.now-stamp(order['available_at'])).total_seconds())

    def valuation(self):
        value, unpriced = Decimal(0), 0
        for ticker, pos in self.positions.items():
            remaining = pos["quantity"]
            book = self.books.get(ticker)
            if book and self.fresh(book):
                for price, size in sorted(book[pos["side"]], key=lambda x: dec(x[0]), reverse=True):
                    quantity = min(remaining, int(dec(size)*dec(self.settings.depth_fraction)))
                    if quantity:
                        price = dec(price)
                        fee, _ = fee_and_cost(price, quantity, pos["fee_rate"])
                        value += max(Decimal(0), price*quantity-fee)
                        remaining -= quantity
            unpriced += remaining
        return value, unpriced

    def mark(self):
        value, unpriced = self.valuation()
        row = plain(dict(at=self.now, equity_lower_bound=self.cash+value, unpriced_contracts=unpriced))
        if not self.curve or row != self.curve[-1]:
            self.curve.append(row)

    def finish(self, at=None):
        if at:
            at = stamp(at)
            if self.now and at < self.now:
                raise ValueError("Cannot finish before the last event")
            self.now = at
        for ticker in list(self.pending):
            self.log("cancel", ticker, reason="run ended")
        self.pending.clear()
        self.mark()

    def scores(self, predictions=None, baseline_key='baseline'):
        model, market, groups = [], [], set()
        for ticker, prediction in (self.predictions if predictions is None else predictions).items():
            outcome = self.outcomes.get(ticker)
            if not outcome or dec(outcome["yes_payout"]) not in (0, 1):
                continue
            if stamp(outcome["settled_at"]) <= stamp(prediction["available_at"]):
                continue
            p, y = float(prediction["probability"]), float(outcome["yes_payout"])
            baseline = prediction.get(baseline_key)
            if baseline is None:
                continue  # Paired comparison only.
            def score(prob):
                clipped = min(1-1e-12, max(1e-12, prob))
                return ((prob-y)**2, -(y*math.log(clipped)+(1-y)*math.log(1-clipped)))
            model.append(score(p))
            market.append(score(float(baseline)))
            groups.add(self.markets.get(ticker, {}).get("event_ticker", ticker))
        if not model:
            return dict(paired_markets=0, independent_event_groups=0)
        return dict(paired_markets=len(model), independent_event_groups=len(groups),
                    model_brier=sum(x[0] for x in model)/len(model),
                    market_brier=sum(x[0] for x in market)/len(market),
                    model_log_loss=sum(x[1] for x in model)/len(model),
                    market_log_loss=sum(x[1] for x in market)/len(market))

    def report(self):
        value, unpriced = self.valuation()
        peak, drawdown = self.initial, Decimal(0)
        for row in self.curve:
            equity = dec(row["equity_lower_bound"])
            peak = max(peak, equity)
            drawdown = max(drawdown, peak-equity)
        fills=[r for r in self.ledger if r['action']=='fill']
        settled={r['ticker'] for r in self.ledger if r['action']=='settle'}
        deployed=sum((dec(r['cost']) for r in fills),dec(0))
        settled_cost=sum((dec(r['cost']) for r in fills if r['ticker'] in settled),dec(0))
        settled_quantity=sum(r['quantity'] for r in fills if r['ticker'] in settled)
        return plain(dict(settings=asdict(self.settings), as_of=self.now,
                         capital_deployed=deployed,settled_capital_deployed=settled_cost,
                         risk_halted=self.risk_halted,daily_realized=self.daily_realized,
                         settled_contracts=settled_quantity,
                         realized_net_per_contract=self.realized/settled_quantity if settled_quantity else None,
                         realized_return_on_deployed=self.realized/settled_cost if settled_cost else None,
                         initial_bankroll=self.initial, cash=self.cash,
                         realized_pnl=self.realized, fees_paid=self.fees,
                         open_cost=sum((x["cost"] for x in self.positions.values()), Decimal(0)),
                         estimated_liquidation_value=value, unpriced_contracts=unpriced,
                         equity_lower_bound=self.cash+value,
                         pnl_lower_bound=self.cash+value-self.initial,
                         max_drawdown_lower_bound=drawdown,
                         positions=self.positions, pending_orders=len(self.pending),
                         fills=sum(x["action"] == "fill" for x in self.ledger),
                         scores=self.scores(), ledger=self.ledger, curve=self.curve,
                         score_cohorts=dict(first_prediction=self.scores(),
                             filled_signal_vs_decision_market=self.scores(self.filled_predictions),
                             filled_signal_vs_execution_market=self.scores(self.filled_predictions,'execution_baseline')),
                         score_definition='Headline scores use the first prediction per market. Filled-signal cohorts use the actual order probability; execution comparison measures stale-signal performance. All are paired resolved-market averages, not independent trade evidence.',
                         limitations=["Paper simulation; no real orders.",
                           "Fees are conservative per-level estimates, not exact exchange accounting.",
                           "Whole-contract entries, one entry per market, hold to settlement.",
                           "Missing/stale liquidation depth is valued at zero in the equity lower bound.",
                           "Snapshots do not establish actual executable fills or queue priority."]))
